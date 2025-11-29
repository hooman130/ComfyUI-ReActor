import os
import shutil
from typing import List, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import insightface
from insightface.app.common import Face
# try:
#     import torch.cuda as cuda
# except:
#     cuda = None
import torch

import folder_paths
import comfy.model_management as model_management
from modules.shared import state

# 1. Добавьте импорт logging наверху файла, если его там нет:
import logging
from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
    progress_bar,
    progress_bar_reset
)
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# ---------------------------
# Helper describe functions for richer logging
# ---------------------------
def _describe_ndarray(arr: np.ndarray, name="array") -> str:
    try:
        return f"{name}(shape={arr.shape}, dtype={arr.dtype}, min={arr.min() if arr.size else 'NA'}, max={arr.max() if arr.size else 'NA'})"
    except Exception:
        return f"{name}(shape={getattr(arr,'shape','?')}, dtype={getattr(arr,'dtype','?')})"

def _describe_pil(img: Image.Image, name="image") -> str:
    try:
        return f"{name}(size={img.size}, mode={img.mode})"
    except Exception:
        return f"{name}(type=PIL.Image)"

def _describe_face(face: Face, name="face") -> str:
    try:
        bbox = getattr(face, "bbox", None)
        kps = getattr(face, "kps", None)
        l5 = getattr(face, "landmark_5", None)
        lm = getattr(face, "landmark", None)
        sex = getattr(face, "sex", None)
        emb = getattr(face, "normed_embedding", None)
        emb_shape = None
        if isinstance(emb, np.ndarray):
            emb_shape = emb.shape
        return f"{name}(bbox={bbox}, sex={sex}, has_kps={kps is not None}, has_landmark_5={l5 is not None}, has_landmark={lm is not None}, emb_shape={emb_shape})"
    except Exception as e:
        return f"{name}(error_describing: {e})"

def _describe_obj(obj, name="obj") -> str:
    if obj is None:
        return f"{name}=None"
    if isinstance(obj, np.ndarray):
        return _describe_ndarray(obj, name)
    if isinstance(obj, Image.Image):
        return _describe_pil(obj, name)
    if isinstance(obj, Face):
        return _describe_face(obj, name)
    if isinstance(obj, tuple):
        return f"{name}=tuple(len={len(obj)}, types={[type(x).__name__ for x in obj]})"
    if isinstance(obj, list):
        return f"{name}=list(len={len(obj)}, types={[type(x).__name__ for x in obj[:3]]}{'...' if len(obj)>3 else ''})"
    return f"{name}=<{type(obj).__name__}>"

def _safe_is_tuple2(x) -> bool:
    return isinstance(x, tuple) and len(x) == 2

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]
# if cuda is not None:
#     if cuda.is_available():
#         providers = ["CUDAExecutionProvider"]
#     else:
#         providers = ["CPUExecutionProvider"]
# else:
#     providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")
hyperswap_path = os.path.join(models_path, "hyperswap")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)
        # Извлекаем имя файла модели из пути
        model_filename = os.path.basename(model_path)
        
        # Определяем правильный путь в зависимости от типа модели
        if "hyperswap" in model_filename.lower():
            # Ищем в директории hyperswap
            correct_path = os.path.join(folder_paths.models_dir, "hyperswap", model_filename)
            FS_MODEL = ort.InferenceSession(correct_path, providers=providers)
        elif "reswapper" in model_filename.lower():
            # Ищем в директории reswapper
            correct_path = os.path.join(folder_paths.models_dir, "reswapper", model_filename)
            FS_MODEL = insightface.model_zoo.get_model(correct_path, providers=providers)
        else:
            # Для моделей insightface используем оригинальный путь
            FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
        logger.debug("Loaded FaceSwap model: path=%s, type=%s, providers=%s", model_path, type(FS_MODEL).__name__, providers)
    return FS_MODEL
    
# Функция для получения 5 ключевых точек из объекта Face
def get_landmarks_5(face):
    # face.landmark_5: np.ndarray shape (5,2)
    # Если нет, попробуй face.kps или face.landmark
    if hasattr(face, 'landmark_5') and face.landmark_5 is not None:
        logger.debug("landmark_5: %s", face.landmark_5)
        return face.landmark_5
    elif hasattr(face, 'kps') and face.kps is not None:
        logger.debug("kps: %s", face.kps)
        return face.kps
    elif hasattr(face, 'landmark') and face.landmark is not None:
        # 68-точечная разметка, берём нужные индексы
        # Иногда landmark shape (68,2) — тогда возьми нужные точки
        # Пример: [36, 45, 30, 48, 54] — левый/правый глаз, нос, левый/правый рот
        if face.landmark.shape[0] >= 68:
            idxs = [36, 45, 30, 48, 54]
            logger.debug("landmark (68 точек): %s", face.landmark[idxs])
            return face.landmark[idxs]
    logger.warning("Нет подходящих точек в объекте Face. Доступные атрибуты: %s", dir(face))
    return None
    
#### Что проверить:
# В логах должны быть координаты точек, например:
# DEBUG:reactor_swapper: landmark_5: [[100 120] [150 125] [125 160] [105 190] [145 190]]
# Если точки отрицательны или за пределами изображения — это ошибка в `M`.    

# Функция для вычисления аффинного преобразования
def get_affine_transform(src_pts, dst_pts):
    # src_pts, dst_pts: np.ndarray shape (5,2)
    # OpenCV требует 3 точки, но можно использовать estimateAffinePartial2D для 5
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M

# Создаём градиентную маску овальной формы без обрезки
# 2. Убедитесь, что эллипс **не выходит** за пределы 256×256  
# Если эллипс "выпирает" за 256×256, `BORDER_CONSTANT` все равно создаст артефакты. Сократите размер эллипса, чтобы он пол�[...]
def create_gradient_mask(crop_size=256):
    # 1. Создаём пустую маску (все пиксели = 0)
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    
    # 2. Определяем центр и размеры эллипса (ИСПРАВЛЕНО: ещё меньше радиусов)
    center = (crop_size // 2, crop_size // 2)
    axes = (int(crop_size * 0.35), int(crop_size * 0.4))  # Уменьшили радиусы; Горизонтальный и вертикальный радиус
    
    # 3. Рисуем эллипс (заполняем белым цветом, значение=1.0)
    cv2.ellipse(
        mask,          # Массив для рисования
        center,        # Центр эллипса
        axes,          # Полуоси (ширина, высота)
        angle=0,       # Угол поворота
        startAngle=0,  # Начальный угол дуги
        endAngle=360,  # Конечный угол дуги (360 = полный эллипс)
        color=1.0,     # Значение для заполнения (белый = 1.0)
        thickness=-1   # -1 = заполнить всю область эллипса   
    )
    
    # 4. Применяем размытие для плавных краёв
    blur_ksize = 15  # Нечётное число, чтобы ядро было симметричным
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    
    # 5. Ограничим значения в диапазоне [0, 1]
    mask = np.clip(mask, 0, 1)
    
    return mask    
    
#### 1. Используйте `cv2.BORDER_TRANSPARENT` (OpenCV ≥ 4.5)  
# Этот флаг позволяет **не заполнять** области за пределами маски никаким цветом (пиксели остаются `0` или "прозр[...]
def paste_back(target_img, swapped_face, M, crop_size=256):
    # Улучшенная функция paste_back с идеальной овальной маской и исправлениями артефактов
    
    # target_img: Исходное изображение (BGR, numpy, uint8)
    # swapped_face: Результат работы модели (256x256, BGR, uint8)
    # M: Матрица аффинного преобразования (Target -> Crop), но здесь используется M_inv из run_hyperswap
    # crop_size: Размер кропа (для HyperSwap это 256)

    logger.debug("paste_back inputs: %s; %s; M=%s", _describe_obj(target_img, "target_img"), _describe_obj(swapped_face, "swapped_face"), M)

    if M is None:
        logger.error("paste_back called with M=None; skipping blend")
        return target_img
    
    # 1. Создание мягкой маски (Эрозия + Размытие)
    mask = create_gradient_mask(crop_size)
    
    # Преобразуем в трехканальную маску
    mask_3c = np.stack([mask] * 3, axis=2)
    
    # 2. Получаем размеры целевого изображения
    h, w = target_img.shape[:2]
    
    # 3. Нормализация swapped_face к float32 [0,1] для warp
    swapped_face_norm = swapped_face.astype(np.float32) / 255.0
    mask_norm = mask_3c.astype(np.float32)  # Маска уже [0,1]
    
    # 4. Обратное преобразование (WARP_INVERSE_MAP) для лица И маски
    # Используем BORDER_CONSTANT с borderValue=0.5 (серый, чтобы избежать синих/зеленых артефактов)
    warped_face = cv2.warpAffine(
        swapped_face_norm,
        M,  # Это M_inv из run_hyperswap
        (w, h),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.5  # Серый фон вместо черного/белого
    )
    
    warped_mask = cv2.warpAffine(
        mask_norm,
        M,  # Это M_inv из run_hyperswap
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0  # Маска: 0 за пределами
    )
    
    # 5. Обработка после warp: Clip, NaN fix
    warped_face = np.clip(warped_face, 0, 1)  # Убираем отрицательные
    warped_face = np.nan_to_num(warped_face, nan=0.5)  # NaN -> серый
    
    warped_mask = np.clip(warped_mask, 0, 1)
    warped_mask = np.nan_to_num(warped_mask, nan=0.0)
    
    # 6. Дополнительное размытие для устранения артефактов (опционально, но помогает)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    
    # Отладочные логи (добавьте после warp)
    logger.debug("Warped face shape: %s | Min: %s | Max: %s | NaN count: %s", 
                 warped_face.shape, warped_face.min(), warped_face.max(), np.isnan(warped_face).sum())
    logger.debug("Warped mask shape: %s | Min: %s | Max: %s | NaN count: %s", 
                 warped_mask.shape, warped_mask.min(), warped_mask.max(), np.isnan(warped_mask).sum())
    
    # 7. Плавное наложение в float32
    target_float = target_img.astype(np.float32) / 255.0
    result_float = target_float * (1.0 - warped_mask) + warped_face * warped_mask
    
    # 8. Обратная нормализация к uint8
    result = (result_float * 255).clip(0, 255).astype(np.uint8)
    
    logger.debug("Final result: shape %s | Min: %s | Max: %s", result.shape, result.min(), result.max())
    
    return result
    
#### Что проверить:
# `"Warped Face"` должен содержать лицо в правильном положении.
# `"Warped Mask"` — маска должна быть градиентной, а не полностью черной или белой.

#### 1. **Логирование точек и матрицы**    
def visualize_points(img, points, color=(0, 255, 0)):
    img = img.copy()
    if points is None:
        logger.debug("visualize_points called with points=None")
        return
    for p in points:
        cv2.circle(img, tuple(p.astype(int)), 3, color, -1)
  # cv2.imshow("Face Points", img)
  # cv2.waitKey(1)    

# Итоговая функция run_hyperswap с аффинным преобразованием
def run_hyperswap(session, source_face, target_face, target_img):
    logger.debug("run_hyperswap(): session=%s; %s; %s; %s",
                 type(session).__name__,
                 _describe_face(source_face, "source_face"),
                 _describe_face(target_face, "target_face"),
                 _describe_ndarray(target_img, "target_img") if isinstance(target_img, np.ndarray) else _describe_obj(target_img, "target_img"))

    # 1. Подготовка эмбеддинга
    try:
        source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)
        logger.debug("_source_embedding: shape=%s dtype=%s min=%.6f max=%.6f",
                     source_embedding.shape, source_embedding.dtype, float(source_embedding.min()), float(source_embedding.max()))
    except Exception as e:
        logger.error("Failed to prepare source embedding: %s", e)
        return None, None

    # 2. Получаем 5 точек target
    target_landmarks_5 = get_landmarks_5(target_face)
    visualize_points(target_img, target_landmarks_5, (0, 255, 0))  # Зеленые точки
    
    if target_landmarks_5 is None:
        logger.error("Не удалось получить 5 точек для целевого лица")
        # Важно: Если ошибка, возвращаем None и исходную матрицу (или обрабатываем ошибку иначе)
        return None, None

    # 3. Определение эталонных точек для выравнивания 256x256 (FFHQ Alignment)
    std_landmarks_256 = np.array([
        [ 84.87, 105.94],  # Левый глаз
        [171.13, 105.94],  # Правый глаз
        [128.00, 146.66],  # Кончик носа
        [ 96.95, 188.64],  # Левый уголок рта
        [159.05, 188.64]   # Правый уголок рта
    ], dtype=np.float32)

    # Вычисляем аффинную матрицу
    M = get_affine_transform(target_landmarks_5.astype(np.float32), std_landmarks_256)
    logger.debug("Affine Matrix M (used for cropping):\n%s", M)
    if M is None or not np.isfinite(M).all():
        logger.error("Affine matrix M invalid (None or contains NaN/Inf): %s", M)
        return None, None

#### Что проверить:
# Матрица `M` не должна содержать `NaN` или бесконечности.
# Если матрица нулевая или искаженная — проблема в точках `target_landmarks_5`.
    
    # Применяем аффинное преобразование с новой матрицей M
    try:
        crop = cv2.warpAffine(target_img, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    except Exception as e:
        logger.error("warpAffine (crop) failed: %s", e)
        return None, None

    # Визуализация crop перед инференсом
    #### Что проверить:
    # Окно `"Crop Before Inference"` должно показывать лицо, вырезанное по аффинному преобразованию.
    # Если изображение черное — проблема в `M` или `target_landmarks_5`.
    logger.debug("Crop shape: %s | Min: %s | Max: %s", crop.shape, crop.min(), crop.max())
  # cv2.imshow("Crop Before Inference", crop)
  # cv2.waitKey(1)  # Отображает изображение

    # 4. Преобразуем crop для модели
    # crop_input = crop[:, :, ::-1] / 255.0
    # crop_input = (crop_input - 0.5) / 0.5
    # crop_input = crop_input.transpose(2, 0, 1)
    # crop_input = np.expand_dims(crop_input, axis=0).astype(np.float32)
    crop_input = crop[:, :, ::-1].astype(np.float32) / 255.0  # RGB -> [0,1]
    crop_input = (crop_input - 0.5) / 0.5  # Нормализация
    crop_input = crop_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    logger.debug("crop_input: %s", _describe_ndarray(crop_input, "crop_input"))

    # 5. Инференс
    try:
        output = session.run(None, {'source': source_embedding, 'target': crop_input})[0][0]
        logger.debug("Model output shape: %s | Min: %s | Max: %s", output.shape, output.min(), output.max())
    except Exception as e:
        logger.error("Ошибка выполнения модели: %s", e)
        # IMPORTANT: keep behavior consistent with old code path, but return a tuple so callers can handle without TypeError
        return None, None

    # --- CPU FLOAT NORMALIZATION FIX ---
    # предотвращает появление "синей кожи" и "шума" при работе на CPU
    # (адаптировано из патча patch_cpu_fix.diff)
    if isinstance(output, np.ndarray):
        # устранение NaN и бесконечностей
        output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
        
        # если диапазон похож на [-1,1] → нормализуем в [0,255]
        if output.min() < 0.0 or output.max() <= 1.5:
            output = ((output + 1.0) / 2.0 * 255.0)
        
        # жёсткое ограничение диапазона и тип для OpenCV
        output = np.clip(output, 0, 255).astype(np.uint8).copy()
        
        # защита от повторного использования буфера (inplace CPU bug)
        try:
            output.setflags(write=True)
        except Exception:
            pass

    # 6. Обратная нормализация (теперь output уже uint8, просто transpose и BGR)
    # (ваш код без изменений, но без старой денормализации)
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    output = output[:, :, ::-1]  # RGB -> BGR (Убедитесь, что это BGR, если вход был BGR)
    logger.debug("Output after denormalization: Min: %s | Max: %s", output.min(), output.max())
    
    # Визуализация после денормализации
    #### Что проверить:
    # `output` должен быть в диапазоне `[0..255]` и содержать лицо.
    # Если `output` черный — проблема в нормализации/денормализации или в самой модели.
    logger.debug("Output after denormalization: Min: %s | Max: %s", output.min(), output.max())
  # cv2.imshow("Output After Denormalization", output)
  # cv2.waitKey(1)
       
    return output, M # Возвращаем лицо (256x256) и матрицу M    

def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # if order == "large-small":
    #     return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        order: str,
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    # If index is outside of bounds, return None, avoid exception
    if face_index >= len(gender):
        logger.status("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(gender))
        return None, 0
    face_gender = gender[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 1
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    logger.debug("analyze_faces: det_size=%s, %s", det_size, _describe_ndarray(img_data, "img_data"))
    face_analyser = getAnalysisModel(det_size)

    faces = []
    try:
        faces = face_analyser.get(img_data)
        logger.debug("analyze_faces: found %d face(s)", len(faces))
        if faces:
            sample = faces[0]
            logger.debug("analyze_faces: sample %s", _describe_face(sample, "face[0]"))
    except Exception as e:
        logger.error("analyze_faces error: %s", e)

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        selected_face = faces_sorted[face_index]
        logger.debug("Выбрано лицо: bbox=%s, landmark_5=%s, kps=%s, landmark=%s",
                     selected_face.bbox,
                     hasattr(selected_face, "landmark_5"),
                     hasattr(selected_face, "kps"),
                     hasattr(selected_face, "landmark"))
        return selected_face, 0
        return faces_sorted[face_index], 0
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    # Rich entry logging
    logger.debug("swap_face() called with model=%s, faces_index=%s, source_faces_index=%s, face_boost_enabled=%s",
                 model, faces_index, source_faces_index, face_boost_enabled)
    logger.debug("swap_face inputs: %s; %s; face_model=%s",
                 _describe_obj(source_img, "source_img"),
                 _describe_obj(target_img, "target_img"),
                 _describe_obj(face_model, "face_model"))
    
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            logger.debug("Decoding source_img from base64 string (len=%d)", len(source_img))
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        logger.debug("Converted target_img to BGR numpy: %s", _describe_ndarray(target_img, "target_img"))

        if source_img is not None:

            source_img_np = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            logger.debug("Converted source_img to BGR numpy: %s", _describe_ndarray(source_img_np, "source_img_np"))

            source_image_md5hash = get_image_md5hash(source_img_np)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img_np)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")
            return result_image  # early return to avoid further errors

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            logger.debug("Target faces detected: %d", len(target_faces) if target_faces is not None else -1)

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img_np, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                logger.debug("Using source_face: %s (wrong_gender=%s)", _describe_face(source_face, "source_face"), src_wrong_gender)
                result = target_img
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)    
                else:
                    model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)
                logger.debug("face_swapper: %s", type(face_swapper).__name__)

                source_face_idx = 0

                for face_num in faces_index:
                    logger.debug("Processing target face index: %d", face_num)
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img_np, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                        logger.debug("Switched to next source_face_idx=%d: %s (wrong_gender=%s)",
                                     source_face_idx, _describe_obj(source_face, "source_face"), src_wrong_gender)
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        logger.debug("Selected target_face: %s (wrong_gender=%s)", _describe_obj(target_face, "target_face"), wrong_gender)
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if "hyperswap" in model:
                                # Safe call with robust logging to avoid unpack TypeError
                                res = run_hyperswap(face_swapper, source_face, target_face, result)
                                logger.debug("run_hyperswap returned: %s", _describe_obj(res, "res"))
                                if _safe_is_tuple2(res):
                                    swapped_face_256, M = res
                                    if swapped_face_256 is not None:
                                        try:
                                            result = paste_back(result, swapped_face_256, M, crop_size=256)
                                        except Exception as e:
                                            logger.error("paste_back failed: %s", e)
                                    else:
                                        logger.error("run_hyperswap returned (None, M=%s) — skipping paste_back", M)
                                else:
                                    # Unexpected return type; log and skip to next face
                                    if isinstance(res, np.ndarray):
                                        logger.error("run_hyperswap returned np.ndarray instead of (face, M): %s", _describe_ndarray(res, "res"))
                                    elif isinstance(res, Image.Image):
                                        logger.error("run_hyperswap returned PIL.Image instead of (face, M): %s", _describe_pil(res, "res"))
                                    else:
                                        logger.error("run_hyperswap returned unexpected %s; skipping", type(res).__name__)
                            elif face_boost_enabled:
                                logger.status(f"Face Boost is enabled")
                                ret = None
                                try:
                                    ret = face_swapper.get(result, target_face, source_face, paste_back=False)
                                except Exception as e:
                                    logger.error("face_swapper.get (boost mode) failed: %s", e)
                                    ret = None
                                logger.debug("face_swapper.get (boost) returned: %s", _describe_obj(ret, "ret"))
                                if isinstance(ret, tuple) and len(ret) >= 2:
                                    bgr_fake, M = ret[0], ret[1]
                                else:
                                    bgr_fake, M = ret, None
                                if bgr_fake is None:
                                    logger.error("Boost: bgr_fake is None; skipping")
                                else:
                                    try:
                                        bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                        if M is not None:
                                            M *= scale
                                        else:
                                            logger.warning("Boost: M is None after get(); paste will not be geometrically aligned perfectly.")
                                        result = swapper.in_swap(target_img, bgr_fake, M)
                                    except Exception as e:
                                        logger.error("Boost restore/swap failed: %s", e)
                            else:
                                # logger.status(f"Swapping as-is")
                                try:
                                    tmp = face_swapper.get(result, target_face, source_face)
                                    logger.debug("face_swapper.get returned: %s", _describe_obj(tmp, "tmp"))
                                    result = tmp
                                except Exception as e:
                                    logger.error("face_swapper.get failed: %s", e)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            # Keep searching for other faces if wrong gender is detected, enhancement
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        # Keep searching for other faces if wrong gender is detected, enhancement
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    else:
        logger.debug("swap_face called with model=None; returning original target_img")
    return result_image

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    logger.debug("swap_face_many(): n_targets=%d, model=%s, faces_index=%s, source_faces_index=%s",
                 len(target_imgs), model, faces_index, source_faces_index)
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    result_images = target_imgs

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            logger.debug("Decoding source_img from base64 string (len=%d)", len(source_img))
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]
        logger.debug("Converted all target_imgs to BGR numpy; count=%d", len(target_imgs))

        if source_img is not None:

            source_img_np = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            logger.debug("Converted source_img to BGR numpy: %s", _describe_ndarray(source_img_np, "source_img_np"))

            source_image_md5hash = get_image_md5hash(source_img_np)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img_np)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")
            return result_images

        if source_faces is not None:

            target_faces = []
            pbar = progress_bar(len(target_imgs))

            if len(TARGET_IMAGE_LIST_HASH) > 0:
                logger.status(f"Using Hashed Target Face(s) Model...")
            else:
                logger.status(f"Analyzing Target Image...")

            for i, target_img in enumerate(target_imgs):
                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    break
                
                target_image_md5hash = get_image_md5hash(target_img)
                if len(TARGET_IMAGE_LIST_HASH) == 0:
                    TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
                    target_image_same = False
                elif len(TARGET_IMAGE_LIST_HASH) == i:
                    TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
                    target_image_same = False
                else:
                    target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
                    if not target_image_same:
                        TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash
                
                logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
                logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

                if len(TARGET_FACES_LIST) == 0:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST = [target_face]
                elif len(TARGET_FACES_LIST) == i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST.append(target_face)
                elif len(TARGET_FACES_LIST) != i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST[i] = target_face
                elif target_image_same:
                    # logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
                    target_face = TARGET_FACES_LIST[i]
                

                # logger.status(f"Analyzing Target Image {i}...")
                # target_face = analyze_faces(target_img)
                if target_face is not None:
                    target_faces.append(target_face)
                else:
                    logger.debug("(Image %s) analyze_faces returned None", i)
                
                pbar.update(1)

            progress_bar_reset(pbar)

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img_np, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                results = target_imgs
                logger.debug("swap_face_many using source_face: %s (wrong_gender=%s)", _describe_obj(source_face, "source_face"), src_wrong_gender)
                # Определяем путь к модели в зависимости от типа
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)
                else:
                    model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)
                logger.debug("face_swapper: %s", type(face_swapper).__name__)

                source_face_idx = 0

                pbar = progress_bar(len(target_imgs))

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img_np, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                        logger.debug("Switched to next source_face_idx=%d: %s (wrong_gender=%s)",
                                     source_face_idx, _describe_obj(source_face, "source_face"), src_wrong_gender)
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        # Reading results to make current face swap on a previous face result
                        logger.status(f"Swapping...")
                        for i, (target_img, target_face) in enumerate(zip(results, target_faces)):
                            target_face_single, wrong_gender = get_face_single(target_img, target_face, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                            logger.debug("(Image %s) target_face_single: %s (wrong_gender=%s)", i, _describe_obj(target_face_single, "target_face_single"), wrong_gender)
                            if target_face_single is not None and wrong_gender == 0:
                                result = target_img
                                # logger.status(f"Swapping {i}...")
                                # Обработка в зависимости от типа модели
                                if "hyperswap" in model:
                                    # Для Hyperswap используем специальную функцию
                                    res = run_hyperswap(face_swapper, source_face, target_face_single, result)
                                    logger.debug("(Image %s) run_hyperswap returned: %s", i, _describe_obj(res, "res"))
                                    if _safe_is_tuple2(res):
                                        swapped_face_256, M = res
                                        if swapped_face_256 is not None:
                                            try:
                                                result = paste_back(result, swapped_face_256, M, crop_size=256)
                                            except Exception as e:
                                                logger.error("(Image %s) paste_back failed: %s", i, e)
                                        else:
                                            logger.error("(Image %s) run_hyperswap returned (None, M=%s) — skipping paste_back", i, M)
                                    else:
                                        logger.error("(Image %s) run_hyperswap returned unexpected %s; skipping", i, type(res).__name__)
                                elif face_boost_enabled:
                                    logger.status(f"Face Boost is enabled")
                                    ret = None
                                    try:
                                        ret = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
                                    except Exception as e:
                                        logger.error("(Image %s) face_swapper.get (boost) failed: %s", i, e)
                                        ret = None
                                    logger.debug("(Image %s) face_swapper.get (boost) returned: %s", i, _describe_obj(ret, "ret"))
                                    if isinstance(ret, tuple) and len(ret) >= 2:
                                        bgr_fake, M = ret[0], ret[1]
                                    else:
                                        bgr_fake, M = ret, None
                                    if bgr_fake is None:
                                        logger.error("(Image %s) Boost: bgr_fake is None; skipping", i)
                                    else:
                                        try:
                                            bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                            if M is not None:
                                                M *= scale
                                            else:
                                                logger.warning("(Image %s) Boost: M is None after get(); paste may be misaligned.", i)
                                            result = swapper.in_swap(target_img, bgr_fake, M)
                                        except Exception as e:
                                            logger.error("(Image %s) Boost restore/swap failed: %s", i, e)
                                else:
                                    # Для остальных моделей используем метод get()
                                    try:
                                        tmp = face_swapper.get(target_img, target_face_single, source_face)
                                        logger.debug("(Image %s) face_swapper.get returned: %s", i, _describe_obj(tmp, "tmp"))
                                        result = tmp
                                    except Exception as e:
                                        logger.error("(Image %s) face_swapper.get failed: %s", i, e)
                                results[i] = result
                                pbar.update(1)
                            elif wrong_gender == 1:
                                wrong_gender = 0
                                logger.status("Wrong target gender detected")
                                pbar.update(1)
                                continue
                            else:
                                logger.status(f"No target face found for {face_num}")
                                pbar.update(1)
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                progress_bar_reset(pbar)
                
                result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    else:
        logger.debug("swap_face_many called with model=None; returning original images")
    return result_images
