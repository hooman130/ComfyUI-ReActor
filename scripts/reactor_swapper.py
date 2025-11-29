import os
import shutil
from typing import List, Union, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import insightface
from insightface.app.common import Face
import torch

import folder_paths
import comfy.model_management as model_management
from modules.shared import state

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

# ============================================================
# Utility / Diagnostic helpers
# ============================================================
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
    if face is None:
        return f"{name}=None"
    try:
        bbox = getattr(face, "bbox", None)
        sex = getattr(face, "sex", None)
        l5 = getattr(face, "landmark_5", None)
        kps = getattr(face, "kps", None)
        lm = getattr(face, "landmark", None)
        emb = getattr(face, "normed_embedding", None)
        emb_shape = emb.shape if isinstance(emb, np.ndarray) else None
        return f"{name}(bbox={bbox}, sex={sex}, has_l5={l5 is not None}, has_kps={kps is not None}, has_lm={lm is not None}, emb_shape={emb_shape})"
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

def ensure_tuple2(ret) -> Tuple[Union[np.ndarray, Image.Image, None], Union[np.ndarray, None]]:
    """
    Normalize return values to (data, meta) style 2-tuple.
    If 'ret' is already length-2 tuple -> return as-is.
    If 'ret' is single ndarray/Image -> (ret, None).
    If None -> (None, None).
    Anything else -> wrap and log.
    """
    if ret is None:
        return None, None
    if isinstance(ret, tuple):
        if len(ret) == 2:
            return ret[0], ret[1]
        elif len(ret) == 0:
            return None, None
        else:
            logger.warning("ensure_tuple2 received a tuple of len=%d; trimming to first two.", len(ret))
            return ret[0], ret[1]
    # single object
    return ret, None

# ============================================================
# Execution Providers
# ============================================================
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

# ============================================================
# Paths / Globals
# ============================================================
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

ANALYSIS_MODELS = {"640": None, "320": None}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

# ============================================================
# Model / Face Analysis Helpers
# ============================================================
def unload_model(model):
    if model is not None:
        del model
    return None

def unload_all_models():
    global FS_MODEL
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size=(640, 640)):
    global ANALYSIS_MODELS
    key = str(det_size[0])
    ANALYSIS_MODEL = ANALYSIS_MODELS.get(key)
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[key] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)
        model_filename = os.path.basename(model_path)
        try:
            if "hyperswap" in model_filename.lower():
                correct_path = os.path.join(folder_paths.models_dir, "hyperswap", model_filename)
                FS_MODEL = ort.InferenceSession(correct_path, providers=providers)
            elif "reswapper" in model_filename.lower():
                correct_path = os.path.join(folder_paths.models_dir, "reswapper", model_filename)
                FS_MODEL = insightface.model_zoo.get_model(correct_path, providers=providers)
            else:
                FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
            logger.debug("Loaded model: path=%s type=%s providers=%s", model_path, type(FS_MODEL).__name__, providers)
        except Exception as e:
            logger.error("Failed to load model at %s: %s", model_path, e)
            FS_MODEL = None
    return FS_MODEL

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    logger.debug("analyze_faces: det_size=%s, %s", det_size, _describe_ndarray(img_data, "img"))
    face_analyser = getAnalysisModel(det_size)
    faces = []
    try:
        faces = face_analyser.get(img_data)
        logger.debug("analyze_faces: found %d faces", len(faces))
        if faces:
            logger.debug("analyze_faces sample: %s", _describe_face(faces[0], "face[0]"))
    except Exception as e:
        logger.error("Face analysis error: %s", e)

    if len(faces) == 0 and det_size[0] > 320:
        return analyze_faces(img_data, half_det_size(det_size))
    return faces

def sort_by_order(face_list, order: str):
    if order == "left-right":
        return sorted(face_list, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face_list, key=lambda x: x.bbox[0], reverse=True)
    if order == "top-bottom":
        return sorted(face_list, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face_list, key=lambda x: x.bbox[1], reverse=True)
    if order == "small-large":
        return sorted(face_list, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return sorted(face_list, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

def get_face_gender(face_list, face_index, gender_condition, operated: str, order: str):
    genders = [x.sex for x in face_list]
    genders.reverse()
    if face_index >= len(genders):
        logger.status("%s face index %s out of bounds (max=%s)", operated, face_index, len(genders))
        return None, 0
    face_gender = genders[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    match = (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M")
    faces_sorted = sort_by_order(face_list, order)
    wrong = 0 if match else 1
    if wrong == 0:
        logger.status("Gender matches condition")
    else:
        logger.status("Gender does NOT match condition")
    try:
        return faces_sorted[face_index], wrong
    except IndexError:
        return None, wrong

def get_face_single(img_data: np.ndarray, face_list, face_index=0, det_size=(640, 640),
                    gender_source=0, gender_target=0, order="large-small"):
    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face_list) == 0 and det_size[0] > 320:
            return get_face_single(img_data, analyze_faces(img_data, half_det_size(det_size)),
                                   face_index, half_det_size(det_size),
                                   gender_source, gender_target, order)
        return get_face_gender(face_list, face_index, gender_source, "Source", order)

    if gender_target != 0:
        if len(face_list) == 0 and det_size[0] > 320:
            return get_face_single(img_data, analyze_faces(img_data, half_det_size(det_size)),
                                   face_index, half_det_size(det_size),
                                   gender_source, gender_target, order)
        return get_face_gender(face_list, face_index, gender_target, "Target", order)

    if len(face_list) == 0 and det_size[0] > 320:
        return get_face_single(img_data, analyze_faces(img_data, half_det_size(det_size)),
                               face_index, half_det_size(det_size),
                               gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face_list, order)
        selected_face = faces_sorted[face_index]
        logger.debug("Selected face: %s", _describe_face(selected_face, "selected_face"))
        return selected_face, 0
    except IndexError:
        return None, 0

# ============================================================
# Landmark / Affine / Swapper Core
# ============================================================
def get_landmarks_5(face):
    if hasattr(face, 'landmark_5') and face.landmark_5 is not None:
        return face.landmark_5
    if hasattr(face, 'kps') and face.kps is not None:
        return face.kps
    if hasattr(face, 'landmark') and face.landmark is not None and face.landmark.shape[0] >= 68:
        idxs = [36, 45, 30, 48, 54]
        return face.landmark[idxs]
    logger.warning("No suitable landmarks in face object: %s", dir(face))
    return None

def get_affine_transform(src_pts, dst_pts):
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M

def create_gradient_mask(crop_size=256):
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    center = (crop_size // 2, crop_size // 2)
    axes = (int(crop_size * 0.35), int(crop_size * 0.4))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return np.clip(mask, 0, 1)

def paste_back(target_img, swapped_face, M, crop_size=256):
    logger.debug("paste_back: %s | %s | M=%s", _describe_ndarray(target_img, "target_img"), _describe_ndarray(swapped_face, "swapped_face"), M)
    if M is None:
        logger.error("paste_back received M=None, returning original target")
        return target_img

    mask = create_gradient_mask(crop_size)
    mask_3c = np.stack([mask]*3, axis=2)

    h, w = target_img.shape[:2]
    swapped_face_norm = swapped_face.astype(np.float32) / 255.0
    mask_norm = mask_3c.astype(np.float32)

    warped_face = cv2.warpAffine(swapped_face_norm, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.5)
    warped_mask = cv2.warpAffine(mask_norm, M, (w, h),
                                 flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    warped_face = np.nan_to_num(np.clip(warped_face, 0, 1), nan=0.5)
    warped_mask = np.nan_to_num(np.clip(warped_mask, 0, 1), nan=0.0)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)

    target_float = target_img.astype(np.float32) / 255.0
    result_float = target_float * (1 - warped_mask) + warped_face * warped_mask
    result = (result_float * 255).clip(0, 255).astype(np.uint8)
    return result

def run_hyperswap(session, source_face, target_face, target_img):
    # Returns (swapped_256, M) or (None, None)
    logger.debug("run_hyperswap: %s | %s | %s | %s",
                 type(session).__name__,
                 _describe_face(source_face, "source_face"),
                 _describe_face(target_face, "target_face"),
                 _describe_ndarray(target_img, "target_img") if isinstance(target_img, np.ndarray) else _describe_obj(target_img, "target_img"))
    try:
        source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)
    except Exception as e:
        logger.error("Source embedding error: %s", e)
        return None, None

    target_landmarks_5 = get_landmarks_5(target_face)
    if target_landmarks_5 is None:
        logger.error("Target landmarks not found")
        return None, None

    std_landmarks_256 = np.array([
        [84.87, 105.94],
        [171.13, 105.94],
        [128.00, 146.66],
        [96.95, 188.64],
        [159.05, 188.64]
    ], dtype=np.float32)

    try:
        M = get_affine_transform(target_landmarks_5.astype(np.float32), std_landmarks_256)
    except Exception as e:
        logger.error("Affine transform error: %s", e)
        return None, None

    if M is None or not np.isfinite(M).all():
        logger.error("Invalid affine matrix M (%s)", M)
        return None, None

    try:
        crop = cv2.warpAffine(target_img, M, (256, 256),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    except Exception as e:
        logger.error("warpAffine crop failed: %s", e)
        return None, None

    crop_input = crop[:, :, ::-1].astype(np.float32) / 255.0
    crop_input = (crop_input - 0.5) / 0.5
    crop_input = crop_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    try:
        output = session.run(None, {'source': source_embedding, 'target': crop_input})[0][0]
    except Exception as e:
        logger.error("Hyperswap model inference failed: %s", e)
        return None, None

    if isinstance(output, np.ndarray):
        output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
        if output.min() < 0.0 or output.max() <= 1.5:
            output = ((output + 1.0) / 2.0 * 255.0)
        output = np.clip(output, 0, 255).astype(np.uint8).copy()
        try: output.setflags(write=True)
        except Exception: pass

    output = output.transpose(1, 2, 0)[:, :, ::-1]  # to BGR
    return output, M

# Safe wrapper to ensure consistent tuple return
def safe_run_hyperswap(session, source_face, target_face, target_img):
    swapped, M = run_hyperswap(session, source_face, target_face, target_img)
    return swapped, M  # already normalized

def safe_faceswap_get(face_swapper_obj, *args, **kwargs):
    """
    Wrap face_swapper.get calls to always return (image, M_or_None).
    If the underlying implementation returns only image, we convert it.
    If it returns (img, M, extra...) we trim to first two.
    """
    ret = None
    try:
        ret = face_swapper_obj.get(*args, **kwargs)
    except Exception as e:
        logger.error("face_swapper.get raised error: %s", e)
        return None, None

    img, M = ensure_tuple2(ret)
    # If M is ndarray but not affine shape (2x3), log warning
    if M is not None and isinstance(M, np.ndarray) and M.shape != (2, 3):
        logger.warning("Returned M has unexpected shape %s; expected (2,3).", M.shape)
    return img, M

# ============================================================
# Face Swapping High-Level
# ============================================================
def _resolve_model_path(model: str) -> str:
    if "inswapper" in model:
        return os.path.join(insightface_path, model)
    if "reswapper" in model:
        return os.path.join(reswapper_path, model)
    if "hyperswap" in model:
        return os.path.join(hyperswap_path, model)
    # fallback
    return os.path.join(insightface_path, model)

def _prepare_source_image(source_img):
    if isinstance(source_img, str):
        import base64, io
        logger.debug("Decoding base64 source image (length=%d)", len(source_img))
        if "base64," in source_img:
            source_img = source_img.split("base64,")[-1]
        img_bytes = base64.b64decode(source_img)
        source_img = Image.open(io.BytesIO(img_bytes))
    return source_img

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def swap_face(
    source_img: Union[Image.Image, str, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = None,
    faces_index: List[int] = None,
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List[str] = None,
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    if source_faces_index is None: source_faces_index = [0]
    if faces_index is None: faces_index = [0]
    if faces_order is None: faces_order = ["large-small", "large-small"]

    logger.debug("swap_face called: model=%s faces_index=%s source_faces_index=%s boost=%s",
                 model, faces_index, source_faces_index, face_boost_enabled)

    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH

    if model is None:
        logger.debug("No model provided; returning original target image.")
        return target_img

    source_img = _prepare_source_image(source_img)
    target_bgr = _pil_to_bgr(target_img)

    # Source faces acquisition
    if source_img is not None:
        source_bgr = _pil_to_bgr(source_img)
        md5_src = get_image_md5hash(source_bgr)
        if SOURCE_IMAGE_HASH != md5_src:
            SOURCE_IMAGE_HASH = md5_src
            SOURCE_FACES = analyze_faces(source_bgr)
        else:
            if SOURCE_FACES is None:
                SOURCE_FACES = analyze_faces(source_bgr)
        source_faces = SOURCE_FACES
    elif face_model is not None:
        source_faces = [face_model]
        source_faces_index = [0]
        logger.status("Using provided face_model as source.")
    else:
        logger.error("No source image or face_model provided; cannot proceed.")
        return target_img

    if source_faces is None or len(source_faces) == 0:
        logger.status("No source faces detected.")
        return target_img

    # Target face acquisition
    md5_tgt = get_image_md5hash(target_bgr)
    if TARGET_IMAGE_HASH != md5_tgt:
        TARGET_IMAGE_HASH = md5_tgt
        TARGET_FACES = analyze_faces(target_bgr)
    else:
        if TARGET_FACES is None:
            TARGET_FACES = analyze_faces(target_bgr)
    target_faces = TARGET_FACES

    if target_faces is None or len(target_faces) == 0:
        logger.status("No target faces detected; skipping.")
        return target_img

    # Select initial source face
    src_face, src_wrong_gender = get_face_single(
        source_bgr, source_faces, face_index=source_faces_index[0],
        gender_source=gender_source, order=faces_order[1]
    )

    if src_face is None or src_wrong_gender == 1:
        logger.status("Initial source face invalid or wrong gender.")
        return target_img

    model_path = _resolve_model_path(model)
    face_swapper = getFaceSwapModel(model_path)
    if face_swapper is None:
        logger.error("Model '%s' could not be loaded.", model_path)
        return target_img

    result = target_bgr
    src_face_idx = 0

    for face_num in faces_index:
        logger.debug("Processing target face index: %d", face_num)
        if face_num >= len(target_faces):
            logger.status("Face index %d out of bounds; break.", face_num)
            break

        # Handle multiple source faces mapping if provided
        if len(source_faces_index) > 1 and src_face_idx > 0:
            src_face, src_wrong_gender = get_face_single(
                source_bgr, source_faces, face_index=source_faces_index[src_face_idx],
                gender_source=gender_source, order=faces_order[1]
            )
            logger.debug("Switched source face idx=%d: %s wrong_gender=%s",
                         src_face_idx, _describe_face(src_face, "src_face"), src_wrong_gender)
        src_face_idx += 1

        if src_face is None or src_wrong_gender == 1:
            logger.status("Skipping due to invalid/wrong-gender source face.")
            continue

        tgt_face, tgt_wrong_gender = get_face_single(
            result, target_faces, face_index=face_num,
            gender_target=gender_target, order=faces_order[0]
        )

        if tgt_face is None:
            logger.status("No target face for index=%d", face_num)
            continue
        if tgt_wrong_gender == 1:
            logger.status("Target face gender mismatch at index=%d; skipping.", face_num)
            continue

        logger.status("Swapping face index=%d...", face_num)

        if "hyperswap" in model:
            swapped_256, M = safe_run_hyperswap(face_swapper, src_face, tgt_face, result)
            if swapped_256 is not None:
                result = paste_back(result, swapped_256, M, crop_size=256)
            else:
                logger.error("Hyperswap returned None; skipping this face.")
        elif face_boost_enabled:
            fake_face, M = safe_faceswap_get(face_swapper, result, tgt_face, src_face, paste_back=False)
            if fake_face is None:
                logger.error("Boost mode: face_swapper.get returned None.")
                continue
            try:
                restored_face, scale = restorer.get_restored_face(fake_face, face_restore_model,
                                                                  face_restore_visibility, codeformer_weight,
                                                                  interpolation)
                if M is not None:
                    M = M * scale if isinstance(M, np.ndarray) else M
                else:
                    logger.warning("Boost mode: M is None; in_swap may misalign.")
                result = swapper.in_swap(result, restored_face, M)
            except Exception as e:
                logger.error("Boost restore/swap error: %s", e)
        else:
            swapped, M_unused = safe_faceswap_get(face_swapper, result, tgt_face, src_face)
            if swapped is None:
                logger.error("Standard swap: model returned None; skipping face.")
                continue
            result = swapped  # paste already done internally for inswapper/reswapper

    final_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    logger.debug("swap_face finished successfully.")
    return final_img

def swap_face_many(
    source_img: Union[Image.Image, str, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = None,
    faces_index: List[int] = None,
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List[str] = None,
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    if source_faces_index is None: source_faces_index = [0]
    if faces_index is None: faces_index = [0]
    if faces_order is None: faces_order = ["large-small", "large-small"]

    logger.debug("swap_face_many: model=%s count=%d faces_index=%s", model, len(target_imgs), faces_index)

    if model is None:
        logger.debug("No model provided; returning targets unchanged.")
        return target_imgs

    source_img = _prepare_source_image(source_img)
    target_bgr_list = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in target_imgs]

    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH

    # Source faces
    if source_img is not None:
        source_bgr = _pil_to_bgr(source_img)
        md5_src = get_image_md5hash(source_bgr)
        if SOURCE_IMAGE_HASH != md5_src:
            SOURCE_IMAGE_HASH = md5_src
            SOURCE_FACES = analyze_faces(source_bgr)
        else:
            if SOURCE_FACES is None:
                SOURCE_FACES = analyze_faces(source_bgr)
        source_faces = SOURCE_FACES
    elif face_model is not None:
        source_faces = [face_model]
        source_faces_index = [0]
        logger.status("Using provided face_model as source.")
    else:
        logger.error("No source image or face_model provided; cannot proceed.")
        return target_imgs

    if not source_faces:
        logger.status("No source faces detected.")
        return target_imgs

    # Analyze all targets with caching
    target_faces_list = []
    pbar = progress_bar(len(target_bgr_list))
    for i, t_img in enumerate(target_bgr_list):
        md5_tgt = get_image_md5hash(t_img)
        if len(TARGET_IMAGE_LIST_HASH) <= i:
            TARGET_IMAGE_LIST_HASH.append(md5_tgt)
            new_or_changed = True
        else:
            new_or_changed = TARGET_IMAGE_LIST_HASH[i] != md5_tgt
            if new_or_changed:
                TARGET_IMAGE_LIST_HASH[i] = md5_tgt

        if len(TARGET_FACES_LIST) <= i or new_or_changed or TARGET_FACES_LIST[i] is None:
            tgt_faces = analyze_faces(t_img)
            if len(TARGET_FACES_LIST) <= i:
                TARGET_FACES_LIST.append(tgt_faces)
            else:
                TARGET_FACES_LIST[i] = tgt_faces
        else:
            tgt_faces = TARGET_FACES_LIST[i]

        target_faces_list.append(tgt_faces)
        pbar.update(1)
    progress_bar_reset(pbar)

    if not any(target_faces_list):
        logger.status("No faces detected in any target images.")
        return target_imgs

    # Pick initial source face
    src_face, src_wrong_gender = get_face_single(
        source_bgr if source_img is not None else source_bgr,
        source_faces,
        face_index=source_faces_index[0],
        gender_source=gender_source,
        order=faces_order[1]
    )
    if src_face is None or src_wrong_gender == 1:
        logger.status("Initial source face invalid or wrong gender.")
        return target_imgs

    model_path = _resolve_model_path(model)
    face_swapper = getFaceSwapModel(model_path)
    if face_swapper is None:
        logger.error("Could not load model for multi swap: %s", model_path)
        return target_imgs

    results = target_bgr_list
    src_face_idx = 0

    outer_pbar = progress_bar(len(faces_index) * len(results))

    for face_num in faces_index:
        if face_num < 0:
            logger.warning("Negative face index %d skipped.", face_num)
            continue

        if len(source_faces_index) > 1 and src_face_idx > 0:
            src_face, src_wrong_gender = get_face_single(
                source_bgr, source_faces,
                face_index=source_faces_index[src_face_idx],
                gender_source=gender_source,
                order=faces_order[1]
            )
            logger.debug("Multi-swap switched source_face idx=%d: %s wrong=%s",
                         src_face_idx, _describe_face(src_face, "src_face"), src_wrong_gender)
        src_face_idx += 1

        if src_face is None or src_wrong_gender == 1:
            logger.status("Skipping due to invalid/wrong-gender source face (multi).")
            continue

        for i, (img_bgr, tgt_faces) in enumerate(zip(results, target_faces_list)):
            if tgt_faces is None or face_num >= len(tgt_faces):
                logger.debug("Image %d: face index %d not present.", i, face_num)
                outer_pbar.update(1)
                continue

            tgt_face, tgt_wrong_gender = get_face_single(
                img_bgr, tgt_faces, face_index=face_num,
                gender_target=gender_target, order=faces_order[0]
            )
            if tgt_face is None or tgt_wrong_gender == 1:
                logger.status("Image %d: target face invalid or wrong gender.", i)
                outer_pbar.update(1)
                continue

            logger.status("Image %d: Swapping face index %d...", i, face_num)

            if "hyperswap" in model:
                swapped_256, M = safe_run_hyperswap(face_swapper, src_face, tgt_face, img_bgr)
                if swapped_256 is not None:
                    img_bgr = paste_back(img_bgr, swapped_256, M, crop_size=256)
                else:
                    logger.error("Image %d hyperswap returned None.", i)
            elif face_boost_enabled:
                fake_face, M = safe_faceswap_get(face_swapper, img_bgr, tgt_face, src_face, paste_back=False)
                if fake_face is not None:
                    try:
                        restored_face, scale = restorer.get_restored_face(fake_face, face_restore_model,
                                                                          face_restore_visibility,
                                                                          codeformer_weight, interpolation)
                        if M is not None:
                            M = M * scale if isinstance(M, np.ndarray) else M
                        else:
                            logger.warning("Image %d boost mode: M is None.", i)
                        img_bgr = swapper.in_swap(img_bgr, restored_face, M)
                    except Exception as e:
                        logger.error("Image %d boost restore/swap error: %s", i, e)
                else:
                    logger.error("Image %d boost mode: fake_face None.", i)
            else:
                swapped_img, _M_unused = safe_faceswap_get(face_swapper, img_bgr, tgt_face, src_face)
                if swapped_img is not None:
                    img_bgr = swapped_img
                else:
                    logger.error("Image %d standard swap returned None.", i)

            results[i] = img_bgr
            outer_pbar.update(1)

    progress_bar_reset(outer_pbar)

    final_images = [Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) for bgr in results]
    logger.debug("swap_face_many finished.")
    return final_images
