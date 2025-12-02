
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, ANY

# Add repository root to sys.path
sys.path.append(os.getcwd())

# --- Mocking Dependencies ---

# 1. Mock 'modules' and 'modules.shared'
modules_mock = MagicMock()
sys.modules['modules'] = modules_mock
shared_mock = MagicMock()
sys.modules['modules.shared'] = shared_mock
modules_mock.shared = shared_mock

cmd_opts_mock = MagicMock()
cmd_opts_mock.reactor_loglevel = "INFO"
shared_mock.cmd_opts = cmd_opts_mock

# 2. Mock 'folder_paths'
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].models_dir = "/tmp/models"

# 3. Mock 'comfy' libs
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()

# 4. Mock 'insightface'
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()
sys.modules['insightface.app.common'] = MagicMock()
sys.modules['insightface.model_zoo'] = MagicMock()

# 5. Mock 'onnxruntime'
sys.modules['onnxruntime'] = MagicMock()

# 6. Mock 'safetensors'
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()

# 7. Mock 'torch' and 'torchvision'
# We keep torch.cuda.is_available() returning False
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
torch_mock.cuda.is_available.return_value = False
torch_mock.backends.mps.is_available.return_value = False

sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.utils'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()

# 8. Mock 'scripts.r_faceboost' to avoid importing complex sub-dependencies
# This is crucial to bypass 'r_basicsr' and 'torch' deep dependencies
faceboost_mock = MagicMock()
sys.modules['scripts.r_faceboost'] = faceboost_mock
sys.modules['scripts.r_faceboost.swapper'] = faceboost_mock.swapper
sys.modules['scripts.r_faceboost.restorer'] = faceboost_mock.restorer

# 9. Import real libs
import numpy as np
import cv2
from PIL import Image

# --- Import Module Under Test ---
import reactor_utils
from scripts import reactor_swapper

class TestReactorSwapper(unittest.TestCase):

    def setUp(self):
        pass

    def create_dummy_image(self, width=100, height=100, color=(255, 0, 0)):
        img = Image.new('RGB', (width, height), color=color)
        return img

    @patch('scripts.reactor_swapper.getFaceSwapModel')
    @patch('scripts.reactor_swapper.analyze_faces')
    @patch('scripts.reactor_swapper.get_face_single')
    def test_swap_face_inswapper(self, mock_get_face_single, mock_analyze_faces, mock_getFaceSwapModel):
        """
        Test standard swap with inswapper model.
        """
        mock_swapper = MagicMock()
        mock_getFaceSwapModel.return_value = mock_swapper

        swapped_bgr = np.zeros((100, 100, 3), dtype=np.uint8) + 50
        mock_swapper.get.return_value = (swapped_bgr, np.eye(2, 3))

        mock_analyze_faces.return_value = [MagicMock()]

        mock_face = MagicMock()
        mock_face.bbox = [10, 10, 50, 50]
        mock_get_face_single.return_value = (mock_face, 0)

        source_img = self.create_dummy_image()
        target_img = self.create_dummy_image(color=(0, 0, 255))

        result_img, bboxes, indices = reactor_swapper.swap_face(
            source_img,
            target_img,
            model="inswapper_128.onnx",
            source_faces_index=[0],
            faces_index=[0]
        )

        self.assertIsNotNone(result_img)
        self.assertIsInstance(result_img, Image.Image)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(indices, [0])
        mock_swapper.get.assert_called()

    @patch('scripts.reactor_swapper.getFaceSwapModel')
    @patch('scripts.reactor_swapper.analyze_faces')
    @patch('scripts.reactor_swapper.get_face_single')
    @patch('scripts.reactor_swapper.safe_run_hyperswap')
    def test_swap_face_hyperswap(self, mock_safe_run_hyperswap, mock_get_face_single, mock_analyze_faces, mock_getFaceSwapModel):
        """
        Test swap with hyperswap model.
        """
        mock_session = MagicMock()
        mock_getFaceSwapModel.return_value = mock_session

        swapped_256 = np.zeros((256, 256, 3), dtype=np.uint8)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mock_safe_run_hyperswap.return_value = (swapped_256, M)

        mock_analyze_faces.return_value = [MagicMock()]
        mock_face = MagicMock()
        mock_face.bbox = [10, 10, 50, 50]
        mock_get_face_single.return_value = (mock_face, 0)

        source_img = self.create_dummy_image()
        target_img = self.create_dummy_image()

        result_img, bboxes, indices = reactor_swapper.swap_face(
            source_img,
            target_img,
            model="hyperswap_model.onnx",
            source_faces_index=[0],
            faces_index=[0]
        )

        self.assertIsNotNone(result_img)
        self.assertEqual(len(bboxes), 1)
        mock_safe_run_hyperswap.assert_called()

    @patch('scripts.reactor_swapper.getFaceSwapModel')
    @patch('scripts.reactor_swapper.analyze_faces')
    @patch('scripts.reactor_swapper.get_face_single')
    def test_swap_face_boost(self, mock_get_face_single, mock_analyze_faces, mock_getFaceSwapModel):
        """
        Test swap with face boost enabled.
        """
        # Note: We need to patch the swapper and restorer functions that are imported in reactor_swapper
        # Since we mocked the module 'scripts.r_faceboost', reactor_swapper.swapper and reactor_swapper.restorer are Mocks.
        # We can configure them here.

        mock_swapper_model = MagicMock()
        mock_getFaceSwapModel.return_value = mock_swapper_model

        fake_face = np.zeros((128, 128, 3), dtype=np.uint8)
        M = np.eye(2, 3)
        mock_swapper_model.get.return_value = (fake_face, M)

        mock_analyze_faces.return_value = [MagicMock()]
        mock_face = MagicMock()
        mock_face.bbox = [10, 10, 50, 50]
        mock_get_face_single.return_value = (mock_face, 0)

        # Configure the mocked swapper and restorer modules
        restored_face = np.zeros((256, 256, 3), dtype=np.uint8)
        reactor_swapper.restorer.get_restored_face.return_value = (restored_face, 2.0)

        final_bgr = np.zeros((100, 100, 3), dtype=np.uint8) + 100
        reactor_swapper.swapper.in_swap.return_value = final_bgr

        source_img = self.create_dummy_image()
        target_img = self.create_dummy_image()

        result_img, bboxes, indices = reactor_swapper.swap_face(
            source_img,
            target_img,
            model="inswapper_128.onnx",
            source_faces_index=[0],
            faces_index=[0],
            face_boost_enabled=True
        )

        self.assertIsNotNone(result_img)
        self.assertEqual(len(bboxes), 1)

        reactor_swapper.restorer.get_restored_face.assert_called()
        reactor_swapper.swapper.in_swap.assert_called()

    @patch('scripts.reactor_swapper.getFaceSwapModel')
    @patch('scripts.reactor_swapper.analyze_faces')
    @patch('scripts.reactor_swapper.get_face_single')
    def test_swap_face_many_multiple_targets(self, mock_get_face_single, mock_analyze_faces, mock_getFaceSwapModel):
        """
        Test swap_face_many with multiple target images.
        """
        mock_swapper = MagicMock()
        mock_getFaceSwapModel.return_value = mock_swapper
        mock_swapper.get.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), np.eye(2, 3))

        mock_analyze_faces.return_value = [MagicMock()]

        mock_face = MagicMock()
        mock_face.bbox = [10, 10, 50, 50]
        mock_get_face_single.return_value = (mock_face, 0)

        source_img = self.create_dummy_image()
        target_imgs = [self.create_dummy_image(), self.create_dummy_image()]

        result_imgs, bboxes, indices = reactor_swapper.swap_face_many(
            source_img,
            target_imgs,
            model="inswapper_128.onnx",
            source_faces_index=[0],
            faces_index=[0]
        )

        self.assertEqual(len(result_imgs), 2)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], [0])
        self.assertEqual(indices[1], [0])

if __name__ == '__main__':
    unittest.main()
