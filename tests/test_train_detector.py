"""Tests for the AI detector training pipeline."""

import pytest
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestBeaconCNN:
    """Tests for the BeaconCNN model architecture."""

    def test_output_shape(self):
        """Model outputs shape [batch, 5] — detection + bbox."""
        from train_detector import BeaconCNN
        model = BeaconCNN()
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert out.shape == (4, 5)

    def test_detection_probability_range(self):
        """Detection probability is in [0, 1]."""
        from train_detector import BeaconCNN
        model = BeaconCNN()
        model.eval()
        x = torch.randn(8, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        det = out[:, 0]
        assert det.min() >= 0.0
        assert det.max() <= 1.0

    def test_bbox_range(self):
        """BBox outputs are in [0, 1]."""
        from train_detector import BeaconCNN
        model = BeaconCNN()
        model.eval()
        x = torch.randn(8, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        bbox = out[:, 1:5]
        assert bbox.min() >= 0.0
        assert bbox.max() <= 1.0

    def test_parameter_count(self):
        """Model has reasonable parameter count (< 500K)."""
        from train_detector import BeaconCNN
        model = BeaconCNN()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 500_000
        assert n_params > 0

    def test_onnx_export(self):
        """Model can be exported to ONNX."""
        from train_detector import BeaconCNN, _export_onnx
        model = BeaconCNN()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'test_model.onnx')
            _export_onnx(model, onnx_path, input_size=(128, 128))
            assert os.path.isfile(onnx_path)
            assert os.path.getsize(onnx_path) > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestBeaconDataset:
    """Tests for the BeaconDataset."""

    def _make_test_data(self, tmpdir):
        """Create minimal test data."""
        frames_dir = os.path.join(tmpdir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        labels = []
        for i in range(10):
            fname = f'frame_{i:06d}.jpg'
            # Create a small dummy image
            import numpy as np
            import cv2
            img = np.zeros((100, 100), dtype=np.uint8)
            if i % 2 == 0:
                # Beacon present
                cv2.circle(img, (50, 50), 5, 255, -1)
                labels.append({
                    'image': fname,
                    'bbox': [40, 40, 60, 60],
                    'center': [50.0, 50.0],
                    'visible': True,
                })
            else:
                labels.append({
                    'image': fname,
                    'bbox': [],
                    'center': [],
                    'visible': False,
                })
            cv2.imwrite(os.path.join(frames_dir, fname), img)

        labels_path = os.path.join(tmpdir, 'labels.json')
        with open(labels_path, 'w') as f:
            json.dump(labels, f)

        return frames_dir, labels_path

    def test_dataset_length(self):
        """Dataset returns correct number of samples."""
        from train_detector import BeaconDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir, labels_path = self._make_test_data(tmpdir)
            ds = BeaconDataset(frames_dir, labels_path)
            assert len(ds) == 10

    def test_dataset_item_shape(self):
        """Each item has correct tensor shapes."""
        from train_detector import BeaconDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir, labels_path = self._make_test_data(tmpdir)
            ds = BeaconDataset(frames_dir, labels_path, input_size=(64, 64))
            img, target = ds[0]
            assert img.shape == (1, 64, 64)
            assert target.shape == (5,)  # det + bbox

    def test_dataset_label_values(self):
        """Labels are 0 or 1 for detection."""
        from train_detector import BeaconDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir, labels_path = self._make_test_data(tmpdir)
            ds = BeaconDataset(frames_dir, labels_path)
            for i in range(len(ds)):
                _, target = ds[i]
                assert target[0].item() in (0.0, 1.0)
