"""Tests for the AI detector and synthetic training data generator."""

import numpy as np
import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from ai_detector import AIDetector
from detect import BeaconDetector, Detection


# ---------------------------------------------------------------------------
# Helper: synthetic beacon frame
# ---------------------------------------------------------------------------

def _make_beacon_frame(width=320, height=240, cx=160, cy=120,
                       sigma=5.0, peak=255.0, bg=10):
    """Create a synthetic BGR frame with a Gaussian beacon."""
    frame = np.full((height, width, 3), bg, dtype=np.uint8)
    xs = np.arange(width) - cx
    ys = np.arange(height) - cy
    xx, yy = np.meshgrid(xs, ys)
    gaussian = peak * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    frame[:, :, 0] = np.clip(
        frame[:, :, 0].astype(np.float32) + gaussian, 0, 255
    ).astype(np.uint8)
    frame[:, :, 1] = frame[:, :, 0]
    frame[:, :, 2] = frame[:, :, 0]
    return frame


def _make_dark_frame(width=320, height=240, bg=10):
    """Create a frame with no beacon (all dark)."""
    return np.full((height, width, 3), bg, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mock ONNX session
# ---------------------------------------------------------------------------

class _MockInput:
    """Mimics onnxruntime model input metadata."""
    def __init__(self, shape):
        self.shape = shape
        self.name = "input"


class _MockSession:
    """Mimics onnxruntime.InferenceSession for testing."""

    def __init__(self, detections):
        """
        Args:
            detections: numpy array of shape (1, N, 6) with
                        [x1, y1, x2, y2, confidence, class_id].
        """
        self._detections = detections
        self._input = _MockInput([1, 3, 64, 64])

    def get_inputs(self):
        return [self._input]

    def run(self, output_names, input_feed):
        return [self._detections]


# ---------------------------------------------------------------------------
# AIDetector tests
# ---------------------------------------------------------------------------

class TestAIDetectorNoModel:
    """AIDetector with no model_path should return empty Detection."""

    def test_returns_empty_detection(self):
        detector = AIDetector(model_path=None)
        frame = _make_beacon_frame()
        result = detector.detect(frame)
        assert isinstance(result, Detection)
        assert result.detected is False

    def test_returns_empty_on_dark_frame(self):
        detector = AIDetector(model_path=None)
        frame = _make_dark_frame()
        result = detector.detect(frame)
        assert result.detected is False


class TestAIDetectorFallback:
    """AIDetector should fall back to a classical detector when model unavailable."""

    def test_fallback_used_when_no_model(self):
        classical = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        detector = AIDetector(model_path=None, fallback=classical)

        frame = _make_beacon_frame(cx=160, cy=120, sigma=5.0, peak=255.0, bg=10)
        result = detector.detect(frame)

        assert result.detected is True
        assert abs(result.cx - 160) < 15.0
        assert abs(result.cy - 120) < 15.0

    def test_fallback_used_when_model_path_invalid(self):
        classical = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        detector = AIDetector(
            model_path="/nonexistent/path/model.onnx",
            fallback=classical,
        )
        # _load_model prints a warning and sets session=None
        assert detector.session is None

        frame = _make_beacon_frame(cx=160, cy=120)
        result = detector.detect(frame)
        assert result.detected is True

    def test_fallback_used_on_inference_exception(self):
        """When the ONNX session raises during inference, fall back."""
        bad_session = MagicMock()
        bad_session.get_inputs.return_value = [_MockInput([1, 3, 64, 64])]
        bad_session.run.side_effect = RuntimeError("inference error")

        classical = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        detector = AIDetector(model_path=None, fallback=classical)
        detector.session = bad_session  # inject the bad session

        frame = _make_beacon_frame()
        result = detector.detect(frame)
        # Fallback should produce a detection from the beacon in the frame
        assert result.detected is True


class TestAIDetectorMockSession:
    """AIDetector with a mock ONNX session should parse output correctly."""

    def test_detects_beacon_from_mock_output(self):
        # Mock output: one detection at roughly center of 64x64 input
        # In 64x64 space: box from (20,20) to (40,40), confidence 0.95
        # Output is normalized, so 20/64 = 0.3125, 40/64 = 0.625
        detections = np.array([[
            0.95, 0.3125, 0.3125, 0.625, 0.625
        ]], dtype=np.float32)
        mock_session = _MockSession(detections)

        detector = AIDetector(model_path=None, confidence_threshold=0.5)
        detector.session = mock_session

        # Frame is 320x240; mock output is for 64x64 input
        frame = _make_beacon_frame(width=320, height=240, cx=160, cy=120)
        result = detector.detect(frame)

        assert result.detected is True
        assert result.confidence >= 0.949
        # After scaling back: cx should be around 300, cy around 450
        # (20+40)/2 = 30, scale_x = 320/64=5, cx = 30*5 = 150
        # (20+40)/2 = 30, scale_y = 240/64=3.75, cy = 30*3.75 = 112.5
        assert abs(result.cx - 150.0) < 1.0
        assert abs(result.cy - 112.5) < 1.0

    def test_ignores_low_confidence_detections(self):
        # Detection below threshold
        detections = np.array([[
            0.2, 20.0, 20.0, 40.0, 40.0
        ]], dtype=np.float32)
        mock_session = _MockSession(detections)

        detector = AIDetector(model_path=None, confidence_threshold=0.5)
        detector.session = mock_session

        frame = _make_beacon_frame()
        result = detector.detect(frame)

        assert result.detected is False

    def test_picks_highest_confidence_detection(self):
        # Two detections: low and high confidence
        detections = np.array([[
            0.9, 25.0, 25.0, 45.0, 45.0
        ]], dtype=np.float32)
        mock_session = _MockSession(detections)

        detector = AIDetector(model_path=None, confidence_threshold=0.5)
        detector.session = mock_session

        frame = _make_beacon_frame()
        result = detector.detect(frame)

        assert result.detected is True
        assert result.confidence >= 0.899

    def test_empty_detections_returns_no_detection(self):
        detections = np.zeros((1, 5), dtype=np.float32)
        mock_session = _MockSession(detections)

        detector = AIDetector(model_path=None, confidence_threshold=0.5)
        detector.session = mock_session

        frame = _make_beacon_frame()
        result = detector.detect(frame)
        assert result.detected is False


class TestAIDetectorPreprocessing:
    """Verify ONNX input preprocessing."""

    def test_preprocesses_to_correct_shape(self):
        """Check that the input blob is (1, 3, H, W) float32 normalised."""
        captured_inputs = {}

        class CapturingSession:
            def __init__(self):
                self._input = _MockInput([1, 1, 48, 64])
            def get_inputs(self):
                return [self._input]
            def run(self, output_names, input_feed):
                captured_inputs.update(input_feed)
                return [np.zeros((1, 5), dtype=np.float32)]

        detector = AIDetector(model_path=None)
        detector.session = CapturingSession()

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        detector.detect(frame)

        blob = captured_inputs["input"]
        assert blob.shape == (1, 1, 48, 64)
        assert blob.dtype == np.float32
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0


# ---------------------------------------------------------------------------
# generate_training_data tests
# ---------------------------------------------------------------------------

class TestGenerateTrainingData:
    """Test the synthetic training data generator."""

    @pytest.fixture
    def tmp_output(self, tmp_path):
        return str(tmp_path / "train_output")

    def test_produces_correct_directory_structure(self, tmp_output):
        from generate_training_data import generate_dataset

        meta = generate_dataset(
            tmp_output, num_frames=20, seeds=[42], scenarios=["clean"]
        )

        assert os.path.isdir(tmp_output)
        assert os.path.isdir(os.path.join(tmp_output, "frames"))
        assert os.path.isfile(os.path.join(tmp_output, "labels.json"))
        assert os.path.isfile(os.path.join(tmp_output, "metadata.json"))

    def test_generates_correct_frame_count(self, tmp_output):
        from generate_training_data import generate_dataset

        meta = generate_dataset(
            tmp_output, num_frames=20, seeds=[42], scenarios=["clean"]
        )

        assert meta["total_frames"] == 20
        frames_dir = os.path.join(tmp_output, "frames")
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
        assert len(frame_files) == 20

    def test_labels_json_valid(self, tmp_output):
        from generate_training_data import generate_dataset

        generate_dataset(
            tmp_output, num_frames=20, seeds=[42], scenarios=["clean"]
        )

        with open(os.path.join(tmp_output, "labels.json")) as f:
            labels = json.load(f)

        assert len(labels) == 20
        for label in labels:
            assert "image" in label
            assert "bbox" in label
            assert "center" in label
            assert "visible" in label
            assert "scenario" in label
            assert "seed" in label

    def test_positive_negative_counts_consistent(self, tmp_output):
        from generate_training_data import generate_dataset

        meta = generate_dataset(
            tmp_output, num_frames=20, seeds=[42], scenarios=["clean"]
        )

        with open(os.path.join(tmp_output, "labels.json")) as f:
            labels = json.load(f)

        num_pos = sum(1 for l in labels if l["visible"])
        num_neg = sum(1 for l in labels if not l["visible"])
        assert num_pos + num_neg == meta["total_frames"]
        assert meta["num_positive"] == num_pos
        assert meta["num_negative"] == num_neg

    def test_visible_labels_have_valid_bboxes(self, tmp_output):
        from generate_training_data import generate_dataset

        generate_dataset(
            tmp_output, num_frames=30, seeds=[42], scenarios=["clean"]
        )

        with open(os.path.join(tmp_output, "labels.json")) as f:
            labels = json.load(f)

        for label in labels:
            if label["visible"]:
                bbox = label["bbox"]
                assert len(bbox) == 4, "bbox should have 4 elements [x1,y1,x2,y2]"
                x1, y1, x2, y2 = bbox
                assert x2 > x1, "bbox x2 must be > x1"
                assert y2 > y1, "bbox y2 must be > y1"
                assert x1 >= 0, "bbox x1 must be non-negative"
                assert y1 >= 0, "bbox y1 must be non-negative"

                center = label["center"]
                assert len(center) == 2
                # Center should be within the image bounds (1280x720 default)
                assert 0 <= center[0] <= 1280
                assert 0 <= center[1] <= 720

    def test_hard_scenario_generates_frames(self, tmp_output):
        from generate_training_data import generate_dataset

        meta = generate_dataset(
            tmp_output, num_frames=20, seeds=[42], scenarios=["hard"]
        )

        assert meta["total_frames"] == 20
        with open(os.path.join(tmp_output, "labels.json")) as f:
            labels = json.load(f)
        assert all(l["scenario"] == "hard" for l in labels)

    def test_multiple_seeds(self, tmp_output):
        from generate_training_data import generate_dataset

        meta = generate_dataset(
            tmp_output, num_frames=40, seeds=[42, 99], scenarios=["clean"]
        )

        # 40 frames / (2 seeds * 1 scenario) = 20 per seed
        assert meta["total_frames"] == 40
        with open(os.path.join(tmp_output, "labels.json")) as f:
            labels = json.load(f)

        seeds_seen = set(l["seed"] for l in labels)
        assert seeds_seen == {42, 99}

    def test_metadata_keys(self, tmp_output):
        from generate_training_data import generate_dataset

        generate_dataset(
            tmp_output, num_frames=10, seeds=[42], scenarios=["clean"]
        )

        with open(os.path.join(tmp_output, "metadata.json")) as f:
            meta = json.load(f)

        assert "total_frames" in meta
        assert "num_positive" in meta
        assert "num_negative" in meta
        assert "seeds" in meta
        assert "scenarios" in meta
