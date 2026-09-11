"""AI-based beacon detector using ONNX Runtime for inference."""

import numpy as np
import cv2
from detect import DetectorBase, Detection


class AIDetector(DetectorBase):
    """ONNX-based beacon detector. Falls back to classical if model unavailable."""

    def __init__(self, model_path=None, confidence_threshold=0.5,
                 fallback=None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fallback = fallback
        self.session = None
        self._load_model()

    def _load_model(self):
        """Load ONNX model if available."""
        if self.model_path is None:
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
        except Exception:
            print(f"  [AI Detector] Model not found at {self.model_path}, using fallback")
            self.session = None

    def detect(self, frame):
        """Detect beacon using AI model, fallback to classical if needed."""
        if self.session is None:
            if self.fallback:
                return self.fallback.detect(frame)
            return Detection()

        try:
            return self._run_inference(frame)
        except Exception as e:
            print(f"  [AI Detector] Inference failed: {e}")
            if self.fallback:
                return self.fallback.detect(frame)
            return Detection()

    def _run_inference(self, frame):
        """Run ONNX inference on frame."""
        # Preprocess: resize to model input size
        input_shape = self.session.get_inputs()[0].shape
        h, w = input_shape[2], input_shape[3]

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        resized = cv2.resize(gray, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0) # Channel dim
        blob = np.expand_dims(blob, axis=0) # Batch dim

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})

        # Parse output — expect [batch, 5] = [confidence, x1, y1, x2, y2]
        preds = outputs[0][0]  # shape: (5,)
        conf = preds[0]

        if conf < self.confidence_threshold:
            return Detection()

        x1, y1, x2, y2 = preds[1:5]

        # Scale back to original frame size
        sx = frame.shape[1]
        sy = frame.shape[0]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        return Detection(
            detected=True, cx=float(cx), cy=float(cy), area=float(area),
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            confidence=float(conf),
            beacon_id=""
        )
