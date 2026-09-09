"""
detect.py — Beacon detector, Kalman filter, and tracking state machine
for the FSOC coarse-alignment simulator.

Pipeline per frame:
  1. Detector: grayscale → threshold → morphology → connected components → centroid
  2. Kalman filter: smooths measurements, predicts through detection gaps
  3. State machine: SEARCHING → ACQUIRING → TRACKING → REACQUIRING

The detector receives ONLY the rendered frame. It never sees ground truth.
The tracker maintains its own state across frames.

Dependencies: numpy, cv2
"""

import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


# ──────────────────────────────────────────────────────────────────────────
# Detection result
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Output of the beacon detector for one frame."""
    detected: bool = False
    cx: float = 0.0         # centroid x (pixels)
    cy: float = 0.0         # centroid y (pixels)
    area: float = 0.0       # blob area (pixels²)
    bbox: tuple = (0, 0, 0, 0)  # (x, y, w, h)
    confidence: float = 0.0  # 0–1, based on brightness vs threshold


# ──────────────────────────────────────────────────────────────────────────
# Beacon detector — classical brightness + connected components
# ──────────────────────────────────────────────────────────────────────────

class BeaconDetector:
    """
    Classical beacon detector.

    Pipeline: grayscale → adaptive threshold → morphological opening →
    connected components → filter by area → select brightest centroid.

    Works for white/bright beacons against a darker background.
    For colored beacons, add HSV filtering before this pipeline.
    """

    def __init__(self,
                 threshold_method: str = "fixed",  # "otsu" or "fixed"
                 fixed_threshold: int = 80,
                 otsu_min_threshold: int = 60,  # floor for Otsu mode
                 min_area: int = 2,
                 max_area: int = 500,
                 morph_kernel_size: int = 3,
                 morph_iterations: int = 1):
        self.threshold_method = threshold_method
        self.fixed_threshold = fixed_threshold
        self.otsu_min_threshold = otsu_min_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.morph_iterations = morph_iterations
        self._last_threshold = 0

    def detect(self, frame: np.ndarray) -> Detection:
        """
        Detect the beacon in a single frame.

        Args:
            frame: BGR or grayscale image (np.ndarray, H×W×C or H×W)

        Returns:
            Detection with centroid, area, and confidence. If no beacon
            is found, Detection.detected is False.
        """
        # 1. Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 2. Threshold
        if self.threshold_method == "otsu":
            otsu_val, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Otsu can pick a very low threshold when the background dominates,
            # catching most of the gradient as 'bright'. Floor it.
            otsu_val = max(int(otsu_val), self.otsu_min_threshold)
            _, binary = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY)
            self._last_threshold = otsu_val
        else:
            _, binary = cv2.threshold(
                gray, self.fixed_threshold, 255, cv2.THRESH_BINARY)
            self._last_threshold = self.fixed_threshold

        # 3. Morphological opening (remove small noise speckles)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, self.morph_kernel,
            iterations=self.morph_iterations)

        # 4. Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 1:
            # Only background label (0); no candidates
            return Detection()

        # 5. Filter by area and select the best candidate
        best_detection = Detection()
        best_brightness = 0

        for i in range(1, num_labels):  # skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue

            cx, cy = centroids[i]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Measure peak brightness in this candidate region
            region = gray[y:y+h, x:x+w]
            if region.size == 0:
                continue
            peak = int(region.max())
            mean = float(region.mean())

            # Confidence: how far above threshold the candidate's mean is
            if self._last_threshold > 0:
                confidence = min(1.0, mean / self._last_threshold)
            else:
                confidence = min(1.0, mean / 100.0)

            # Select the brightest candidate (highest peak)
            if peak > best_brightness:
                best_brightness = peak
                best_detection = Detection(
                    detected=True,
                    cx=float(cx),
                    cy=float(cy),
                    area=float(area),
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=float(confidence),
                )

        return best_detection


# ──────────────────────────────────────────────────────────────────────────
# Kalman filter — 2-D constant velocity
# ──────────────────────────────────────────────────────────────────────────

class KalmanFilter2D:
    """
    2-D constant-velocity Kalman filter for beacon position smoothing.

    State vector: [x, y, vx, vy] (position + velocity in pixels)
    Measurement: [x, y] (centroid from detector)

    Predict every frame. Update only when a detection is available.
    During detection gaps, prediction extrapolates from the last velocity.
    """

    def __init__(self,
                 process_noise_pos: float = 1.0,
                 process_noise_vel: float = 0.5,
                 measurement_noise: float = 4.0,
                 initial_x: float = 640.0,
                 initial_y: float = 360.0):
        """
        Args:
            process_noise_pos: Process noise for position (Q position diag). Higher = trusts predictions less.
            process_noise_vel: Process noise for velocity (Q velocity diag).
            measurement_noise: Measurement noise (R). Higher = trusts detections less.
            initial_x, initial_y: Initial position estimate.
        """
        # State transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt
        # dt is set to 1.0 (one frame per step); scale velocities to pixels/frame
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (we observe x, y directly)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.diag([
            process_noise_pos, process_noise_pos,
            process_noise_vel, process_noise_vel
        ]).astype(np.float64)

        # Measurement noise covariance
        self.R = np.diag([measurement_noise, measurement_noise]).astype(np.float64)

        # Initial state and covariance
        self.x = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([100.0, 100.0, 10.0, 10.0]).astype(np.float64)

        self.initialized = False

    def predict(self) -> tuple[float, float]:
        """
        Predict the next state. Call this every frame, regardless of detection.

        Returns predicted (x, y) position.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, measurement_x: float, measurement_y: float):
        """
        Update with a measurement. Call this only when the detector returns
        a valid detection.
        """
        z = np.array([measurement_x, measurement_y], dtype=np.float64)

        # Innovation (residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        self.initialized = True

    def get_position(self) -> tuple[float, float]:
        """Return the current estimated position."""
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self) -> tuple[float, float]:
        """Return the current estimated velocity (pixels/frame)."""
        return float(self.x[2]), float(self.x[3])

    def initialize(self, x: float, y: float):
        """Initialize the filter at a known position."""
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0]).astype(np.float64)
        self.initialized = True

    def reset(self):
        """Reset to default state."""
        self.x = np.array([640.0, 360.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([100.0, 100.0, 10.0, 10.0]).astype(np.float64)
        self.initialized = False


# ──────────────────────────────────────────────────────────────────────────
# Tracker state machine
# ──────────────────────────────────────────────────────────────────────────

class TrackerState(Enum):
    """Finite state machine states for the beacon tracker."""
    SEARCHING = "SEARCHING"        # Sweeping for the beacon
    ACQUIRING = "ACQUIRING"        # Confirming detections before committing
    TRACKING = "TRACKING"          # Locked on, continuous feedback
    REACQUIRING = "REACQUIRING"    # Lost track, predicting + searching locally


@dataclass
class TrackerOutput:
    """Output of the tracker for one frame."""
    state: TrackerState
    estimated_x: float   # Kalman-estimated position (pixels)
    estimated_y: float
    confidence: float     # Detection confidence (0 if no detection)
    detected: bool        # Whether the detector found the beacon this frame
    consecutive_detections: int  # Running count for acquisition logic
    consecutive_misses: int      # Running count for loss detection


class Tracker:
    """
    Beacon tracker combining detector, Kalman filter, and state machine.

    The tracker manages state transitions and provides a clean interface
    for the controller: call track() each frame with the frame image,
    get back the estimated position, confidence, and current state.

    State transitions:
        SEARCHING  --detection-->  ACQUIRING
        ACQUIRING  --3 consecutive detections-->  TRACKING
        ACQUIRING  --miss-->  SEARCHING
        TRACKING   --5 consecutive misses-->  REACQUIRING
        REACQUIRING --detection-->  TRACKING
        REACQUIRING --timeout-->  SEARCHING
    """

    def __init__(self,
                 detector: BeaconDetector = None,
                 kalman: KalmanFilter2D = None,
                 acquire_threshold: int = 3,       # consecutive detections needed
                 lose_threshold: int = 5,            # consecutive misses before reacquire
                 reacquire_timeout_frames: int = 150, # ~5 seconds at 30 FPS
                 image_width: int = 1280,
                 image_height: int = 720):
        self.detector = detector or BeaconDetector()
        self.kalman = kalman or KalmanFilter2D()
        self.acquire_threshold = acquire_threshold
        self.lose_threshold = lose_threshold
        self.reacquire_timeout = reacquire_timeout_frames
        self.img_w = image_width
        self.img_h = image_height

        self.state = TrackerState.SEARCHING
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.reacquire_counter = 0
        self.last_detection: Optional[Detection] = None

    def track(self, frame: np.ndarray) -> TrackerOutput:
        """
        Process one frame: detect, update Kalman, run state machine.

        Args:
            frame: BGR image from the simulator.

        Returns:
            TrackerOutput with state, estimated position, and confidence.
        """
        # 1. Always predict (advance Kalman even without detection)
        pred_x, pred_y = self.kalman.predict()

        # 2. Detect
        detection = self.detector.detect(frame)

        # 3. Update Kalman if detected
        if detection.detected:
            self.kalman.update(detection.cx, detection.cy)
            self.consecutive_detections += 1
            self.consecutive_misses = 0
            self.last_detection = detection
        else:
            self.consecutive_detections = 0
            self.consecutive_misses += 1

        # 4. Get current estimate
        est_x, est_y = self.kalman.get_position()

        # 5. State machine
        self._update_state(detection.detected)

        # 6. Build output
        confidence = detection.confidence if detection.detected else 0.0
        if not detection.detected and self.last_detection is not None:
            # During prediction, carry forward the last confidence (decayed)
            confidence = self.last_detection.confidence * 0.5

        return TrackerOutput(
            state=self.state,
            estimated_x=est_x,
            estimated_y=est_y,
            confidence=confidence,
            detected=detection.detected,
            consecutive_detections=self.consecutive_detections,
            consecutive_misses=self.consecutive_misses,
        )

    def _update_state(self, detected: bool):
        """Run the state machine transition logic."""
        if self.state == TrackerState.SEARCHING:
            if detected:
                self.kalman.initialize(
                    self.last_detection.cx, self.last_detection.cy)
                self.state = TrackerState.ACQUIRING
                self.consecutive_detections = 1

        elif self.state == TrackerState.ACQUIRING:
            if detected and self.consecutive_detections >= self.acquire_threshold:
                self.state = TrackerState.TRACKING
            elif not detected:
                self.state = TrackerState.SEARCHING
                self.consecutive_detections = 0

        elif self.state == TrackerState.TRACKING:
            if self.consecutive_misses >= self.lose_threshold:
                self.state = TrackerState.REACQUIRING
                self.reacquire_counter = 0

        elif self.state == TrackerState.REACQUIRING:
            if detected:
                self.state = TrackerState.TRACKING
                self.reacquire_counter = 0
            else:
                self.reacquire_counter += 1
                if self.reacquire_counter >= self.reacquire_timeout:
                    self.state = TrackerState.SEARCHING
                    self.reacquire_counter = 0
                    self.kalman.reset()

    def reset(self):
        """Reset tracker to initial state."""
        self.state = TrackerState.SEARCHING
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.reacquire_counter = 0
        self.last_detection = None
        self.kalman.reset()


# ──────────────────────────────────────────────────────────────────────────
# Self-tests
# ──────────────────────────────────────────────────────────────────────────

def _test_detector():
    """Test the beacon detector on synthetic frames."""
    print("Testing detector...")
    from sim import Simulator, CameraState, CircularMotion

    cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
    motion = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                            radius_deg=8.0, angular_speed_deg_s=6.0)
    sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0)
    detector = BeaconDetector()

    detections = 0
    for i in range(30):
        frame, gt = sim.step(i)
        det = detector.detect(frame)
        if det.detected:
            detections += 1
            error = np.sqrt((det.cx - gt.target_pixel_x)**2 +
                            (det.cy - gt.target_pixel_y)**2)
            if i < 5 or i % 10 == 0:
                print(f"  Frame {i}: detected at ({det.cx:.1f}, {det.cy:.1f}), "
                      f"GT ({gt.target_pixel_x:.1f}, {gt.target_pixel_y:.1f}), "
                      f"error={error:.2f}px, conf={det.confidence:.3f}")

    assert detections >= 25, f"Expected >=25 detections, got {detections}"
    print(f"  Detected beacon in {detections}/30 frames")
    print("Detector test passed.\n")


def _test_kalman():
    """Test the Kalman filter smooths noisy measurements."""
    print("Testing Kalman filter...")
    from sim import Simulator, CameraState, CircularMotion, SensorNoise

    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                            radius_deg=10.0, angular_speed_deg_s=8.0)
    noise = SensorNoise(sigma=15.0)
    sim = Simulator(cam=cam, motion=motion, sensor_noise=noise, beacon_sigma_px=4.0)
    detector = BeaconDetector()
    kalman = KalmanFilter2D(measurement_noise=10.0)

    raw_errors = []
    filtered_errors = []
    initialized = False

    for i in range(60):
        frame, gt = sim.step(i)
        det = detector.detect(frame)

        # Predict
        pred_x, pred_y = kalman.predict()

        if det.detected:
            if not initialized:
                kalman.initialize(det.cx, det.cy)
                initialized = True
            else:
                kalman.update(det.cx, det.cy)

            raw_error = np.sqrt((det.cx - gt.target_pixel_x)**2 +
                                (det.cy - gt.target_pixel_y)**2)
            raw_errors.append(raw_error)

        if initialized:
            est_x, est_y = kalman.get_position()
            filt_error = np.sqrt((est_x - gt.target_pixel_x)**2 +
                                 (est_y - gt.target_pixel_y)**2)
            filtered_errors.append(filt_error)

    if raw_errors and filtered_errors:
        raw_mean = np.mean(raw_errors)
        filt_mean = np.mean(filtered_errors)
        print(f"  Raw measurement error mean: {raw_mean:.2f}px")
        print(f"  Kalman-filtered error mean: {filt_mean:.2f}px")
        assert filt_mean <= raw_mean * 1.5, \
            "Kalman filter is making things worse!"
    print("Kalman filter test passed.\n")


def _test_state_machine():
    """Test state transitions through the full lifecycle."""
    print("Testing state machine...")

    tracker = Tracker(
        detector=BeaconDetector(),
        kalman=KalmanFilter2D(),
        acquire_threshold=3,
        lose_threshold=5,
        reacquire_timeout_frames=10,
    )

    # Verify initial state
    assert tracker.state == TrackerState.SEARCHING
    print(f"  Initial state: {tracker.state.value}")

    # Simulate detections to move through states
    # We need a frame with a beacon to trigger detection
    from sim import Simulator, CameraState, CircularMotion
    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                            radius_deg=5.0, angular_speed_deg_s=3.0)
    sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=5.0)

    states_seen = set()
    for i in range(90):
        frame, _ = sim.step(i)
        result = tracker.track(frame)
        states_seen.add(result.state)

        if result.state != tracker.state:
            print(f"  Frame {i}: state -> {result.state.value}")

        if i < 5 or result.state != tracker.state:
            if result.detected:
                print(f"  Frame {i}: state={result.state.value}, "
                      f"pos=({result.estimated_x:.1f}, {result.estimated_y:.1f}), "
                      f"dets={result.consecutive_detections}")

    # Should have seen SEARCHING (initial, verified above) and at least ACQUIRING + TRACKING
    assert TrackerState.TRACKING in states_seen, "Never entered TRACKING"
    print(f"  States observed: {[s.value for s in states_seen]}")
    print("State machine test passed.\n")


def _test_tracker_with_disturbances():
    """Test the full tracker pipeline with vibration and noise."""
    print("Testing tracker with disturbances...")
    from sim import Simulator, CameraState, CircularMotion, PlatformVibration, SensorNoise

    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                            radius_deg=8.0, angular_speed_deg_s=5.0)
    vibration = PlatformVibration(rms_deg=0.2)
    noise = SensorNoise(sigma=10.0)
    sim = Simulator(cam=cam, motion=motion, vibration=vibration,
                    sensor_noise=noise, beacon_sigma_px=4.0)
    tracker = Tracker(
        detector=BeaconDetector(min_area=2, max_area=300),
        kalman=KalmanFilter2D(measurement_noise=6.0),
        acquire_threshold=3,
        lose_threshold=5,
    )

    detections = 0
    tracking_frames = 0
    for i in range(60):
        frame, _ = sim.step(i)
        result = tracker.track(frame)
        if result.detected:
            detections += 1
        if result.state == TrackerState.TRACKING:
            tracking_frames += 1

    print(f"  Detections: {detections}/60")
    print(f"  Tracking frames: {tracking_frames}/60")
    print(f"  Final state: {tracker.state.value}")
    assert detections > 40, f"Too few detections: {detections}"
    print("Disturbance tracker test passed.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("detect.py — Self-tests")
    print("=" * 60)
    print()

    _test_detector()
    _test_kalman()
    _test_state_machine()
    _test_tracker_with_disturbances()

    print("=" * 60)
    print("All self-tests passed. detect.py is ready to use.")
    print("=" * 60)
