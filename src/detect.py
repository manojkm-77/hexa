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
from abc import ABC, abstractmethod
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
    beacon_id: str = ""      # beacon identity


# ──────────────────────────────────────────────────────────────────────────
# DetectorBase — abstract interface for all beacon detectors
# ──────────────────────────────────────────────────────────────────────────

class DetectorBase(ABC):
    """Common interface for all beacon detectors (classical, AI, etc.)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        """Detect the beacon in a single frame.

        Args:
            frame: BGR or grayscale image (np.ndarray)

        Returns:
            Detection with centroid, area, bbox, and confidence.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────
# Beacon detector — classical brightness + connected components
# ──────────────────────────────────────────────────────────────────────────

def _aspect_ratio(bbox: tuple) -> float:
    """Return the aspect ratio (max(w,h)/min(w,h)) of a (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return float('inf')
    return max(w, h) / min(w, h)


class BeaconDetector(DetectorBase):
    """
    Classical beacon detector.

    Pipeline: grayscale → adaptive threshold → morphological opening →
    connected components → filter by area → shape filter → select brightest centroid.

    Optionally applies HSV color filtering before the threshold step.
    """

    def __init__(self,
                 threshold_method: str = "fixed",  # "otsu" or "fixed"
                 fixed_threshold: int = 80,
                 otsu_min_threshold: int = 60,  # floor for Otsu mode
                 min_area: int = 2,
                 max_area: int = 500,
                 morph_kernel_size: int = 3,
                 morph_iterations: int = 1,
                 use_hsv: bool = False,
                 hsv_low: tuple = (0, 0, 150),
                 hsv_high: tuple = (180, 255, 255),
                 max_aspect_ratio: float = 3.0,
                 persistence_threshold: int = 0):
        self.threshold_method = threshold_method
        self.fixed_threshold = fixed_threshold
        self.otsu_min_threshold = otsu_min_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.morph_iterations = morph_iterations
        self._last_threshold = 0
        # HSV color filtering
        self.use_hsv = use_hsv
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        # Shape filtering
        self.max_aspect_ratio = max_aspect_ratio
        # Persistence filtering
        self.persistence_threshold = persistence_threshold
        self._previous_centroids: dict = {}  # {(cx_bin, cy_bin): consecutive_count}

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

        # 2b. Optional HSV color filtering (AND with grayscale mask)
        if self.use_hsv:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, np.array(self.hsv_low), np.array(self.hsv_high))
            binary = cv2.bitwise_and(binary, hsv_mask)

        # 3. Morphological opening (remove small noise speckles)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, self.morph_kernel,
            iterations=self.morph_iterations)

        # Morphological closing: fill small holes in detected blobs (PRD FR-DE2)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, self.morph_kernel,
            iterations=1)

        # 4. Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 1:
            # Only background label (0); no candidates
            self._previous_centroids = {}
            return Detection()

        # 5. Build candidate list with area and shape filtering
        candidates = []
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

            candidates.append((cx, cy, area, (int(x), int(y), int(w), int(h)), peak, confidence))

        # Shape filter: reject elongated blobs
        if self.max_aspect_ratio and self.max_aspect_ratio > 0:
            candidates = [(cx, cy, area, bbox, peak, conf) for cx, cy, area, bbox, peak, conf in candidates
                          if _aspect_ratio(bbox) <= self.max_aspect_ratio]

        if not candidates:
            self._previous_centroids = {}
            return Detection()

        # 6. Select the brightest candidate
        best_detection = Detection()
        best_brightness = -1
        for cx, cy, area, bbox, peak, conf in candidates:
            if peak > best_brightness:
                best_brightness = peak
                best_detection = Detection(
                    detected=True,
                    cx=float(cx),
                    cy=float(cy),
                    area=float(area),
                    bbox=bbox,
                    confidence=float(conf),
                )

        # 7. Persistence filtering: reject candidates not seen in recent frames
        if self.persistence_threshold > 0 and best_detection.detected:
            # Bin the centroid to a coarse grid to match across frames
            bin_size = 10.0
            bin_key = (int(best_detection.cx / bin_size),
                       int(best_detection.cy / bin_size))

            # Update all bin counts: decay existing, boost the matched one
            new_counts = {}
            for key, count in self._previous_centroids.items():
                new_counts[key] = count - 1  # decay

            if bin_key in new_counts:
                new_counts[bin_key] = new_counts[bin_key] + 2  # boost
            else:
                new_counts[bin_key] = 1

            # Remove zero/negative entries
            self._previous_centroids = {k: v for k, v in new_counts.items() if v > 0}

            # Check persistence
            if new_counts.get(bin_key, 0) < self.persistence_threshold:
                best_detection.detected = False
        else:
            self._previous_centroids = {}

        return best_detection


# ──────────────────────────────────────────────────────────────────────────
# Alpha-beta filter — simpler alternative to Kalman
# ──────────────────────────────────────────────────────────────────────────

class AlphaBetaFilter:
    """Simpler alternative to KalmanFilter2D.

    State: [x, y, vx, vy]
    Uses fixed alpha (position) and beta (velocity) gains.
    Simpler than Kalman but sufficient for many tracking scenarios.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.1, dt: float = 1.0/30.0,
                 initial_x: float = 640.0, initial_y: float = 360.0):
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.x = initial_x
        self.y = initial_y
        self.vx = 0.0
        self.vy = 0.0
        self._initialized = False

    def predict(self) -> tuple[float, float]:
        """Predict next position. Called every frame."""
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        return (self.x, self.y)

    def update(self, x: float, y: float) -> None:
        """Update with measurement."""
        if not self._initialized:
            self.x = x
            self.y = y
            self._initialized = True
            return
        # Innovation
        dx = x - self.x
        dy = y - self.y
        # Update position
        self.x += self.alpha * dx
        self.y += self.alpha * dy
        # Update velocity
        self.vx += self.beta * dx / self.dt
        self.vy += self.beta * dy / self.dt

    def reset(self, x: float = 640.0, y: float = 360.0) -> None:
        """Reset filter state."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self._initialized = False

    def get_position(self) -> tuple[float, float]:
        """Return the current estimated position."""
        return (float(self.x), float(self.y))

    def get_velocity(self) -> tuple[float, float]:
        """Return the current estimated velocity."""
        return (float(self.vx), float(self.vy))

    def initialize(self, x: float, y: float) -> None:
        """Initialize filter at given position."""
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self._initialized = True

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> tuple[float, float]:
        return (self.vx, self.vy)


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
    IDLE = "IDLE"                            # Not started
    SEARCHING = "SEARCHING"                  # Sweeping for the beacon
    CANDIDATE_VERIFICATION = "CANDIDATE_VERIFICATION"  # Verifying a candidate detection
    ACQUIRING = "ACQUIRING"                  # Confirming detections before committing
    TRACKING = "TRACKING"                    # Locked on, continuous feedback
    REACQUIRING = "REACQUIRING"              # Lost track, predicting + searching locally
    FAILED = "FAILED"                        # Fatal error, requires reset


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
    tracked_id: str = ""         # identity of the tracked beacon


class Tracker:
    """
    Beacon tracker combining detector, Kalman filter, and state machine.

    The tracker manages state transitions and provides a clean interface
    for the controller: call track() each frame with the frame image,
    get back the estimated position, confidence, and current state.

    State transitions:
        IDLE  --start()-->  SEARCHING
        SEARCHING  --detection-->  CANDIDATE_VERIFICATION
        CANDIDATE_VERIFICATION  --verify_threshold consecutive detections-->  ACQUIRING
        CANDIDATE_VERIFICATION  --miss-->  SEARCHING
        ACQUIRING  --acquire_threshold consecutive detections (error OK)-->  TRACKING
        ACQUIRING  --timeout-->  SEARCHING
        TRACKING   --lose_threshold consecutive misses-->  REACQUIRING
        REACQUIRING --detection-->  TRACKING
        REACQUIRING --timeout-->  SEARCHING
        Any --fatal error-->  FAILED
        FAILED  --reset()-->  IDLE
    """

    def __init__(self,
                 detector: BeaconDetector = None,
                 kalman: KalmanFilter2D = None,
                 acquire_threshold: int = 3,       # consecutive detections to enter TRACKING
                 verify_threshold: int = 2,         # consecutive detections for CANDIDATE_VERIFICATION
                 acquire_error_threshold: float = 100.0,  # pixel error to enter TRACKING
                 lose_threshold: int = 5,            # consecutive misses before reacquire
                 reacquire_timeout_frames: int = 150, # ~5 seconds at 30 FPS
                 image_width: int = 1280,
                 image_height: int = 720):
        self.detector = detector or BeaconDetector()
        self.kalman = kalman or KalmanFilter2D()
        self.filter = self.kalman
        self.acquire_threshold = acquire_threshold
        self.verify_threshold = verify_threshold
        self.acquire_error_threshold = acquire_error_threshold
        self.lose_threshold = lose_threshold
        self.reacquire_timeout = reacquire_timeout_frames
        self.img_w = image_width
        self.img_h = image_height

        self.state = TrackerState.IDLE
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.reacquire_counter = 0
        self.last_detection: Optional[Detection] = None

    def start(self):
        """Transition from IDLE to SEARCHING. Call this to begin tracking."""
        if self.state == TrackerState.IDLE:
            self.state = TrackerState.SEARCHING

    def track(self, frame: np.ndarray) -> TrackerOutput:
        """
        Process one frame: detect, update Kalman, run state machine.

        Args:
            frame: BGR image from the simulator.

        Returns:
            TrackerOutput with state, estimated position, and confidence.
        """
        # IDLE: predict only (no detection, covariance grows)
        if self.state == TrackerState.IDLE:
            self.kalman.predict()
            est_x, est_y = self.kalman.get_position()
            return TrackerOutput(
                state=self.state,
                estimated_x=est_x,
                estimated_y=est_y,
                confidence=0.0,
                detected=False,
                consecutive_detections=self.consecutive_detections,
                consecutive_misses=self.consecutive_misses,
            )

        # FAILED: predict only (no detection, covariance grows)
        if self.state == TrackerState.FAILED:
            self.kalman.predict()
            est_x, est_y = self.kalman.get_position()
            return TrackerOutput(
                state=self.state,
                estimated_x=est_x,
                estimated_y=est_y,
                confidence=0.0,
                detected=False,
                consecutive_detections=self.consecutive_detections,
                consecutive_misses=self.consecutive_misses,
            )

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
                self.state = TrackerState.CANDIDATE_VERIFICATION
                self.consecutive_detections = 1

        elif self.state == TrackerState.CANDIDATE_VERIFICATION:
            if detected:
                if self.consecutive_detections >= self.verify_threshold:
                    self.state = TrackerState.ACQUIRING
                    self.consecutive_detections = 0
            else:
                # Allow 1 missed frame before reverting (noise tolerance)
                self.consecutive_misses += 1
                if self.consecutive_misses > 1:
                    self.state = TrackerState.SEARCHING
                    self.consecutive_detections = 0
                    self.consecutive_misses = 0

        elif self.state == TrackerState.ACQUIRING:
            if detected and self.consecutive_detections >= self.acquire_threshold:
                # Check pixel error between Kalman estimate and last detection
                pred_x, pred_y = self.kalman.get_position()
                error = math.hypot(pred_x - self.last_detection.cx,
                                   pred_y - self.last_detection.cy)
                if error < self.acquire_error_threshold:
                    self.state = TrackerState.TRACKING
                else:
                    # Error too large -- go back to SEARCHING
                    self.state = TrackerState.SEARCHING
                    self.consecutive_detections = 0
                    self.kalman.reset()
            elif not detected:
                self.state = TrackerState.SEARCHING
                self.consecutive_detections = 0

        elif self.state == TrackerState.TRACKING:
            if self.consecutive_misses >= self.lose_threshold:
                self.state = TrackerState.REACQUIRING
                self.reacquire_counter = 0

        elif self.state == TrackerState.REACQUIRING:
            if detected:
                # consecutive_detections already incremented in track()
                # Require verify_threshold consecutive detections before
                # returning to TRACKING (prevents single-frame false lock)
                if self.consecutive_detections >= self.verify_threshold:
                    self.state = TrackerState.TRACKING
                    self.reacquire_counter = 0
                    self.consecutive_misses = 0
            else:
                self.consecutive_detections = 0
                self.reacquire_counter += 1
                if self.reacquire_counter >= self.reacquire_timeout:
                    self.state = TrackerState.SEARCHING
                    self.reacquire_counter = 0
                    self.kalman.reset()

    def reset(self):
        """Reset tracker to initial state."""
        self.state = TrackerState.IDLE
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.reacquire_counter = 0
        self.last_detection = None
        self.kalman.reset()


# ──────────────────────────────────────────────────────────────────────────
# Multi-target tracking
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class MultiTrackerOutput:
    """Output of the multi-target tracker for one frame."""
    tracklets: dict  # {beacon_id: TrackerOutput}
    identity_switches: int  # count of identity switches this frame
    total_identity_switches: int  # cumulative


class MultiTracker:
    """
    Multi-target tracker using nearest-neighbor data association.

    Maintains one Tracker per beacon ID. On each frame, the detector finds
    all blobs in the image, and each detection is assigned to the closest
    existing track by pixel distance.

    Tracks identity switches: when a track picks up a detection that was
    previously associated with a different beacon (i.e., two tracks swap
    assignments).

    Usage:
        multi_tracker = MultiTracker(beacon_ids=["b1", "b2"])
        multi_tracker.start()
        for frame in frames:
            output = multi_tracker.track(frame)
    """

    def __init__(self,
                 beacon_ids: list,
                 detector: BeaconDetector = None,
                 acquire_threshold: int = 3,
                 verify_threshold: int = 2,
                 acquire_error_threshold: float = 100.0,
                 lose_threshold: int = 5,
                 reacquire_timeout_frames: int = 150,
                 image_width: int = 1280,
                 image_height: int = 720,
                 association_max_distance: float = 200.0):
        """
        Args:
            beacon_ids: List of unique beacon ID strings.
            detector: BeaconDetector instance used by all sub-trackers.
            acquire_threshold: Consecutive detections to enter TRACKING.
            verify_threshold: Consecutive detections for CANDIDATE_VERIFICATION.
            acquire_error_threshold: Pixel error threshold to enter TRACKING.
            lose_threshold: Consecutive misses before REACQUIRING.
            reacquire_timeout_frames: Frames before returning to SEARCHING.
            image_width: Width of the image.
            image_height: Height of the image.
            association_max_distance: Max pixel distance for nearest-neighbor
                association. Detections farther than this from any track are
                ignored (treated as clutter).
        """
        self.beacon_ids = list(beacon_ids)
        self.association_max_distance = association_max_distance
        self.total_identity_switches = 0
        self._prev_assignment = {}  # {detection_idx: beacon_id}

        self.trackers: dict = {}
        for bid in self.beacon_ids:
            self.trackers[bid] = Tracker(
                detector=detector or BeaconDetector(),
                kalman=KalmanFilter2D(),
                acquire_threshold=acquire_threshold,
                verify_threshold=verify_threshold,
                acquire_error_threshold=acquire_error_threshold,
                lose_threshold=lose_threshold,
                reacquire_timeout_frames=reacquire_timeout_frames,
                image_width=image_width,
                image_height=image_height,
            )

    def start(self):
        """Start all sub-trackers."""
        for tracker in self.trackers.values():
            tracker.start()

    def track(self, frame: np.ndarray) -> MultiTrackerOutput:
        """
        Process one frame for all tracked beacons.

        1. Run detection on the frame to find all blobs.
        2. For each existing track, predict the next position.
        3. Assign each detection to the nearest track (nearest-neighbor).
        4. Update each track with its assigned detection (or miss).
        5. Count identity switches (detection reassignment between tracks).

        Args:
            frame: BGR image from the simulator.

        Returns:
            MultiTrackerOutput with per-beacon TrackerOutput and identity
            switch counts.
        """
        # 1. Run the detector once to get all blobs
        detector = self.trackers[self.beacon_ids[0]].detector
        raw_detection = detector.detect(frame)

        # We need to find ALL blobs, not just the brightest one.
        # Re-implement blob extraction to get all candidates.
        all_detections = self._extract_all_detections(frame)

        # 2. Predict all tracks and collect their predicted positions
        predictions = {}  # {beacon_id: (pred_x, pred_y)}
        for bid in self.beacon_ids:
            tracker = self.trackers[bid]
            if tracker.state == TrackerState.IDLE or tracker.state == TrackerState.FAILED:
                predictions[bid] = tracker.kalman.get_position()
            else:
                predictions[bid] = tracker.kalman.predict()

        # 3. Nearest-neighbor assignment
        assignments = {}  # {beacon_id: Detection or None}
        assigned_dets = set()  # indices of detections already assigned

        for bid in self.beacon_ids:
            pred_x, pred_y = predictions[bid]
            best_dist = float('inf')
            best_idx = -1

            for i, det in enumerate(all_detections):
                if i in assigned_dets or not det.detected:
                    continue
                dist = math.hypot(det.cx - pred_x, det.cy - pred_y)
                if dist < best_dist and dist < self.association_max_distance:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                assignments[bid] = all_detections[best_idx]
                assigned_dets.add(best_idx)
            else:
                # Create an empty detection (miss)
                assignments[bid] = Detection()

        # 4. Count identity switches
        frame_switches = self._count_identity_switches(assignments)

        # 5. Update each track with its assigned detection
        tracklet_outputs = {}
        for bid in self.beacon_ids:
            tracker = self.trackers[bid]
            detection = assignments[bid]

            # Manually drive the tracker state machine with the assigned detection
            if tracker.state == TrackerState.IDLE or tracker.state == TrackerState.FAILED:
                est_x, est_y = tracker.kalman.get_position()
                tracklet_outputs[bid] = TrackerOutput(
                    state=tracker.state,
                    estimated_x=est_x,
                    estimated_y=est_y,
                    confidence=0.0,
                    detected=False,
                    consecutive_detections=tracker.consecutive_detections,
                    consecutive_misses=tracker.consecutive_misses,
                )
                continue

            # Update Kalman if we have a detection
            if detection.detected:
                tracker.kalman.update(detection.cx, detection.cy)
                tracker.consecutive_detections += 1
                tracker.consecutive_misses = 0
                tracker.last_detection = detection
            else:
                tracker.consecutive_detections = 0
                tracker.consecutive_misses += 1

            est_x, est_y = tracker.kalman.get_position()
            tracker._update_state(detection.detected)

            confidence = detection.confidence if detection.detected else 0.0
            if not detection.detected and tracker.last_detection is not None:
                confidence = tracker.last_detection.confidence * 0.5

            tracklet_outputs[bid] = TrackerOutput(
                state=tracker.state,
                estimated_x=est_x,
                estimated_y=est_y,
                confidence=confidence,
                detected=detection.detected,
                consecutive_detections=tracker.consecutive_detections,
                consecutive_misses=tracker.consecutive_misses,
            )

        # 6. Store current assignments for next-frame switch detection
        self._prev_assignment = {}
        for bid, det in assignments.items():
            if det.detected:
                # Store which beacon currently owns each detection region
                self._prev_assignment[bid] = (det.cx, det.cy)

        self.total_identity_switches += frame_switches

        return MultiTrackerOutput(
            tracklets=tracklet_outputs,
            identity_switches=frame_switches,
            total_identity_switches=self.total_identity_switches,
        )

    def _extract_all_detections(self, frame: np.ndarray) -> list:
        """
        Extract ALL blob detections from a frame, not just the brightest one.

        This is a multi-target extension of the BeaconDetector that returns
        all qualifying candidates rather than picking the single best.
        """
        detector = self.trackers[self.beacon_ids[0]].detector

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if detector.threshold_method == "otsu":
            otsu_val, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = max(int(otsu_val), detector.otsu_min_threshold)
            _, binary = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY)
            threshold = otsu_val
        else:
            _, binary = cv2.threshold(
                gray, detector.fixed_threshold, 255, cv2.THRESH_BINARY)
            threshold = detector.fixed_threshold

        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, detector.morph_kernel,
            iterations=detector.morph_iterations)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < detector.min_area or area > detector.max_area:
                continue

            cx, cy = centroids[i]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            region = gray[y:y+h, x:x+w]
            if region.size == 0:
                continue
            peak = int(region.max())
            mean = float(region.mean())

            if threshold > 0:
                confidence = min(1.0, mean / threshold)
            else:
                confidence = min(1.0, mean / 100.0)

            detections.append(Detection(
                detected=True,
                cx=float(cx),
                cy=float(cy),
                area=float(area),
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(confidence),
            ))

        # If no detections found, return one empty detection
        if not detections:
            detections.append(Detection())

        return detections

    def _count_identity_switches(self, current_assignments: dict) -> int:
        """
        Count identity switches between the previous and current frame.

        An identity switch occurs when two beacons swap which detection
        they are tracking — e.g., beacon A was tracking detection at (100,100)
        and beacon B at (200,200), but now A tracks (200,200) and B tracks
        (100,100).

        We detect this by checking if the set of (beacon, detection-center)
        pairs changed in a way that indicates a swap.
        """
        if not self._prev_assignment:
            return 0

        # Build current assignment: {beacon_id: (cx, cy)} for detected beacons
        curr_detected = {}
        for bid, det in current_assignments.items():
            if det.detected:
                curr_detected[bid] = (det.cx, det.cy)

        if not curr_detected:
            return 0

        # Check for pairwise swaps: two beacons that both had detections
        # and swapped which detection center they're near
        switches = 0
        bids = list(curr_detected.keys())
        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                bid_a, bid_b = bids[i], bids[j]
                if bid_a not in self._prev_assignment or bid_b not in self._prev_assignment:
                    continue
                prev_a = self._prev_assignment[bid_a]
                prev_b = self._prev_assignment[bid_b]
                curr_a = curr_detected[bid_a]
                curr_b = curr_detected[bid_b]

                # Swap detected: prev_a was close to curr_a, now close to curr_b, and vice versa
                dist_a_to_b = math.hypot(curr_a[0] - prev_b[0], curr_a[1] - prev_b[1])
                dist_b_to_a = math.hypot(curr_b[0] - prev_a[0], curr_b[1] - prev_a[1])
                dist_a_to_a = math.hypot(curr_a[0] - prev_a[0], curr_a[1] - prev_a[1])
                dist_b_to_b = math.hypot(curr_b[0] - prev_b[0], curr_b[1] - prev_b[1])

                # If cross-distances are shorter than same-beacon distances, it's a swap
                if (dist_a_to_b < dist_a_to_a and dist_b_to_a < dist_b_to_b and
                        dist_a_to_b < self.association_max_distance):
                    switches += 1

        return switches

    def reset(self):
        """Reset all sub-trackers."""
        for tracker in self.trackers.values():
            tracker.reset()
        self.total_identity_switches = 0
        self._prev_assignment = {}


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
    assert tracker.state == TrackerState.IDLE
    print(f"  Initial state: {tracker.state.value}")

    # Start the tracker
    tracker.start()
    assert tracker.state == TrackerState.SEARCHING
    print(f"  After start(): {tracker.state.value}")

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

    # Should have seen SEARCHING (verified above), CANDIDATE_VERIFICATION, ACQUIRING, and TRACKING
    assert TrackerState.TRACKING in states_seen, "Never entered TRACKING"
    assert TrackerState.ACQUIRING in states_seen, "Never entered ACQUIRING"
    assert TrackerState.CANDIDATE_VERIFICATION in states_seen, "Never entered CANDIDATE_VERIFICATION"
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
    tracker.start()

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
