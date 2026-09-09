"""Tests for the detection module: BeaconDetector, KalmanFilter2D, Tracker FSM."""

import numpy as np
import pytest

from detect import (
    BeaconDetector,
    Detection,
    KalmanFilter2D,
    Tracker,
    TrackerState,
    TrackerOutput,
)
from sim import (
    CameraState,
    Simulator,
    CircularMotion,
)


# ---------------------------------------------------------------------------
# BeaconDetector
# ---------------------------------------------------------------------------

class TestBeaconDetector:
    """Detector should find a bright blob and return nothing on empty frames."""

    def _make_beacon_frame(self, width=320, height=240, cx=160, cy=120,
                           sigma=5.0, peak=255.0, bg=10):
        """Create a synthetic grayscale-like BGR frame with a Gaussian beacon."""
        frame = np.full((height, width, 3), bg, dtype=np.uint8)
        # Draw a Gaussian blob
        xs = np.arange(width) - cx
        ys = np.arange(height) - cy
        xx, yy = np.meshgrid(xs, ys)
        gaussian = peak * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.float32) + gaussian, 0, 255).astype(np.uint8)
        frame[:, :, 1] = frame[:, :, 0]
        frame[:, :, 2] = frame[:, :, 0]
        return frame

    def test_finds_bright_blob(self):
        """A synthetic frame with a bright Gaussian should be detected near its center."""
        cx, cy = 160, 120
        frame = self._make_beacon_frame(cx=cx, cy=cy, sigma=5.0, peak=255.0, bg=10)
        detector = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        det = detector.detect(frame)

        assert det.detected is True
        # Centroid should be close to the true center
        assert abs(det.cx - cx) < 10.0, f"cx={det.cx}, expected near {cx}"
        assert abs(det.cy - cy) < 10.0, f"cy={det.cy}, expected near {cy}"
        assert det.area > 0
        assert det.confidence > 0

    def test_finds_blob_in_noise(self):
        """Detector should still find a strong beacon even with some background noise."""
        rng = np.random.default_rng(42)
        frame = self._make_beacon_frame(cx=200, cy=100, sigma=4.0, peak=255.0, bg=20)
        noise = rng.normal(0, 15, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        detector = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        det = detector.detect(frame)
        assert det.detected is True

    def test_returns_false_on_empty_frame(self):
        """An all-dark frame should produce no detection."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detector = BeaconDetector(fixed_threshold=80)
        det = detector.detect(frame)

        assert det.detected is False
        assert det.area == 0.0
        assert det.confidence == 0.0

    def test_returns_false_on_flat_frame(self):
        """A uniformly lit frame (no blob) should not be detected."""
        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        detector = BeaconDetector(fixed_threshold=80)
        det = detector.detect(frame)
        assert det.detected is False

    def test_detects_from_simulator_frame(self):
        """Integration test: render a frame from the simulator and detect the beacon."""
        cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
        motion = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                                radius_deg=3.0, angular_speed_deg_s=3.0)
        sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0)
        detector = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)

        detections = 0
        for i in range(10):
            frame, gt = sim.step(i)
            det = detector.detect(frame)
            if det.detected:
                detections += 1

        assert detections >= 5, f"Expected at least 5/10 detections, got {detections}"

    def test_detector_bbox_valid(self):
        """When detected, bbox should have positive width and height."""
        frame = self._make_beacon_frame(cx=160, cy=120, sigma=5.0, peak=255.0, bg=10)
        detector = BeaconDetector(fixed_threshold=80, min_area=2, max_area=500)
        det = detector.detect(frame)
        if det.detected:
            x, y, w, h = det.bbox
            assert w > 0
            assert h > 0


# ---------------------------------------------------------------------------
# KalmanFilter2D
# ---------------------------------------------------------------------------

class TestKalmanFilter2D:
    """Kalman filter should converge toward repeated measurements and handle gaps."""

    def test_prediction_advances_state(self):
        kf = KalmanFilter2D(initial_x=100.0, initial_y=200.0)
        kf.initialize(100.0, 200.0)
        px, py = kf.predict()
        # With zero velocity, prediction should stay near initial position
        assert abs(px - 100.0) < 1.0
        assert abs(py - 200.0) < 1.0

    def test_update_moves_toward_measurement(self):
        kf = KalmanFilter2D(initial_x=100.0, initial_y=100.0,
                            measurement_noise=4.0)
        kf.initialize(100.0, 100.0)

        # Repeatedly update with the same measurement far away
        for _ in range(50):
            kf.predict()
            kf.update(200.0, 200.0)

        x, y = kf.get_position()
        # The filter should have moved significantly toward 200, 200
        assert x > 150.0, f"Expected x > 150, got {x}"
        assert y > 150.0, f"Expected y > 150, got {y}"

    def test_prediction_without_update(self):
        """During a detection gap, prediction should extrapolate from velocity."""
        kf = KalmanFilter2D(initial_x=0.0, initial_y=0.0)
        kf.initialize(0.0, 0.0)

        # Give it some velocity through updates
        kf.predict()
        kf.update(10.0, 10.0)
        kf.predict()
        kf.update(20.0, 20.0)

        # Now run several predictions without updates
        for _ in range(5):
            kf.predict()

        x, y = kf.get_position()
        # Should have moved forward from 20, 20
        assert x > 20.0, f"Expected x > 20 after extrapolation, got {x}"
        assert y > 20.0, f"Expected y > 20 after extrapolation, got {y}"

    def test_get_velocity(self):
        kf = KalmanFilter2D()
        kf.initialize(0.0, 0.0)

        # Update with increasing positions to build velocity
        kf.predict()
        kf.update(10.0, 5.0)
        kf.predict()
        kf.update(20.0, 10.0)

        vx, vy = kf.get_velocity()
        # Velocity should be approximately 10 pixels/frame in x, 5 in y
        assert vx > 5.0, f"Expected vx > 5, got {vx}"
        assert vy > 2.0, f"Expected vy > 2, got {vy}"

    def test_reset(self):
        kf = KalmanFilter2D()
        kf.initialize(500.0, 500.0)
        kf.reset()
        x, y = kf.get_position()
        assert abs(x - 640.0) < 1e-6
        assert abs(y - 360.0) < 1e-6
        assert kf.initialized is False

    def test_covariance_shrinks_with_updates(self):
        """Repeated measurements should reduce the covariance."""
        kf = KalmanFilter2D(initial_x=0.0, initial_y=0.0,
                            measurement_noise=4.0)
        kf.initialize(0.0, 0.0)
        p_trace_before = np.trace(kf.P)

        for _ in range(20):
            kf.predict()
            kf.update(0.0, 0.0)

        p_trace_after = np.trace(kf.P)
        assert p_trace_after < p_trace_before, "Covariance should shrink with measurements"


# ---------------------------------------------------------------------------
# Tracker FSM transitions
# ---------------------------------------------------------------------------

class TestTrackerFSM:
    """Tracker state machine should transition through SEARCHING -> ACQUIRING -> TRACKING -> REACQUIRING."""

    def _make_frame_with_beacon(self, width=320, height=240, cx=160, cy=120):
        """Create a frame with a detectable beacon."""
        frame = np.full((height, width, 3), 10, dtype=np.uint8)
        xs = np.arange(width) - cx
        ys = np.arange(height) - cy
        xx, yy = np.meshgrid(xs, ys)
        gaussian = 255.0 * np.exp(-(xx**2 + yy**2) / (2 * 5.0**2))
        frame[:, :, 0] = np.clip(gaussian, 0, 255).astype(np.uint8)
        frame[:, :, 1] = frame[:, :, 0]
        frame[:, :, 2] = frame[:, :, 0]
        return frame

    def _make_empty_frame(self, width=320, height=240):
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_initial_state_is_searching(self):
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        assert tracker.state == TrackerState.SEARCHING

    def test_searching_to_acquiring_on_detection(self):
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        frame = self._make_frame_with_beacon()
        result = tracker.track(frame)

        # On first detection, should move to ACQUIRING
        assert result.state == TrackerState.ACQUIRING
        assert result.detected is True
        assert result.consecutive_detections == 1

    def test_acquiring_to_tracking_after_threshold(self):
        acquire_threshold = 3
        tracker = Tracker(
            acquire_threshold=acquire_threshold,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        frame = self._make_frame_with_beacon()

        # Feed enough detections to cross the acquire threshold
        for i in range(acquire_threshold + 1):
            result = tracker.track(frame)

        assert result.state == TrackerState.TRACKING

    def test_acquiring_resets_on_miss(self):
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        beacon_frame = self._make_frame_with_beacon()
        empty_frame = self._make_empty_frame()

        # Start acquiring
        tracker.track(beacon_frame)
        assert tracker.state == TrackerState.ACQUIRING

        # Miss resets to SEARCHING
        tracker.track(empty_frame)
        assert tracker.state == TrackerState.SEARCHING

    def test_tracking_to_reacquiring_on_misses(self):
        lose_threshold = 5
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=lose_threshold,
            reacquire_timeout_frames=10,
        )
        beacon_frame = self._make_frame_with_beacon()
        empty_frame = self._make_empty_frame()

        # Get to TRACKING
        for _ in range(4):
            tracker.track(beacon_frame)
        assert tracker.state == TrackerState.TRACKING

        # Lose track with consecutive misses
        for _ in range(lose_threshold):
            result = tracker.track(empty_frame)

        assert result.state == TrackerState.REACQUIRING

    def test_reacquiring_to_tracking_on_redetection(self):
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        beacon_frame = self._make_frame_with_beacon()
        empty_frame = self._make_empty_frame()

        # Get to TRACKING
        for _ in range(4):
            tracker.track(beacon_frame)

        # Lose track (enter REACQUIRING)
        for _ in range(6):
            tracker.track(empty_frame)
        assert tracker.state == TrackerState.REACQUIRING

        # Redetect → back to TRACKING
        result = tracker.track(beacon_frame)
        assert result.state == TrackerState.TRACKING

    def test_reacquiring_to_searching_on_timeout(self):
        timeout = 5
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=timeout,
        )
        beacon_frame = self._make_frame_with_beacon()
        empty_frame = self._make_empty_frame()

        # Get to TRACKING
        for _ in range(4):
            tracker.track(beacon_frame)

        # Lose track
        for _ in range(6):
            tracker.track(empty_frame)
        assert tracker.state == TrackerState.REACQUIRING

        # Keep missing until timeout
        for _ in range(timeout + 1):
            result = tracker.track(empty_frame)

        assert result.state == TrackerState.SEARCHING

    def test_tracker_output_fields(self):
        tracker = Tracker()
        frame = self._make_frame_with_beacon()
        result = tracker.track(frame)

        assert isinstance(result, TrackerOutput)
        assert isinstance(result.state, TrackerState)
        assert isinstance(result.estimated_x, float)
        assert isinstance(result.estimated_y, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.detected, bool)
        assert isinstance(result.consecutive_detections, int)
        assert isinstance(result.consecutive_misses, int)

    def test_tracker_with_simulator_full_lifecycle(self):
        """Integration: run tracker on simulator frames through multiple states."""
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        motion = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                                radius_deg=5.0, angular_speed_deg_s=3.0)
        sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=5.0)
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )

        states_seen = set()
        for i in range(90):
            frame, _ = sim.step(i)
            result = tracker.track(frame)
            states_seen.add(result.state)

        # Over 90 frames with a visible beacon, we should see at least ACQUIRING and TRACKING
        assert TrackerState.TRACKING in states_seen, "Never entered TRACKING"
        assert TrackerState.ACQUIRING in states_seen, "Never entered ACQUIRING"

    def test_tracker_reset(self):
        tracker = Tracker(
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        frame = self._make_frame_with_beacon()
        tracker.track(frame)
        tracker.track(frame)

        tracker.reset()
        assert tracker.state == TrackerState.SEARCHING
        assert tracker.consecutive_detections == 0
        assert tracker.consecutive_misses == 0
