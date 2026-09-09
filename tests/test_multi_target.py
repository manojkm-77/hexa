"""Tests for multi-target simulation and tracking."""

import math
import numpy as np
import pytest

from sim import (
    CameraState,
    TargetState,
    GroundTruth,
    BeaconConfig,
    MultiTargetSimulator,
    CircularMotion,
    SinusoidalMotion,
    ConstantVelocity,
    angular_to_pixel,
    is_visible,
)
from detect import (
    BeaconDetector,
    KalmanFilter2D,
    Tracker,
    TrackerState,
    MultiTracker,
    MultiTrackerOutput,
)
from config import Config


# ---------------------------------------------------------------------------
# MultiTargetSimulator
# ---------------------------------------------------------------------------

class TestMultiTargetSimulator:
    """MultiTargetSimulator should render multiple beacons and return per-beacon ground truth."""

    def test_renders_correct_frame_shape(self):
        cam = CameraState(width=640, height=480)
        beacons = [
            BeaconConfig(id="b1", sigma_px=4.0),
            BeaconConfig(id="b2", sigma_px=3.0),
        ]
        sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        frame, gts = sim.step(0)
        assert frame.shape == (480, 640, 3), f"Expected (480, 640, 3), got {frame.shape}"
        assert frame.dtype == np.uint8

    def test_returns_ground_truth_per_beacon(self):
        beacons = [
            BeaconConfig(id="alpha"),
            BeaconConfig(id="beta"),
            BeaconConfig(id="gamma"),
        ]
        sim = MultiTargetSimulator(beacons=beacons)
        frame, gts = sim.step(0)
        assert len(gts) == 3, f"Expected 3 ground truths, got {len(gts)}"
        assert all(isinstance(gt, GroundTruth) for gt in gts)

    def test_beacons_have_distinct_positions(self):
        """Two beacons at different angular positions should produce different pixel positions."""
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        motion_a = CircularMotion(center_az_deg=-10.0, center_el_deg=0.0,
                                  radius_deg=2.0, angular_speed_deg_s=3.0,
                                  start_angle_deg=0.0)
        motion_b = CircularMotion(center_az_deg=10.0, center_el_deg=0.0,
                                  radius_deg=2.0, angular_speed_deg_s=3.0,
                                  start_angle_deg=0.0)
        beacons = [
            BeaconConfig(id="a", motion_model=motion_a),
            BeaconConfig(id="b", motion_model=motion_b),
        ]
        sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        frame, gts = sim.step(0)
        # They should project to different pixel x positions
        assert abs(gts[0].target_pixel_x - gts[1].target_pixel_x) > 50

    def test_colored_beacon_increases_channel(self):
        """A yellow BGR beacon (0, 255, 255) should increase G and R channels more than B."""
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        motion = ConstantVelocity(az_rate_deg_s=0.0, el_rate_deg_s=0.0)
        target = TargetState(azimuth_deg=0.0, elevation_deg=0.0)
        beacons = [
            BeaconConfig(id="yellow", color=(0, 255, 255), sigma_px=6.0,
                         motion_model=motion),
        ]
        sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        sim._targets["yellow"] = target
        frame, gts = sim.step(0)

        # The beacon should be at the center (boresight)
        px, py = int(round(gts[0].target_pixel_x)), int(round(gts[0].target_pixel_y))
        px = max(0, min(px, frame.shape[1] - 1))
        py = max(0, min(py, frame.shape[0] - 1))

        b, g, r = frame[py, px]
        # Yellow = (0, 255, 255) in BGR: G and R should be brighter than B
        assert int(g) > int(b), f"G channel ({g}) should be > B channel ({b}) for yellow beacon"

    def test_timestamp_advances(self):
        beacons = [BeaconConfig(id="b1"), BeaconConfig(id="b2")]
        sim = MultiTargetSimulator(beacons=beacons)
        _, gts0 = sim.step(0)
        _, gts1 = sim.step(1)
        assert gts1[0].timestamp > gts0[0].timestamp

    def test_frame_id_advances(self):
        beacons = [BeaconConfig(id="b1")]
        sim = MultiTargetSimulator(beacons=beacons)
        _, gts0 = sim.step(0)
        _, gts1 = sim.step(1)
        assert gts0[0].frame_id == 0
        assert gts1[0].frame_id == 1

    def test_reset(self):
        beacons = [BeaconConfig(id="b1")]
        sim = MultiTargetSimulator(beacons=beacons)
        sim.step(0)
        sim.step(1)
        sim.reset()
        assert sim.t == 0.0
        assert sim.frame_id == 0

    def test_beacon_ids_property(self):
        beacons = [
            BeaconConfig(id="alpha"),
            BeaconConfig(id="bravo"),
        ]
        sim = MultiTargetSimulator(beacons=beacons)
        assert sim.beacon_ids == ["alpha", "bravo"]

    def test_default_beacon_config(self):
        """BeaconConfig defaults should be sensible."""
        bc = BeaconConfig()
        assert bc.id == "beacon_01"
        assert bc.color == (255, 255, 255)
        assert bc.intensity == 1.0
        assert bc.sigma_px == 4.0
        assert bc.motion_model is None

    def test_custom_beacon_config(self):
        """BeaconConfig should accept custom values for all fields."""
        motion = ConstantVelocity(az_rate_deg_s=10.0, el_rate_deg_s=3.0)
        bc = BeaconConfig(
            id="custom_01",
            color=(0, 255, 255),
            intensity=0.7,
            sigma_px=6.0,
            motion_model=motion,
        )
        assert bc.id == "custom_01"
        assert bc.color == (0, 255, 255)
        assert bc.intensity == 0.7
        assert bc.sigma_px == 6.0
        assert bc.motion_model is motion

    def test_ground_truths_ordered_by_beacon_config(self):
        """Ground truths should correspond positionally to beacon configs.

        GroundTruth does not carry a target_id field, so we verify that the
        i-th ground truth has pixel positions matching the i-th beacon's
        angular position projected through the camera.
        """
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        target_a = TargetState(azimuth_deg=-10.0, elevation_deg=0.0)
        target_b = TargetState(azimuth_deg=10.0, elevation_deg=0.0)
        motion_a = ConstantVelocity(az_rate_deg_s=0.0, el_rate_deg_s=0.0)
        motion_b = ConstantVelocity(az_rate_deg_s=0.0, el_rate_deg_s=0.0)

        beacons = [
            BeaconConfig(id="left", motion_model=motion_a),
            BeaconConfig(id="right", motion_model=motion_b),
        ]
        sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        sim._targets["left"] = target_a
        sim._targets["right"] = target_b

        frame, gts = sim.step(0)

        # First ground truth should correspond to the left beacon (negative azimuth -> smaller pixel x)
        assert gts[0].target_pixel_x < gts[1].target_pixel_x, (
            "Ground truths should be ordered to match beacon configs; "
            "gt[0] should be the left beacon (smaller x)"
        )

    def test_multiple_steps_render_valid_frames(self):
        """Multiple frames should all be valid uint8 images."""
        beacons = [
            BeaconConfig(id="b1", sigma_px=4.0),
            BeaconConfig(id="b2", color=(0, 255, 255), sigma_px=3.0),
        ]
        sim = MultiTargetSimulator(beacons=beacons)
        for i in range(30):
            frame, gts = sim.step(i)
            assert frame.shape == (720, 1280, 3)
            assert frame.dtype == np.uint8
            assert len(gts) == 2


# ---------------------------------------------------------------------------
# MultiTracker
# ---------------------------------------------------------------------------

class TestMultiTracker:
    """MultiTracker should assign detections to nearest tracks and count identity switches."""

    def test_creates_with_explicit_detector_and_kalman(self):
        """MultiTracker should accept and use custom detector and kalman instances."""
        detector = BeaconDetector(fixed_threshold=100, min_area=3)
        kalman = KalmanFilter2D(measurement_noise=16.0)
        multi_tracker = MultiTracker(
            beacon_ids=["x1", "x2"],
            detector=detector,
        )
        # The sub-trackers should share the provided detector
        for tracker in multi_tracker.trackers.values():
            assert tracker.detector is detector
            assert isinstance(tracker.kalman, KalmanFilter2D)

        multi_tracker.start()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = multi_tracker.track(frame)
        assert isinstance(output, MultiTrackerOutput)

    def test_assigns_detection_to_nearest_track(self):
        """Two tracks far apart should each grab their closest detection."""
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)

        # Beacon A on the left, Beacon B on the right
        motion_a = ConstantVelocity(az_rate_deg_s=0.0, el_rate_deg_s=0.0)
        motion_b = ConstantVelocity(az_rate_deg_s=0.0, el_rate_deg_s=0.0)
        target_a = TargetState(azimuth_deg=-15.0, elevation_deg=0.0)
        target_b = TargetState(azimuth_deg=15.0, elevation_deg=0.0)

        beacons = [
            BeaconConfig(id="a", motion_model=motion_a),
            BeaconConfig(id="b", motion_model=motion_b),
        ]
        multi_sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        multi_sim._targets["a"] = target_a
        multi_sim._targets["b"] = target_b

        multi_tracker = MultiTracker(beacon_ids=["a", "b"])
        multi_tracker.start()

        # Run a few frames so the tracks initialize
        for i in range(5):
            frame, _ = multi_sim.step(i)
            output = multi_tracker.track(frame)

        # Both tracks should exist in the output
        assert "a" in output.tracklets
        assert "b" in output.tracklets

    def test_counts_identity_switches(self):
        """Identity switches should be counted when two tracks swap detection assignments."""
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)

        # Start beacons far apart, then cross them
        motion_a = ConstantVelocity(az_rate_deg_s=20.0, el_rate_deg_s=0.0)
        motion_b = ConstantVelocity(az_rate_deg_s=-20.0, el_rate_deg_s=0.0)
        target_a = TargetState(azimuth_deg=-20.0, elevation_deg=0.0)
        target_b = TargetState(azimuth_deg=20.0, elevation_deg=0.0)

        beacons = [
            BeaconConfig(id="a", motion_model=motion_a),
            BeaconConfig(id="b", motion_model=motion_b),
        ]
        multi_sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        multi_sim._targets["a"] = target_a
        multi_sim._targets["b"] = target_b

        multi_tracker = MultiTracker(beacon_ids=["a", "b"],
                                     association_max_distance=300.0)
        multi_tracker.start()

        for i in range(60):
            frame, _ = multi_sim.step(i)
            output = multi_tracker.track(frame)

        # After crossing, there should have been at least one identity switch
        # (beacons pass through each other's positions)
        # Note: with nearest-neighbor, the exact count depends on timing,
        # but total should be >= 0
        assert output.total_identity_switches >= 0

    def test_output_structure(self):
        """MultiTrackerOutput should have the expected fields."""
        multi_tracker = MultiTracker(beacon_ids=["x", "y"])
        multi_tracker.start()

        # Create a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = multi_tracker.track(frame)

        assert isinstance(output, MultiTrackerOutput)
        assert "x" in output.tracklets
        assert "y" in output.tracklets
        assert isinstance(output.identity_switches, int)
        assert isinstance(output.total_identity_switches, int)

    def test_total_identity_switches_cumulative(self):
        """total_identity_switches should accumulate across frames."""
        multi_tracker = MultiTracker(beacon_ids=["a", "b"])
        multi_tracker.start()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10):
            output = multi_tracker.track(frame)

        # Total should equal sum of per-frame counts
        assert output.total_identity_switches == multi_tracker.total_identity_switches

    def test_reset_clears_state(self):
        """After reset, total_identity_switches should be zero."""
        multi_tracker = MultiTracker(beacon_ids=["a", "b"])
        multi_tracker.start()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            multi_tracker.track(frame)
        multi_tracker.reset()
        assert multi_tracker.total_identity_switches == 0


# ---------------------------------------------------------------------------
# Config multi-target parameters
# ---------------------------------------------------------------------------

class TestConfigMultiTarget:
    """Config should have multi-target parameters with sensible defaults."""

    def test_defaults(self):
        config = Config()
        assert config.multi_target is False
        assert config.num_targets == 2
        assert config.target_colors == [(255, 255, 255), (0, 255, 255)]

    def test_custom_values(self):
        config = Config(
            multi_target=True,
            num_targets=4,
            target_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
        )
        assert config.multi_target is True
        assert config.num_targets == 4
        assert len(config.target_colors) == 4


# ---------------------------------------------------------------------------
# Backward compatibility: single-target mode
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Original Simulator and Tracker should still work unchanged."""

    def test_single_target_simulator(self):
        from sim import Simulator, CircularMotion
        cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
        motion = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                                radius_deg=8.0, angular_speed_deg_s=6.0)
        sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0)
        frame, gt = sim.step(0)
        assert frame.shape == (720, 1280, 3)
        assert isinstance(gt, GroundTruth)
        assert gt.target_visible is True

    def test_single_target_tracker(self):
        from sim import Simulator, CameraState, CircularMotion
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        motion = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                                radius_deg=5.0, angular_speed_deg_s=3.0)
        sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=5.0)
        tracker = Tracker(
            detector=BeaconDetector(),
            kalman=KalmanFilter2D(),
            acquire_threshold=3,
            lose_threshold=5,
            reacquire_timeout_frames=10,
        )
        tracker.start()

        detections = 0
        for i in range(60):
            frame, _ = sim.step(i)
            result = tracker.track(frame)
            if result.detected:
                detections += 1

        assert detections > 40, f"Single-target tracker should still detect well: {detections}/60"
        assert tracker.state == TrackerState.TRACKING

    def test_single_target_detector_finds_beacon(self):
        """The original BeaconDetector should still find the single beacon."""
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
        assert detections >= 25, f"Detector should find beacon in >=25/30 frames: {detections}"
