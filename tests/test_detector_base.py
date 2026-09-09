"""Tests for DetectorBase abstract interface and AlphaBetaFilter."""

import numpy as np
import pytest
from detect import DetectorBase, BeaconDetector, AlphaBetaFilter, Detection


class TestDetectorBase:
    """Tests for the DetectorBase abstract interface."""

    def test_cannot_instantiate_directly(self):
        """DetectorBase is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract method"):
            DetectorBase()

    def test_beacon_detector_is_instance_of_detector_base(self):
        """BeaconDetector inherits from DetectorBase."""
        detector = BeaconDetector()
        assert isinstance(detector, DetectorBase)

    def test_beacon_detector_has_detect_method(self):
        """BeaconDetector exposes a detect() method returning Detection."""
        detector = BeaconDetector()
        # A blank dark frame should return an undetected result
        frame = np.zeros((720, 1280), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, Detection)


class TestAlphaBetaFilter:
    """Tests for the AlphaBetaFilter class."""

    def test_predict_returns_position(self):
        """predict() returns a (x, y) tuple."""
        filt = AlphaBetaFilter()
        pos = filt.predict()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_update_sets_position(self):
        """First update() call initializes the filter to the measurement."""
        filt = AlphaBetaFilter(initial_x=0.0, initial_y=0.0)
        filt.update(100.0, 200.0)
        assert filt.position == (100.0, 200.0)
        assert filt._initialized is True

    def test_predict_update_cycle_converges(self):
        """After repeated predict+update on a fixed point, position converges."""
        filt = AlphaBetaFilter(alpha=0.5, beta=0.1, dt=1.0/30.0)
        target_x, target_y = 500.0, 300.0
        for _ in range(50):
            filt.predict()
            filt.update(target_x, target_y)
        px, py = filt.position
        assert abs(px - target_x) < 1.0, f"x={px} did not converge to {target_x}"
        assert abs(py - target_y) < 1.0, f"y={py} did not converge to {target_y}"

    def test_reset_works(self):
        """reset() returns filter to default or specified state."""
        filt = AlphaBetaFilter(initial_x=0.0, initial_y=0.0)
        filt.update(500.0, 500.0)
        filt.vx = 10.0
        filt.vy = 20.0

        filt.reset()
        assert filt.position == (640.0, 360.0)
        assert filt.velocity == (0.0, 0.0)
        assert filt._initialized is False

        filt.reset(x=100.0, y=200.0)
        assert filt.position == (100.0, 200.0)

    def test_tracks_moving_target(self):
        """Filter tracks a linearly moving target without diverging."""
        filt = AlphaBetaFilter(alpha=0.6, beta=0.05, dt=1.0/30.0)
        # Target moves right at 30 px/frame = 900 px/s at 30 FPS
        errors = []
        for i in range(60):
            target_x = 100.0 + 30.0 * i
            target_y = 360.0
            filt.predict()
            filt.update(target_x, target_y)
            px, py = filt.position
            errors.append(abs(px - target_x) + abs(py - target_y))

        # After initial transient (first ~10 frames), error should stay small
        steady_state_errors = errors[10:]
        assert max(steady_state_errors) < 50.0, (
            f"Tracking error too large: max={max(steady_state_errors):.1f}")
        # Velocity estimate should approximate the true velocity (in px/s)
        # Target moves 30 px/frame at dt=1/30 => 900 px/s
        vx, vy = filt.velocity
        assert abs(vx - 900.0) < 50.0, f"vx={vx:.1f}, expected ~900"
        assert abs(vy) < 50.0, f"vy={vy:.1f}, expected ~0"
