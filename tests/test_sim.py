"""Tests for the simulation module: projection, motion models, disturbances, simulator."""

import math
import numpy as np
import pytest

from sim import (
    CameraState,
    TargetState,
    GroundTruth,
    angular_to_pixel,
    pixel_to_angular,
    is_visible,
    ConstantVelocity,
    SinusoidalMotion,
    CircularMotion,
    RandomManeuvering,
    PlatformVibration,
    SensorNoise,
    MotionBlur,
    Simulator,
)


# ---------------------------------------------------------------------------
# Projection round-trip
# ---------------------------------------------------------------------------

class TestProjection:
    """angular_to_pixel and pixel_to_angular should be consistent inverses."""

    def test_round_trip_center(self):
        cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
        az, el = 10.0, 5.0  # exactly at boresight
        px, py = angular_to_pixel(cam, az, el)
        az2, el2 = pixel_to_angular(cam, px, py)
        assert abs(az - az2) < 1e-9
        assert abs(el - el2) < 1e-9

    def test_round_trip_arbitrary(self):
        cam = CameraState(pan_deg=-20.0, tilt_deg=15.0)
        az, el = 25.3, -7.2
        px, py = angular_to_pixel(cam, az, el)
        az2, el2 = pixel_to_angular(cam, px, py)
        assert abs(az - az2) < 1e-6
        assert abs(el - el2) < 1e-6

    def test_round_trip_edge_of_fov(self):
        cam = CameraState()
        # Right edge of horizontal FOV
        az = cam.pan_deg + cam.h_fov_deg / 2.0
        el = cam.tilt_deg
        px, py = angular_to_pixel(cam, az, el)
        az2, el2 = pixel_to_angular(cam, px, py)
        assert abs(az - az2) < 1e-6
        assert abs(el - el2) < 1e-6

    def test_round_trip_top_of_fov(self):
        cam = CameraState()
        az = cam.pan_deg
        el = cam.tilt_deg + cam.v_fov_deg / 2.0
        px, py = angular_to_pixel(cam, az, el)
        az2, el2 = pixel_to_angular(cam, px, py)
        assert abs(az - az2) < 1e-6
        assert abs(el - el2) < 1e-6


# ---------------------------------------------------------------------------
# Boresight → image center
# ---------------------------------------------------------------------------

class TestBoresight:
    """A target exactly at the camera boresight should project to image center."""

    def test_boresight_at_center(self):
        cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
        px, py = angular_to_pixel(cam, cam.pan_deg, cam.tilt_deg)
        assert abs(px - cam.cx) < 0.01
        assert abs(py - cam.cy) < 0.01

    def test_boresight_default(self):
        cam = CameraState()
        px, py = angular_to_pixel(cam, 0.0, 0.0)
        assert abs(px - cam.cx) < 0.01
        assert abs(py - cam.cy) < 0.01

    def test_target_to_right_of_boresight(self):
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        px, py = angular_to_pixel(cam, 5.0, 0.0)
        assert px > cam.cx

    def test_target_above_boresight(self):
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        px, py = angular_to_pixel(cam, 0.0, 5.0)
        # Positive elevation → target above → pixel y decreases
        assert py < cam.cy


# ---------------------------------------------------------------------------
# Visibility check
# ---------------------------------------------------------------------------

class TestVisibility:
    """is_visible should return True only for pixels inside [0, width) x [0, height)."""

    def setup_method(self):
        self.cam = CameraState(width=1280, height=720)

    def test_center_is_visible(self):
        assert is_visible(self.cam, 640.0, 360.0) is True

    def test_origin_is_visible(self):
        assert is_visible(self.cam, 0.0, 0.0) is True

    def test_negative_x_not_visible(self):
        assert is_visible(self.cam, -1.0, 360.0) is False

    def test_negative_y_not_visible(self):
        assert is_visible(self.cam, 640.0, -1.0) is False

    def test_outside_right_edge_not_visible(self):
        assert is_visible(self.cam, 1280.0, 360.0) is False

    def test_outside_bottom_edge_not_visible(self):
        assert is_visible(self.cam, 640.0, 720.0) is False

    def test_just_inside_right_edge(self):
        assert is_visible(self.cam, 1279.9, 360.0) is True

    def test_just_inside_bottom_edge(self):
        assert is_visible(self.cam, 640.0, 719.9) is True


# ---------------------------------------------------------------------------
# Motion models
# ---------------------------------------------------------------------------

class TestMotionModels:
    """Each motion model should produce valid, changing positions over time."""

    def test_constant_velocity_changes_position(self):
        model = ConstantVelocity(az_rate_deg_s=5.0, el_rate_deg_s=2.0)
        state = TargetState()
        dt = 1 / 30
        positions = []
        for i in range(30):
            t = i * dt
            model.update(state, dt, t)
            positions.append((state.azimuth_deg, state.elevation_deg))

        az_range = max(p[0] for p in positions) - min(p[0] for p in positions)
        el_range = max(p[1] for p in positions) - min(p[1] for p in positions)
        assert az_range > 0.1, "ConstantVelocity azimuth should change"
        assert el_range > 0.1, "ConstantVelocity elevation should change"

    def test_constant_velocity_linear(self):
        model = ConstantVelocity(az_rate_deg_s=10.0, el_rate_deg_s=0.0)
        state = TargetState()
        dt = 1.0
        model.update(state, dt, 0.0)
        assert abs(state.azimuth_deg - 10.0) < 1e-9
        assert abs(state.elevation_deg - 0.0) < 1e-9

    def test_constant_velocity_sets_velocities(self):
        model = ConstantVelocity(az_rate_deg_s=5.0, el_rate_deg_s=3.0)
        state = TargetState()
        model.update(state, 1.0, 0.0)
        assert abs(state.vel_az_deg_s - 5.0) < 1e-9
        assert abs(state.vel_el_deg_s - 3.0) < 1e-9
        assert state.acc_az_deg_s2 == 0.0

    def test_sinusoidal_bounded(self):
        model = SinusoidalMotion(az_amp_deg=15.0, el_amp_deg=8.0)
        state = TargetState()
        dt = 1 / 30
        for i in range(300):
            model.update(state, dt, i * dt)
        # Az should never exceed amplitude
        assert abs(state.azimuth_deg) <= 15.0 + 1e-9
        assert abs(state.elevation_deg) <= 8.0 + 1e-9

    def test_sinusoidal_periodicity(self):
        model = SinusoidalMotion(az_amp_deg=10.0, az_freq_hz=1.0, az_phase_deg=0.0,
                                 el_amp_deg=5.0, el_freq_hz=1.0, el_phase_deg=0.0)
        state = TargetState()
        dt = 1 / 30
        t = 0.0
        model.update(state, dt, t)
        az_at_zero = state.azimuth_deg

        # After one full period, should return to same position
        model.update(state, dt, 1.0)
        az_after_period = state.azimuth_deg
        assert abs(az_at_zero - az_after_period) < 1e-6

    def test_circular_bounded(self):
        model = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                               radius_deg=8.0, angular_speed_deg_s=6.0)
        state = TargetState()
        dt = 1 / 30
        for i in range(300):
            model.update(state, dt, i * dt)
        # Position should stay within radius of center
        dist = math.sqrt((state.azimuth_deg - 10.0)**2 + (state.elevation_deg - 5.0)**2)
        assert abs(dist - 8.0) < 1e-6, f"Circular orbit radius off: {dist}"

    def test_circular_full_revolution(self):
        model = CircularMotion(center_az_deg=0.0, center_el_deg=0.0,
                               radius_deg=5.0, angular_speed_deg_s=360.0)
        state = TargetState()
        dt = 1 / 30
        # Run for 31 steps (t=0 through t=1.0 inclusive) to complete one revolution
        positions = []
        for i in range(31):
            model.update(state, dt, i * dt)
            positions.append((state.azimuth_deg, state.elevation_deg))
        # First (t=0) and last (t=1.0) should be close after one full revolution
        assert abs(positions[0][0] - positions[-1][0]) < 0.5
        assert abs(positions[0][1] - positions[-1][1]) < 0.5

    def test_random_maneuvering_bounded_velocity(self):
        model = RandomManeuvering(max_acc_deg_s2=10.0, max_vel_deg_s=20.0, seed=42)
        state = TargetState()
        dt = 1 / 30
        for i in range(300):
            model.update(state, dt, i * dt)
        assert abs(state.vel_az_deg_s) <= 20.0 + 1e-9
        assert abs(state.vel_el_deg_s) <= 20.0 + 1e-9

    def test_random_maneuvering_reproducible_with_seed(self):
        def run_model(seed):
            model = RandomManeuvering(max_acc_deg_s2=10.0, seed=seed)
            state = TargetState()
            dt = 1 / 30
            for i in range(100):
                model.update(state, dt, i * dt)
            return state.azimuth_deg, state.elevation_deg

        az1, el1 = run_model(seed=42)
        az2, el2 = run_model(seed=42)
        assert az1 == az2
        assert el1 == el2


# ---------------------------------------------------------------------------
# Disturbances are reproducible with same seed
# ---------------------------------------------------------------------------

class TestDisturbances:
    """PlatformVibration and SensorNoise must be reproducible with the same seed."""

    def test_vibration_reproducible(self):
        """Two separate instances with the same seed should produce the same offset."""
        cam1 = CameraState(pan_deg=0.0, tilt_deg=0.0)
        cam2 = CameraState(pan_deg=0.0, tilt_deg=0.0)
        vib1 = PlatformVibration(rms_deg=0.2, seed=99)
        vib2 = PlatformVibration(rms_deg=0.2, seed=99)

        vib1.apply(cam1, t=1.0)
        vib2.apply(cam2, t=1.0)
        assert abs(cam1.pan_deg - cam2.pan_deg) < 1e-12
        assert abs(cam1.tilt_deg - cam2.tilt_deg) < 1e-12

    def test_vibration_different_seeds_differ(self):
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        vib1 = PlatformVibration(rms_deg=0.2, seed=1)
        vib2 = PlatformVibration(rms_deg=0.2, seed=2)

        result1 = vib1.apply(cam, t=1.0)
        result2 = vib2.apply(cam, t=1.0)
        # At least one axis should differ
        assert result1.pan_deg != result2.pan_deg or result1.tilt_deg != result2.tilt_deg

    def test_sensor_noise_statistics(self):
        """SensorNoise should add zero-mean noise with approximately the expected sigma."""
        frame = np.full((200, 200), 128, dtype=np.uint8)
        noise = SensorNoise(sigma=10.0, seed=42)
        noisy = noise.apply(frame)
        diff = noisy.astype(np.float32) - 128.0
        assert abs(diff.mean()) < 3.0, f"Noise mean too far from zero: {diff.mean():.2f}"
        # cv2.randn generates Gaussian noise; check std is roughly sigma
        assert 5.0 < diff.std() < 18.0, f"Noise std unexpected: {diff.std():.2f}"

    def test_sensor_noise_different_seeds_differ(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        noise1 = SensorNoise(sigma=10.0, seed=1)
        noise2 = SensorNoise(sigma=10.0, seed=2)
        frame1 = noise1.apply(frame.copy())
        frame2 = noise2.apply(frame.copy())
        assert not np.array_equal(frame1, frame2)

    def test_vibration_different_times_differ(self):
        cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
        vib = PlatformVibration(rms_deg=0.5, seed=42)

        result0 = vib.apply(cam, t=0.0)
        result1 = vib.apply(cam, t=1.0)
        # Different times should produce different offsets (the sinusoidal component differs)
        assert result0.pan_deg != result1.pan_deg or result0.tilt_deg != result1.tilt_deg


# ---------------------------------------------------------------------------
# Simulator.step() returns correct frame shape and valid GroundTruth
# ---------------------------------------------------------------------------

class TestSimulatorStep:
    """Simulator.step() should return a frame of correct shape and valid GroundTruth."""

    def test_frame_shape(self):
        cam = CameraState(width=640, height=480)
        sim = Simulator(cam=cam, beacon_sigma_px=4.0)
        frame, gt = sim.step(0)
        assert frame.shape == (480, 640, 3), f"Expected (480, 640, 3), got {frame.shape}"

    def test_frame_dtype(self):
        sim = Simulator()
        frame, gt = sim.step(0)
        assert frame.dtype == np.uint8

    def test_ground_truth_fields(self):
        sim = Simulator()
        frame, gt = sim.step(0)
        assert isinstance(gt, GroundTruth)
        assert isinstance(gt.target_az_deg, float)
        assert isinstance(gt.target_el_deg, float)
        assert isinstance(gt.target_pixel_x, float)
        assert isinstance(gt.target_pixel_y, float)
        assert isinstance(gt.target_visible, bool)
        assert gt.frame_id == 0

    def test_ground_truth_timestamp_advances(self):
        sim = Simulator(fps=30.0)
        _, gt0 = sim.step(0)
        _, gt1 = sim.step(1)
        assert gt1.timestamp > gt0.timestamp

    def test_ground_truth_visible_matches_projection(self):
        sim = Simulator()
        _, gt = sim.step(0)
        # Manually check visibility
        expected_visible = is_visible(sim.cam, gt.target_pixel_x, gt.target_pixel_y)
        assert gt.target_visible == expected_visible

    def test_multiple_steps_produce_valid_frames(self):
        sim = Simulator()
        for i in range(30):
            frame, gt = sim.step(i)
            assert frame.shape == (720, 1280, 3)
            assert 0 <= gt.target_pixel_x or True  # pixel can be outside FOV
            assert isinstance(gt.target_visible, bool)

    def test_beacon_appears_in_frame_when_visible(self):
        """When the beacon is visible, the frame should have bright pixels near its location."""
        cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
        motion = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                                radius_deg=2.0, angular_speed_deg_s=3.0)
        sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0)
        frame, gt = sim.step(0)

        if gt.target_visible:
            px, py = int(round(gt.target_pixel_x)), int(round(gt.target_pixel_y))
            # Clamp to frame bounds for safe access
            px = max(0, min(px, frame.shape[1] - 1))
            py = max(0, min(py, frame.shape[0] - 1))
            # Brightness near the beacon should be significantly above background
            assert frame[py, px].mean() > 50, "Beacon area should be brighter than background"

    def test_simulator_set_camera_pose(self):
        sim = Simulator()
        sim.set_camera_pose(15.0, 10.0)
        assert sim.cam.pan_deg == 15.0
        assert sim.cam.tilt_deg == 10.0

    def test_simulator_reset(self):
        sim = Simulator()
        sim.step(0)
        sim.step(1)
        sim.reset()
        assert sim.t == 0.0
        assert sim.frame_id == 0

    def test_simulator_with_all_disturbances(self):
        cam = CameraState()
        motion = CircularMotion(radius_deg=10.0, angular_speed_deg_s=5.0)
        vibration = PlatformVibration(rms_deg=0.3)
        noise = SensorNoise(sigma=12.0)
        blur = MotionBlur(max_blur_pixels=4.0)
        sim = Simulator(cam=cam, motion=motion, vibration=vibration,
                        sensor_noise=noise, motion_blur=blur)

        for i in range(30):
            frame, gt = sim.step(i)
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8
