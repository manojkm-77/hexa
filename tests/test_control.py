"""Tests for the control module: PIDController, PanTiltController."""

import numpy as np
import pytest

from control import (
    PIDController,
    PanTiltController,
    ControllerOutput,
)
from sim import CameraState


# ---------------------------------------------------------------------------
# PIDController: convergence to zero error
# ---------------------------------------------------------------------------

class TestPIDConvergence:
    """PIDController should drive error toward zero over time."""

    def test_proportional_only_converges(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=0.5)

        current = 0.0
        target = 10.0
        for _ in range(90):  # 3 seconds at 30 FPS
            error = target - current
            new_output, _ = pid.compute(error, current)
            current = new_output

        final_error = abs(target - current)
        assert final_error < 2.0, f"P-only did not converge: error={final_error:.2f}"

    def test_pid_with_integral_converges(self):
        """With integral gain, the controller should eliminate steady-state error."""
        pid = PIDController(kp=0.8, ki=0.3, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=0.0)

        current = 0.0
        target = 15.0
        for _ in range(300):  # 10 seconds
            error = target - current
            new_output, _ = pid.compute(error, current)
            current = new_output

        final_error = abs(target - current)
        assert final_error < 2.0, f"PID with integral did not converge: error={final_error:.2f}"

    def test_convergence_from_negative(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=0.5)
        current = 0.0
        target = -20.0
        for _ in range(120):
            error = target - current
            new_output, _ = pid.compute(error, current)
            current = new_output

        final_error = abs(target - current)
        assert final_error < 3.0, f"Did not converge to negative target: error={final_error:.2f}"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Rate limiter should prevent instantaneous jumps in output."""

    def test_large_error_is_rate_limited(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0)

        new_output, rate = pid.compute(100.0, 0.0)
        max_delta = 60.0 / 30.0  # 2 degrees per frame
        assert abs(new_output) <= max_delta + 0.01, \
            f"Rate limit violated: output={new_output:.2f}, max_delta={max_delta:.2f}"

    def test_rate_limit_is_consistent(self):
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0)

        max_delta = 60.0 / 30.0
        current = 0.0
        for _ in range(10):
            new_output, _ = pid.compute(1000.0, current)
            actual_delta = abs(new_output - current)
            assert actual_delta <= max_delta + 0.01, \
                f"Rate limit violated at step: delta={actual_delta:.3f}"
            current = new_output

    def test_small_error_not_rate_limited(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=0.0)

        new_output, _ = pid.compute(1.0, 0.0)
        # A small error with kp=1 should produce a small output, not rate-limited
        expected_raw = 1.0 * (1.0 / 30.0)
        assert abs(new_output - expected_raw) < 0.01

    def test_rate_limit_across_steps(self):
        """Output should change smoothly across multiple steps."""
        pid = PIDController(kp=5.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=30.0)
        max_delta = 30.0 / 30.0

        current = 0.0
        for i in range(20):
            new_output, _ = pid.compute(100.0, current)
            delta = abs(new_output - current)
            assert delta <= max_delta + 0.01, f"Step {i}: delta={delta:.3f}"
            current = new_output


# ---------------------------------------------------------------------------
# Deadband
# ---------------------------------------------------------------------------

class TestDeadband:
    """Deadband should ignore small errors, leaving output unchanged."""

    def test_small_error_ignored(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=2.0)

        new_output, rate = pid.compute(0.5, 10.0)
        assert new_output == 10.0, \
            f"Deadband should ignore error 0.5 < 2.0: output={new_output}"

    def test_large_error_acted_on(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=2.0)

        new_output, rate = pid.compute(5.0, 10.0)
        assert new_output != 10.0, "Deadband should NOT ignore error 5.0 > 2.0"

    def test_error_at_deadband_boundary(self):
        """Error exactly at the deadband threshold: abs(error) < deadband → ignored."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=2.0)

        # abs(1.999) < 2.0 → should be zeroed
        new_output, _ = pid.compute(1.999, 10.0)
        assert new_output == 10.0

        # abs(2.001) >= 2.0 → should NOT be zeroed
        new_output2, _ = pid.compute(2.001, 10.0)
        assert new_output2 != 10.0

    def test_zero_deadband_always_acts(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1 / 30.0,
                            rate_limit_deg_s=60.0, deadband=0.0)

        new_output, _ = pid.compute(0.001, 0.0)
        # Even tiny error should produce some output with deadband=0
        assert new_output != 0.0


# ---------------------------------------------------------------------------
# PanTiltController: pixel error → angular commands
# ---------------------------------------------------------------------------

class TestPanTiltController:
    """PanTiltController should convert pixel error to angular commands correctly."""

    def setup_method(self):
        self.cam = CameraState(width=1280, height=720,
                               h_fov_deg=60.0, v_fov_deg=40.0,
                               pan_deg=0.0, tilt_deg=0.0)
        self.controller = PanTiltController(
            cam=self.cam, kp=0.8, ki=0.0, kd=0.0,
            dt=1 / 30.0,
            pan_range=(-180.0, 180.0),
            tilt_range=(-30.0, 90.0),
            rate_limit_deg_s=60.0,
            deadband_pixels=3.0,
        )

    def test_target_at_center_produces_zero_error(self):
        """Target at image center should produce zero angular error."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx,
            target_pixel_y=self.cam.cy,
            cam_pan=0.0, cam_tilt=0.0,
        )
        assert abs(output.error_pan_deg) < 0.01
        assert abs(output.error_tilt_deg) < 0.01
        assert output.error_pixels < 0.01

    def test_target_to_right_produces_positive_pan_error(self):
        """A target to the right of center should produce a positive pan error."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx + 100,
            target_pixel_y=self.cam.cy,
            cam_pan=0.0, cam_tilt=0.0,
        )
        assert output.error_pan_deg > 0, f"Expected positive pan error, got {output.error_pan_deg}"

    def test_target_above_produces_positive_tilt_error(self):
        """A target above center (lower pixel y) should produce positive tilt error."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx,
            target_pixel_y=self.cam.cy - 100,  # lower y = above center
            cam_pan=0.0, cam_tilt=0.0,
        )
        assert output.error_tilt_deg > 0, f"Expected positive tilt error, got {output.error_tilt_deg}"

    def test_pixel_to_angular_conversion_correct(self):
        """Pixel error should convert to angular error via deg_per_pixel."""
        px_offset = 100.0
        output = self.controller.compute(
            target_pixel_x=self.cam.cx + px_offset,
            target_pixel_y=self.cam.cy,
            cam_pan=0.0, cam_tilt=0.0,
        )
        expected_pan_error = px_offset * self.cam.deg_per_pixel_x
        assert abs(output.error_pan_deg - expected_pan_error) < 1e-6

    def test_tilt_conversion_correct(self):
        """Pixel y offset should convert to angular tilt error via deg_per_pixel."""
        py_offset = 50.0
        output = self.controller.compute(
            target_pixel_x=self.cam.cx,
            target_pixel_y=self.cam.cy - py_offset,  # above center
            cam_pan=0.0, cam_tilt=0.0,
        )
        expected_tilt_error = py_offset * self.cam.deg_per_pixel_y
        assert abs(output.error_tilt_deg - expected_tilt_error) < 1e-6

    def test_output_is_controller_output(self):
        output = self.controller.compute(
            target_pixel_x=800, target_pixel_y=400,
            cam_pan=0.0, cam_tilt=0.0,
        )
        assert isinstance(output, ControllerOutput)
        assert isinstance(output.pan_cmd_deg, float)
        assert isinstance(output.tilt_cmd_deg, float)
        assert isinstance(output.error_pixels, float)

    def test_commands_move_toward_target(self):
        """Pan/tilt commands should move toward reducing pixel error."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx + 200,  # target is right of center
            target_pixel_y=self.cam.cy,
            cam_pan=0.0, cam_tilt=0.0,
        )
        # With kp=0.8, the pan command should be a positive correction
        assert output.pan_cmd_deg > 0.0

    def test_deadband_in_pixels(self):
        """A target within deadband pixels of center should not cause movement."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx + 1.0,  # 1 pixel offset, within 3px deadband
            target_pixel_y=self.cam.cy,
            cam_pan=5.0, cam_tilt=5.0,
        )
        # The deadband suppresses the error inside the PID, so the camera
        # should not move: pan_cmd should equal the input cam_pan.
        assert output.pan_cmd_deg == 5.0, \
            f"Pan should stay at 5.0 with sub-deadband error, got {output.pan_cmd_deg}"
        assert output.tilt_cmd_deg == 5.0

    def test_reset(self):
        """After reset, controller should return to default state."""
        self.controller.compute(
            target_pixel_x=1000, target_pixel_y=500,
            cam_pan=0.0, cam_tilt=0.0,
        )
        self.controller.reset()
        assert self.controller.current_pan == 0.0
        assert self.controller.current_tilt == 0.0

    def test_rate_limiting_applied(self):
        """Very large pixel offset should be rate-limited."""
        output = self.controller.compute(
            target_pixel_x=self.cam.cx + 600,  # huge offset
            target_pixel_y=self.cam.cy,
            cam_pan=0.0, cam_tilt=0.0,
        )
        max_pan_change = 60.0 / 30.0  # rate_limit * dt
        actual_change = abs(output.pan_cmd_deg - 0.0)
        assert actual_change <= max_pan_change + 0.1, \
            f"Pan rate limit violated: change={actual_change:.3f}, max={max_pan_change:.3f}"
