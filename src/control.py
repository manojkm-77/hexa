"""
control.py — Pan-tilt controller and state-machine integration
for the FSOC coarse-alignment simulator.

Converts image-space tracking error into angular pan/tilt commands.
Uses a PID controller with saturation, rate limits, deadband, and anti-windup.

The controller receives ONLY the tracker's estimated position and the camera
state. It never sees ground truth.

Dependencies: numpy
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import math

# Import from sibling modules
from sim import CameraState, pixel_to_angular, angular_to_pixel
from detect import Tracker, TrackerState, TrackerOutput


# ──────────────────────────────────────────────────────────────────────────
# Controller output
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerOutput:
    """Output of the pan-tilt controller for one frame."""
    pan_cmd_deg: float       # commanded pan angle
    tilt_cmd_deg: float      # commanded tilt angle
    pan_rate_cmd_deg_s: float  # commanded pan rate (deg/s)
    tilt_rate_cmd_deg_s: float # commanded tilt rate (deg/s)
    error_pan_deg: float     # angular error in pan
    error_tilt_deg: float    # angular error in tilt
    error_pixels: float      # Euclidean pixel error from image center
    saturated: bool           # whether rate limiting was applied
    pan_saturated: bool = False   # pan axis rate-limited (PRD FR-CT10)
    tilt_saturated: bool = False  # tilt axis rate-limited (PRD FR-CT10)


# ──────────────────────────────────────────────────────────────────────────
# PID Controller with saturation, rate limits, deadband, anti-windup
# ──────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    PID controller for a single axis (pan or tilt).

    Features:
      - Proportional, integral, derivative terms
      - Output saturation (absolute angle limits)
      - Rate limiting (max degrees per second)
      - Deadband (ignore errors below a threshold)
      - Anti-windup (clamp integral term when saturated)
    """

    def __init__(self,
                 kp: float = 0.8,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 dt: float = 1.0 / 30.0,
                 output_min: float = -180.0,
                 output_max: float = 180.0,
                 rate_limit_deg_s: float = 60.0,
                 deadband: float = 0.0,
                 integral_limit: float = 30.0):
        """
        Args:
            kp: Proportional gain.
            ki: Integral gain (start at 0, add only if steady-state offset).
            kd: Derivative gain.
            dt: Fixed time step (seconds).
            output_min, output_max: Absolute output limits (degrees).
            rate_limit_deg_s: Maximum rate of change (deg/s).
            deadband: Errors below this (in degrees) are ignored.
            integral_limit: Anti-windup clamp on the integral term.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.rate_limit = rate_limit_deg_s
        self.deadband = deadband
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def compute(self, error: float, current_output: float) -> tuple[float, float]:
        """
        Compute the next output for this axis.

        Args:
            error: Current error (target - current) in degrees.
            current_output: The current command (for rate limiting).

        Returns:
            (new_output, rate_command) where new_output is the absolute
            angle command and rate_command is the rate (deg/s) used.
        """
        # Deadband
        if abs(error) < self.deadband:
            error = 0.0

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        # Derivative
        d_term = self.kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        # Raw output
        raw_output = current_output + (p_term + i_term + d_term) * self.dt

        # Rate limiting
        max_delta = self.rate_limit * self.dt
        delta = raw_output - current_output
        saturated = False
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
            saturated = True

        new_output = current_output + delta

        # Output saturation
        new_output = np.clip(new_output, self.output_min, self.output_max)
        rate_cmd = delta / self.dt

        self._prev_output = new_output
        return float(new_output), float(rate_cmd)

    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Pan-Tilt Controller — combines two PID controllers
# ──────────────────────────────────────────────────────────────────────────

class PanTiltController:
    """
    Pan-tilt controller that converts image-space error to angular commands.

    Uses a PID controller for each axis. The error conversion:
        error_pan  = (target_x - cx) / width  * h_fov
        error_tilt = (target_y - cy) / height * v_fov

    The controller maintains the current pan/tilt state and applies rate
    limits and saturation.
    """

    def __init__(self,
                 cam: CameraState,
                 kp: float = 0.8,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 dt: float = 1.0 / 30.0,
                 pan_range: tuple = (-180.0, 180.0),
                 tilt_range: tuple = (-30.0, 90.0),
                 rate_limit_deg_s: float = 60.0,
                 deadband_pixels: float = 3.0):
        self.cam = cam
        self.dt = dt
        self.pan_range = pan_range
        self.tilt_range = tilt_range

        # Convert deadband from pixels to degrees
        deadband_pan_deg = deadband_pixels * cam.deg_per_pixel_x
        deadband_tilt_deg = deadband_pixels * cam.deg_per_pixel_y

        self.pan_pid = PIDController(
            kp=kp, ki=ki, kd=kd, dt=dt,
            output_min=pan_range[0], output_max=pan_range[1],
            rate_limit_deg_s=rate_limit_deg_s,
            deadband=deadband_pan_deg,
        )
        self.tilt_pid = PIDController(
            kp=kp, ki=ki, kd=kd, dt=dt,
            output_min=tilt_range[0], output_max=tilt_range[1],
            rate_limit_deg_s=rate_limit_deg_s,
            deadband=deadband_tilt_deg,
        )

        self.current_pan = 0.0
        self.current_tilt = 0.0

    def compute(self, target_pixel_x: float, target_pixel_y: float,
                cam_pan: float, cam_tilt: float) -> ControllerOutput:
        """
        Compute pan/tilt commands to center the target in the image.

        Args:
            target_pixel_x, target_pixel_y: Estimated target position (pixels).
            cam_pan, cam_tilt: Current camera pose (degrees).

        Returns:
            ControllerOutput with new pan/tilt commands and diagnostics.
        """
        # Update internal state from current camera pose
        self.current_pan = cam_pan
        self.current_tilt = cam_tilt

        # Pixel error from image center
        err_px_x = target_pixel_x - self.cam.cx
        # Image y increases downward, but angular elevation increases upward.
        # Negate y so a target above center produces a positive tilt error
        # (move camera up), matching the PID's positive-error convention.
        err_px_y = self.cam.cy - target_pixel_y
        error_pixels = np.sqrt(err_px_x**2 + err_px_y**2)

        # Convert pixel error to angular error
        error_pan_deg = err_px_x * self.cam.deg_per_pixel_x
        error_tilt_deg = err_px_y * self.cam.deg_per_pixel_y

        # Compute PID output for each axis
        new_pan, pan_rate = self.pan_pid.compute(error_pan_deg, self.current_pan)
        new_tilt, tilt_rate = self.tilt_pid.compute(error_tilt_deg, self.current_tilt)

        pan_saturated = abs(new_pan - self.current_pan) > self.pan_pid.rate_limit * self.dt - 0.01
        tilt_saturated = abs(new_tilt - self.current_tilt) > self.tilt_pid.rate_limit * self.dt - 0.01
        saturated = pan_saturated or tilt_saturated

        return ControllerOutput(
            pan_cmd_deg=new_pan,
            tilt_cmd_deg=new_tilt,
            pan_rate_cmd_deg_s=pan_rate,
            tilt_rate_cmd_deg_s=tilt_rate,
            error_pan_deg=error_pan_deg,
            error_tilt_deg=error_tilt_deg,
            error_pixels=float(error_pixels),
            saturated=saturated,
            pan_saturated=pan_saturated,
            tilt_saturated=tilt_saturated,
        )

    def reset(self):
        """Reset both PID controllers."""
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.current_pan = 0.0
        self.current_tilt = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Search controller — generates sweep patterns for SEARCHING state
# ──────────────────────────────────────────────────────────────────────────

class SearchPattern:
    """
    Generates a spiral search pattern for the SEARCHING state.

    The spiral expands from the current camera position, covering the
    full azimuth range. Tilt is kept near the horizon.
    """

    def __init__(self,
                 pan_range: tuple = (-180.0, 180.0),
                 tilt_range: tuple = (-30.0, 90.0),
                 sweep_rate_deg_s: float = 60.0,
                 tilt_step_deg: float = 10.0,
                 center_pan: float = 0.0,
                 center_tilt: float = 0.0):
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.sweep_rate = sweep_rate_deg_s
        self.tilt_step = tilt_step_deg
        self.center_pan = center_pan
        self.center_tilt = center_tilt
        self._direction = 1  # 1 = right, -1 = left
        self._current_tilt = center_tilt

    def next_command(self, current_pan: float, current_tilt: float,
                     dt: float) -> tuple[float, float]:
        """
        Generate the next search command.

        Returns (pan_cmd, tilt_cmd) — an absolute angle to move to.
        """
        # Sweep horizontally at the current tilt level
        new_pan = current_pan + self._direction * self.sweep_rate * dt

        # Bounce at pan range limits and step tilt
        if new_pan >= self.pan_range[1]:
            new_pan = self.pan_range[1]
            self._direction = -1
            self._current_tilt += self.tilt_step
        elif new_pan <= self.pan_range[0]:
            new_pan = self.pan_range[0]
            self._direction = 1
            self._current_tilt += self.tilt_step

        # Clamp tilt
        new_tilt = np.clip(self._current_tilt,
                           self.tilt_range[0], self.tilt_range[1])

        # If we've swept through all tilt levels, reset
        if new_tilt >= self.tilt_range[1] and self._direction == -1:
            self._current_tilt = self.tilt_range[0]
            new_tilt = self._current_tilt

        return float(new_pan), float(new_tilt)

    def reset(self):
        self._direction = 1
        self._current_tilt = self.center_tilt


# ──────────────────────────────────────────────────────────────────────────
# Integrated controller — combines search, tracking, and state machine
# ──────────────────────────────────────────────────────────────────────────

class IntegratedController:
    """
    Integrated controller that combines the tracker state machine with
    pan-tilt control. This is the top-level control object that main.py
    calls each frame.

    Usage:
        controller = IntegratedController(cam, tracker)
        for frame_id in range(num_frames):
            frame, gt = sim.step(frame_id)
            result = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
            sim.set_camera_pose(result.pan_cmd_deg, result.tilt_cmd_deg)
    """

    def __init__(self,
                 cam: CameraState,
                 tracker: Tracker,
                 kp: float = 0.8,
                 ki: float = 0.0,
                 kd: float = 0.1,
                 dt: float = 1.0 / 30.0,
                 pan_range: tuple = (-180.0, 180.0),
                 tilt_range: tuple = (-30.0, 90.0),
                 rate_limit_deg_s: float = 60.0,
                 deadband_pixels: float = 3.0):
        self.cam = cam
        self.tracker = tracker
        self.dt = dt
        self.controller = PanTiltController(
            cam=cam, kp=kp, ki=ki, kd=kd, dt=dt,
            pan_range=pan_range, tilt_range=tilt_range,
            rate_limit_deg_s=rate_limit_deg_s,
            deadband_pixels=deadband_pixels,
        )
        self.search = SearchPattern(
            pan_range=pan_range, tilt_range=tilt_range,
            sweep_rate_deg_s=rate_limit_deg_s,
        )
        self.tracking_center_threshold_px = cam.width * 0.10  # central 20% = 10% radius
        self.last_track_output = None  # TrackerOutput from most recent update()

    def update(self, frame: np.ndarray,
               current_pan: float, current_tilt: float) -> ControllerOutput:
        """
        Process one frame: track the beacon and generate pan/tilt commands.

        Args:
            frame: BGR image from the simulator.
            current_pan, current_tilt: Current camera pose.

        Returns:
            ControllerOutput with commands for the next frame.
        """
        # 1. Run the tracker (detector + Kalman + state machine)
        track_result = self.tracker.track(frame)
        self.last_track_output = track_result

        state = track_result.state
        est_x = track_result.estimated_x
        est_y = track_result.estimated_y

        # 2. Generate commands based on state
        if state == TrackerState.SEARCHING:
            # Sweep the camera to find the beacon
            new_pan, new_tilt = self.search.next_command(
                current_pan, current_tilt, self.dt)
            return ControllerOutput(
                pan_cmd_deg=new_pan,
                tilt_cmd_deg=new_tilt,
                pan_rate_cmd_deg_s=(new_pan - current_pan) / self.dt,
                tilt_rate_cmd_deg_s=(new_tilt - current_tilt) / self.dt,
                error_pan_deg=0.0,
                error_tilt_deg=0.0,
                error_pixels=0.0,
                saturated=False,
            )

        elif state == TrackerState.ACQUIRING:
            # Move toward the detected position, but cautiously
            cmd = self.controller.compute(est_x, est_y, current_pan, current_tilt)
            return cmd

        elif state == TrackerState.TRACKING:
            # Full feedback control
            cmd = self.controller.compute(est_x, est_y, current_pan, current_tilt)
            return cmd

        elif state == TrackerState.REACQUIRING:
            # Hold the predicted position (Kalman extrapolation)
            # The tracker's Kalman filter is already predicting — use that
            cmd = self.controller.compute(est_x, est_y, current_pan, current_tilt)
            return cmd

        # Fallback (shouldn't reach here)
        return ControllerOutput(
            pan_cmd_deg=current_pan,
            tilt_cmd_deg=current_tilt,
            pan_rate_cmd_deg_s=0.0,
            tilt_rate_cmd_deg_s=0.0,
            error_pan_deg=0.0,
            error_tilt_deg=0.0,
            error_pixels=0.0,
            saturated=False,
        )

    def reset(self):
        """Reset all controller state."""
        self.controller.reset()
        self.search.reset()
        self.tracker.reset()


# ──────────────────────────────────────────────────────────────────────────
# Self-tests
# ──────────────────────────────────────────────────────────────────────────

def _test_pid_basic():
    """Test basic PID controller behavior."""
    print("Testing PID controller...")
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1/30.0,
                        rate_limit_deg_s=60.0, deadband=0.5)

    # Start at 0, target is at 10 degrees
    current = 0.0
    errors = []
    for _ in range(90):  # 3 seconds
        error = 10.0 - current
        new_output, rate = pid.compute(error, current)
        current = new_output
        errors.append(error)

    # Should converge toward the target
    final_error = abs(10.0 - current)
    print(f"  P-only: target=10°, final position={current:.2f}°, error={final_error:.2f}°")
    assert final_error < 2.0, f"P-only didn't converge: error={final_error}"
    print("PID basic test passed.\n")


def _test_rate_limiting():
    """Test that the rate limiter prevents instantaneous jumps."""
    print("Testing rate limiting...")
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, dt=1/30.0,
                        rate_limit_deg_s=60.0)

    # Large error should be rate-limited
    new_output, rate = pid.compute(100.0, 0.0)
    max_delta = 60.0 / 30.0  # 2 degrees per frame
    assert abs(new_output) <= max_delta + 0.01, \
        f"Rate limit violated: output={new_output}, max={max_delta}"
    print(f"  Rate limited: requested 100° jump, got {new_output:.2f}° (max {max_delta:.2f}°)")
    print("Rate limiting test passed.\n")


def _test_deadband():
    """Test that the deadband suppresses small errors."""
    print("Testing deadband...")
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, dt=1/30.0,
                        rate_limit_deg_s=60.0, deadband=2.0)

    # Small error should be ignored
    new_output, rate = pid.compute(0.5, 10.0)
    assert new_output == 10.0, f"Deadband failed: output={new_output}, expected 10.0"
    print(f"  Deadband: error=0.5° (< 2° threshold), output unchanged at {new_output:.1f}°")

    # Large error should be acted on
    new_output, rate = pid.compute(5.0, 10.0)
    assert new_output != 10.0, "Deadband too aggressive"
    print(f"  Deadband: error=5.0° (> 2° threshold), output moved to {new_output:.1f}°")
    print("Deadband test passed.\n")


def _test_closed_loop():
    """
    Test the full closed loop: simulator -> tracker -> controller -> camera.

    This is the critical integration test — the beacon should be brought
    into the center of the image and held there.
    """
    print("Testing closed loop...")
    from sim import (Simulator, CameraState, CircularMotion)

    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(center_az_deg=5.0, center_el_deg=3.0,
                            radius_deg=8.0, angular_speed_deg_s=4.0)
    sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0, fps=30.0)

    from detect import BeaconDetector, KalmanFilter2D
    tracker = Tracker(
        detector=BeaconDetector(),
        kalman=KalmanFilter2D(measurement_noise=4.0),
        acquire_threshold=3,
        lose_threshold=5,
    )
    controller = IntegratedController(
        cam=cam, tracker=tracker, kp=0.8, ki=0.0, kd=0.1, dt=1/30.0,
    )

    errors = []
    states = []
    tracking_started = -1

    for i in range(300):  # 10 seconds
        frame, gt = sim.step(i)
        cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
        sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)

        # Compute pixel error from image center
        pixel_error = np.sqrt((gt.target_pixel_x - cam.cx)**2 +
                              (gt.target_pixel_y - cam.cy)**2)
        errors.append(pixel_error)

        state = cmd  # We need the tracker state
        track_result = tracker.track.__self__.state if hasattr(tracker.track, '__self__') else None
        # Actually let's just read tracker state directly
        states.append(tracker.state.value)

        if tracker.state.value == "TRACKING" and tracking_started < 0:
            tracking_started = i
            print(f"  First entered TRACKING at frame {i} ({i/30:.1f}s)")

        if i % 60 == 0 or i == 299:
            print(f"  Frame {i}: state={tracker.state.value}, "
                  f"error={pixel_error:.1f}px, "
                  f"pan={sim.cam.pan_deg:.1f}°, tilt={sim.cam.tilt_deg:.1f}°")

    # Analysis
    if tracking_started >= 0:
        # Look at errors after tracking started
        tracking_errors = errors[tracking_started:]
        mean_err = np.mean(tracking_errors)
        max_err = np.max(tracking_errors)
        min_err = np.min(tracking_errors)
        print(f"  After tracking started (frame {tracking_started}):")
        print(f"    Mean error: {mean_err:.1f}px")
        print(f"    Min error:  {min_err:.1f}px")
        print(f"    Max error:  {max_err:.1f}px")
        print(f"    Tracking rate: {sum(1 for s in states[tracking_started:] if s == 'TRACKING')}/{len(tracking_errors)} frames in TRACKING")

    # The beacon should be brought reasonably close to center
    last_30 = errors[-30:]
    assert np.mean(last_30) < 100, \
        f"Final error too high: {np.mean(last_30):.1f}px"
    print(f"  Final 30 frames mean error: {np.mean(last_30):.1f}px")
    print("Closed-loop test passed.\n")


def _test_closed_loop_with_disturbances():
    """Test the closed loop with vibration and noise."""
    print("Testing closed loop with disturbances...")
    from sim import (Simulator, CameraState, CircularMotion,
                     PlatformVibration, SensorNoise)

    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(center_az_deg=3.0, center_el_deg=2.0,
                            radius_deg=6.0, angular_speed_deg_s=5.0)
    vibration = PlatformVibration(rms_deg=0.15)
    noise = SensorNoise(sigma=8.0)
    sim = Simulator(cam=cam, motion=motion, vibration=vibration,
                    sensor_noise=noise, beacon_sigma_px=4.0, fps=30.0)

    from detect import BeaconDetector, KalmanFilter2D
    tracker = Tracker(
        detector=BeaconDetector(min_area=2, max_area=200),
        kalman=KalmanFilter2D(measurement_noise=8.0),
        acquire_threshold=3,
        lose_threshold=5,
    )
    controller = IntegratedController(
        cam=cam, tracker=tracker, kp=0.8, ki=0.0, kd=0.1, dt=1/30.0,
    )

    errors = []
    for i in range(300):
        frame, gt = sim.step(i)
        cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
        sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)
        pixel_error = np.sqrt((gt.target_pixel_x - cam.cx)**2 +
                              (gt.target_pixel_y - cam.cy)**2)
        errors.append(pixel_error)

    last_30 = errors[-30:]
    print(f"  Final 30 frames: mean={np.mean(last_30):.1f}px, "
          f"max={np.max(last_30):.1f}px")
    print(f"  Final state: {tracker.state.value}")
    assert np.mean(last_30) < 150, \
        f"Error too high with disturbances: {np.mean(last_30):.1f}px"
    print("Disturbance closed-loop test passed.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("control.py — Self-tests")
    print("=" * 60)
    print()

    _test_pid_basic()
    _test_rate_limiting()
    _test_deadband()
    _test_closed_loop()
    _test_closed_loop_with_disturbances()

    print("=" * 60)
    print("All self-tests passed. control.py is ready to use.")
    print("=" * 60)
