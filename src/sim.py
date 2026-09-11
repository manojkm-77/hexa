"""
sim.py — Virtual environment, target motion, camera, and rendering for the
FSOC coarse-alignment simulator.

Everything here produces synthetic frames and ground truth. The ground truth is
returned alongside each frame but must NEVER be passed to the detector or
controller — only to the logger and the dev-mode HUD overlay.

Coordinate system:
  - World coordinates are angular: azimuth (pan axis) and elevation (tilt axis),
    in degrees. Azimuth increases to the right, elevation increases upward.
  - Camera pose is (pan, tilt) in degrees.
  - Image coordinates are pixels: x to the right, y downward (standard image convention).

Dependencies: numpy, cv2 (OpenCV)
"""

import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List
import math


# ──────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CameraState:
    """Camera pose and intrinsics."""
    pan_deg: float = 0.0       # azimuth of boresight
    tilt_deg: float = 0.0     # elevation of boresight
    width: int = 1280
    height: int = 720
    h_fov_deg: float = 60.0   # horizontal field of view
    v_fov_deg: float = 40.0   # vertical field of view

    @property
    def cx(self) -> float:
        """Image center x (pixels)."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Image center y (pixels)."""
        return self.height / 2.0

    @property
    def deg_per_pixel_x(self) -> float:
        """Degrees per pixel horizontally."""
        return self.h_fov_deg / self.width

    @property
    def deg_per_pixel_y(self) -> float:
        """Degrees per pixel vertically."""
        return self.v_fov_deg / self.height


@dataclass
class TargetState:
    """Target (beacon) state in angular coordinates."""
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    vel_az_deg_s: float = 0.0   # angular velocity (deg/s)
    vel_el_deg_s: float = 0.0
    acc_az_deg_s2: float = 0.0  # angular acceleration (deg/s²)
    acc_el_deg_s2: float = 0.0


@dataclass
class GroundTruth:
    """Ground truth for one frame — NEVER expose to detector or controller."""
    target_az_deg: float
    target_el_deg: float
    target_pixel_x: float
    target_pixel_y: float
    target_visible: bool
    camera_pan_deg: float
    camera_tilt_deg: float
    timestamp: float
    frame_id: int


# ──────────────────────────────────────────────────────────────────────────
# Pinhole camera projection
# ──────────────────────────────────────────────────────────────────────────

def angular_to_pixel(cam: CameraState, az_deg: float, el_deg: float) -> tuple[float, float]:
    """
    Project an angular world position into image pixel coordinates.

    The target's angular offset from the camera boresight is converted to
    pixel offsets using the camera's field of view and resolution.

    Returns (pixel_x, pixel_y). These may be outside [0, width)×[0, height)
    if the target is outside the FOV — check visibility separately.
    """
    # Angular offset from boresight
    delta_az = az_deg - cam.pan_deg
    delta_el = el_deg - cam.tilt_deg

    # Convert angular offset to pixel offset
    # Positive delta_az → target is to the right → pixel_x increases
    # Positive delta_el → target is above boresight → pixel_y decreases
    pixel_x = cam.cx + delta_az / cam.deg_per_pixel_x
    pixel_y = cam.cy - delta_el / cam.deg_per_pixel_y

    return pixel_x, pixel_y


def pixel_to_angular(cam: CameraState, px: float, py: float) -> tuple[float, float]:
    """
    Inverse projection: convert pixel coordinates to angular world position.
    Used by the controller to convert image error to angular error.
    """
    delta_az = (px - cam.cx) * cam.deg_per_pixel_x
    delta_el = (cam.cy - py) * cam.deg_per_pixel_y

    az_deg = cam.pan_deg + delta_az
    el_deg = cam.tilt_deg + delta_el

    return az_deg, el_deg


def is_visible(cam: CameraState, px: float, py: float) -> bool:
    """Check whether a pixel position is inside the image frame."""
    return 0 <= px < cam.width and 0 <= py < cam.height


# ──────────────────────────────────────────────────────────────────────────
# Target motion models
# ──────────────────────────────────────────────────────────────────────────

class MotionModel:
    """Base class — override update() to implement a trajectory."""

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        raise NotImplementedError

    def reset(self):
        pass


class ConstantVelocity(MotionModel):
    """Constant angular velocity on both axes."""

    def __init__(self, az_rate_deg_s: float = 5.0, el_rate_deg_s: float = 2.0):
        self.az_rate = az_rate_deg_s
        self.el_rate = el_rate_deg_s

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        state.azimuth_deg += self.az_rate * dt
        state.elevation_deg += self.el_rate * dt
        state.vel_az_deg_s = self.az_rate
        state.vel_el_deg_s = self.el_rate
        state.acc_az_deg_s2 = 0.0
        state.acc_el_deg_s2 = 0.0
        return state


class SinusoidalMotion(MotionModel):
    """Sinusoidal motion with configurable amplitude, frequency, and phase per axis."""

    def __init__(self,
                 az_amp_deg: float = 15.0, az_freq_hz: float = 0.1, az_phase_deg: float = 0.0,
                 el_amp_deg: float = 8.0, el_freq_hz: float = 0.15, el_phase_deg: float = 90.0):
        self.az_amp = az_amp_deg
        self.az_freq = az_freq_hz
        self.az_phase = math.radians(az_phase_deg)
        self.el_amp = el_amp_deg
        self.el_freq = el_freq_hz
        self.el_phase = math.radians(el_phase_deg)

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        omega_az = 2 * math.pi * self.az_freq
        omega_el = 2 * math.pi * self.el_freq

        state.azimuth_deg = self.az_amp * math.sin(omega_az * t + self.az_phase)
        state.elevation_deg = self.el_amp * math.sin(omega_el * t + self.el_phase)
        state.vel_az_deg_s = self.az_amp * omega_az * math.cos(omega_az * t + self.az_phase)
        state.vel_el_deg_s = self.el_amp * omega_el * math.cos(omega_el * t + self.el_phase)
        state.acc_az_deg_s2 = -self.az_amp * omega_az**2 * math.sin(omega_az * t + self.az_phase)
        state.acc_el_deg_s2 = -self.el_amp * omega_el**2 * math.sin(omega_el * t + self.el_phase)
        return state


class CircularMotion(MotionModel):
    """Circular motion at constant angular speed around a center point."""

    def __init__(self, center_az_deg: float = 10.0, center_el_deg: float = 5.0,
                 radius_deg: float = 12.0, angular_speed_deg_s: float = 8.0,
                 start_angle_deg: float = 0.0):
        self.center_az = center_az_deg
        self.center_el = center_el_deg
        self.radius = radius_deg
        self.speed = angular_speed_deg_s  # degrees per second along the circle
        self.start_angle = math.radians(start_angle_deg)
        self._angle = self.start_angle

    def reset(self):
        self._angle = self.start_angle

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        self._angle = self.start_angle + math.radians(self.speed * t)

        state.azimuth_deg = self.center_az + self.radius * math.cos(self._angle)
        state.elevation_deg = self.center_el + self.radius * math.sin(self._angle)

        omega = math.radians(self.speed)
        state.vel_az_deg_s = -self.radius * omega * math.sin(self._angle)
        state.vel_el_deg_s = self.radius * omega * math.cos(self._angle)
        state.acc_az_deg_s2 = -self.radius * omega**2 * math.cos(self._angle)
        state.acc_el_deg_s2 = -self.radius * omega**2 * math.sin(self._angle)
        return state


class RandomManeuvering(MotionModel):
    """Random maneuvering with bounded acceleration, changed at random intervals."""

    def __init__(self, max_acc_deg_s2: float = 10.0, change_interval_s: float = 2.0,
                 max_vel_deg_s: float = 20.0, seed: int = 42):
        self.max_acc = max_acc_deg_s2
        self.change_interval = change_interval_s
        self.max_vel = max_vel_deg_s
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._acc_az = 0.0
        self._acc_el = 0.0
        self._next_change = 0.0

    def reset(self):
        self._acc_az = 0.0
        self._acc_el = 0.0
        self._next_change = 0.0
        self.rng = np.random.default_rng(self._seed)  # deterministic reset

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        # Randomly change acceleration at intervals
        if t >= self._next_change:
            self._acc_az = self.rng.uniform(-self.max_acc, self.max_acc)
            self._acc_el = self.rng.uniform(-self.max_acc, self.max_acc)
            self._next_change = t + self.rng.uniform(0.5 * self.change_interval,
                                                      1.5 * self.change_interval)

        # Integrate acceleration
        state.vel_az_deg_s += self._acc_az * dt
        state.vel_el_deg_s += self._acc_el * dt

        # Clamp velocity
        state.vel_az_deg_s = np.clip(state.vel_az_deg_s, -self.max_vel, self.max_vel)
        state.vel_el_deg_s = np.clip(state.vel_el_deg_s, -self.max_vel, self.max_vel)

        # Integrate position
        state.azimuth_deg += state.vel_az_deg_s * dt
        state.elevation_deg += state.vel_el_deg_s * dt

        state.acc_az_deg_s2 = self._acc_az
        state.acc_el_deg_s2 = self._acc_el
        return state


class ScriptedTrajectory(MotionModel):
    """Replays a trajectory from a list of (az, el) positions."""

    def __init__(self, positions: list, fps: float = 30.0):
        """
        Args:
            positions: List of (azimuth_deg, elevation_deg) tuples, one per frame.
            fps: Frames per second (for time indexing).
        """
        self.positions = positions
        self.fps = fps
        self._index = 0

    def update(self, state: TargetState, dt: float, t: float) -> TargetState:
        """Set (az, el) at time t by index lookup."""
        idx = min(int(t * self.fps), len(self.positions) - 1)
        idx = max(0, idx)
        az, el = self.positions[idx]
        state.azimuth_deg = az
        state.elevation_deg = el
        state.vel_az_deg_s = 0.0
        state.vel_el_deg_s = 0.0
        state.acc_az_deg_s2 = 0.0
        state.acc_el_deg_s2 = 0.0
        return state

    def reset(self):
        self._index = 0


# ──────────────────────────────────────────────────────────────────────────
# Disturbance models — each has an apply() method
# ──────────────────────────────────────────────────────────────────────────

class PlatformVibration:
    """
    Platform vibration: sum of sinusoidal signals plus random jitter.
    Adds a rapid angular offset to the camera boresight, simulating
    mechanical vibration of the platform.
    """

    def __init__(self, rms_deg: float = 0.15, freqs_hz: list = None, seed: int = 42):
        self.rms = rms_deg
        # Default: a few vibration modes at typical mechanical frequencies
        self.freqs = freqs_hz or [5.0, 12.0, 25.0, 47.0]
        self.phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(self.freqs))
        self.amplitudes = [rms_deg * 0.5 / np.sqrt(len(self.freqs))] * len(self.freqs)
        self.jitter_rng = np.random.default_rng(seed + 1)

    def apply(self, cam: CameraState, t: float) -> CameraState:
        """Return a NEW camera state with vibration offset applied (never mutates input)."""
        offset_az = sum(a * np.sin(2 * np.pi * f * t + p)
                        for a, f, p in zip(self.amplitudes, self.freqs, self.phases))
        offset_el = sum(a * np.cos(2 * np.pi * f * t + p)
                        for a, f, p in zip(self.amplitudes, self.freqs, self.phases))

        # Add Gaussian jitter
        offset_az += self.jitter_rng.normal(0, self.rms * 0.2)
        offset_el += self.jitter_rng.normal(0, self.rms * 0.2)

        return CameraState(
            pan_deg=cam.pan_deg + offset_az,
            tilt_deg=cam.tilt_deg + offset_el,
            width=cam.width, height=cam.height,
            h_fov_deg=cam.h_fov_deg, v_fov_deg=cam.v_fov_deg,
        )


class SensorNoise:
    """
    Sensor noise: additive Gaussian noise on pixel values,
    optionally combined with Poisson noise.
    Models sensor readout noise and thermal noise.
    """

    def __init__(self, sigma: float = 8.0, seed: int = 42, use_poisson: bool = False):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.use_poisson = use_poisson

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (and optionally Poisson noise) to the frame."""
        # Use seeded NumPy RNG for reproducibility (NF-RE2/NF-RE3).
        # cv2.randn uses OpenCV's internal RNG which cannot be seeded externally.
        h, w = frame.shape[:2]
        noise2d = self.rng.normal(0.0, self.sigma, size=(h, w)).astype(np.float32)
        if len(frame.shape) == 3:
            noise = noise2d[:, :, np.newaxis]
        else:
            noise = noise2d
        noisy = frame.astype(np.float32) + noise
        frame = np.clip(noisy, 0, 255).astype(np.uint8)

        if self.use_poisson:
            # Poisson noise: converts intensity to rate parameter
            poisson_noise = self.rng.poisson(frame.astype(np.float32)).astype(np.uint8)
            frame = cv2.addWeighted(frame, 0.7, poisson_noise, 0.3, 0)

        return frame


class MotionBlur:
    """
    Motion blur: directional kernel based on relative angular velocity.
    Elongates and weakens the beacon when the camera or target moves fast.
    """

    def __init__(self, max_blur_pixels: float = 5.0):
        self.max_blur = max_blur_pixels

    def apply(self, frame: np.ndarray, blur_pixels: float = 0.0,
              angle_deg: float = 0.0) -> np.ndarray:
        """Apply motion blur in the given direction."""
        blur_pixels = min(blur_pixels, self.max_blur)
        if blur_pixels < 1.0:
            return frame

        # Create a directional kernel
        kernel_size = int(np.ceil(blur_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        angle_rad = np.radians(angle_deg)

        # Draw a line in the kernel
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel /= kernel.sum()
        return cv2.filter2D(frame, -1, kernel)


class AtmosphericTurbulence:
    """
    Atmospheric turbulence: low-frequency random angular displacement using
    an Ornstein-Uhlenbeck process. Models slowly varying pointing errors on
    the camera boresight caused by refractive index fluctuations in the
    atmosphere.
    """

    def __init__(self, rms_deg: float = 0.1, correlation_time_s: float = 1.0, seed: int = 42):
        self.rms = rms_deg
        self.tau = correlation_time_s
        self.rng = np.random.default_rng(seed)
        # OU process: theta = drift * theta_prev + noise
        # drift = exp(-dt / tau), noise std = rms * sqrt(1 - exp(-2*dt/tau))
        # We pre-compute nothing here since dt varies; state is kept in _offset.
        self._offset_az = 0.0
        self._offset_el = 0.0

    def apply(self, cam: CameraState, t: float) -> CameraState:
        """
        Return a NEW camera state with atmospheric turbulence offset applied
        (never mutates the input camera). Uses an Ornstein-Uhlenbeck filter:
        each call advances the internal filtered angular offset by one timestep,
        producing smooth, slowly varying pointing errors with the configured RMS.
        """
        # Step size: use a fixed internal dt derived from the correlation time
        # to keep the process stable regardless of frame rate.
        dt = self.tau / 10.0
        decay = np.exp(-dt / self.tau)
        noise_std = self.rms * np.sqrt(1.0 - np.exp(-2.0 * dt / self.tau))

        self._offset_az = decay * self._offset_az + self.rng.normal(0.0, noise_std)
        self._offset_el = decay * self._offset_el + self.rng.normal(0.0, noise_std)

        return CameraState(
            pan_deg=cam.pan_deg + self._offset_az,
            tilt_deg=cam.tilt_deg + self._offset_el,
            width=cam.width, height=cam.height,
            h_fov_deg=cam.h_fov_deg, v_fov_deg=cam.v_fov_deg,
        )


class ExposureVariation:
    """
    Exposure variation: brightness and gain modulation via a sinusoidal signal.
    Models slow variations in beacon contrast due to scintillation and
    detector gain instability.
    """

    def __init__(self, rate_hz: float = 0.1, amplitude: float = 0.3, seed: int = 42):
        self.rate = rate_hz
        self.amplitude = amplitude
        self.rng = np.random.default_rng(seed)
        # Random phase offset for reproducibility
        self._phase = self.rng.uniform(0, 2 * np.pi)

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        """
        Return a modified frame with sinusoidal brightness modulation applied.

        Factor = 1.0 + amplitude * sin(2*pi*rate*t + phase), then
        frame = clip(frame * factor, 0, 255).
        """
        factor = 1.0 + self.amplitude * np.sin(2 * np.pi * self.rate * t + self._phase)
        modulated = frame.astype(np.float32) * factor
        return np.clip(modulated, 0, 255).astype(np.uint8)


class Occlusion:
    """
    Occlusion: random or scripted masked regions that temporarily block
    parts of the frame. Models debris, birds, or other transient occluders
    that force temporary track loss.
    """

    def __init__(self, duration_s: float = 1.0, frequency_hz: float = 0.05,
                 region_size_frac: float = 0.15, seed: int = 42):
        self.duration = duration_s
        self.frequency = frequency_hz
        self.region_size_frac = region_size_frac
        self.rng = np.random.default_rng(seed)

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        """
        Return a frame with a black rectangle mask drawn during occlusion periods.

        Occlusion on/off timing is determined deterministically by a sine wave
        threshold: when sin(2*pi*frequency*t) > 0 and the fractional phase is
        within [0, duration*frequency), an occlusion is active.
        """
        phase = (self.frequency * t) % 1.0
        is_occluded = (np.sin(2 * np.pi * self.frequency * t) > 0.0) and (phase < self.frequency * self.duration)

        if not is_occluded:
            return frame

        h, w = frame.shape[:2]
        rw = max(1, int(w * self.region_size_frac))
        rh = max(1, int(h * self.region_size_frac))

        # Deterministic position from seeded RNG
        x = self.rng.integers(0, max(1, w - rw))
        y = self.rng.integers(0, max(1, h - rh))

        occluded = frame.copy()
        occluded[y:y + rh, x:x + rw] = 0
        return occluded


class FalseBeacons:
    """
    False beacons: distractor points with similar appearance to the real
    beacon. Tests identification robustness of the detector and tracker
    against spurious targets.
    """

    def __init__(self, count: int = 2, brightness_range: tuple = (80, 180),
                 sigma_range: tuple = (2.0, 5.0), seed: int = 42):
        self.count = count
        self.brightness_range = brightness_range
        self.sigma_range = sigma_range
        self.rng = np.random.default_rng(seed)
        # Pre-compute fixed positions and base parameters from the seed
        self._positions = None  # lazily initialized on first apply
        self._sigmas = None
        self._base_brightness = None

    def _init_positions(self, h: int, w: int):
        """Generate fixed random positions, sigmas, and base brightness values."""
        self._positions = []
        self._sigmas = self.rng.uniform(self.sigma_range[0], self.sigma_range[1], self.count)
        self._base_brightness = self.rng.uniform(
            self.brightness_range[0], self.brightness_range[1], self.count
        )
        margin = 20
        for _ in range(self.count):
            x = self.rng.uniform(margin, w - margin)
            y = self.rng.uniform(margin, h - margin)
            self._positions.append((x, y))

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        """
        Return a frame with additional Gaussian blobs drawn at fixed random
        positions. Blob brightness varies slightly per frame to simulate
        flickering distractor beacons.
        """
        h, w = frame.shape[:2]
        if self._positions is None:
            self._init_positions(h, w)

        result = frame.astype(np.float32)

        for i, (px, py) in enumerate(self._positions):
            sigma = self._sigmas[i]
            # Slight per-frame brightness variation
            flicker = 1.0 + 0.1 * np.sin(2 * np.pi * 0.3 * t + i * 1.5)
            brightness = self._base_brightness[i] * flicker

            # Region of influence: +/-3 sigma
            r = int(np.ceil(3 * sigma))
            x0 = max(0, int(px - r))
            x1 = min(w, int(px + r) + 1)
            y0 = max(0, int(py - r))
            y1 = min(h, int(py + r) + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            xs = np.arange(x0, x1) - px
            ys = np.arange(y0, y1) - py
            xx, yy = np.meshgrid(xs, ys)
            gaussian = brightness * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

            result[y0:y1, x0:x1] += gaussian[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────────

class FrameRenderer:
    """
    Renders synthetic camera frames from scene state.
    Produces a brightness-gradient background and a Gaussian beacon blob.
    """

    def __init__(self, cam: CameraState,
                 beacon_sigma_px: float = 4.0,
                 beacon_peak_intensity: float = 255.0,
                 background_top: int = 40,
                 background_bottom: int = 10,
                 stars_enabled: bool = False,
                 star_count: int = 50):
        self.cam = cam
        self.beacon_sigma = beacon_sigma_px
        self.beacon_peak = beacon_peak_intensity
        self.bg_top = background_top      # brightness at top of frame
        self.bg_bottom = background_bottom  # brightness at bottom of frame
        self.stars_enabled = stars_enabled

        # Pre-compute star positions so they are consistent across frames
        if stars_enabled:
            star_rng = np.random.RandomState(seed=42)
            self._star_positions = [
                (star_rng.uniform(0, cam.width), star_rng.uniform(0, cam.height))
                for _ in range(star_count)
            ]
        else:
            self._star_positions = []

    def render_background(self) -> np.ndarray:
        """Create a vertical brightness gradient (sky-like)."""
        # Linear gradient from top to bottom
        gradient = np.linspace(self.bg_top, self.bg_bottom, self.cam.height,
                                dtype=np.float32)
        bg = np.tile(gradient[:, np.newaxis], (1, self.cam.width))
        # Convert to 3-channel
        bg = np.stack([bg, bg, bg], axis=-1).astype(np.uint8)
        return bg

    def render_beacon(self, frame: np.ndarray, px: float, py: float) -> np.ndarray:
        """
        Draw a 2-D Gaussian blob at (px, py) on the frame.
        The blob is white (all channels equal) to simulate an optical beacon.
        """
        sigma = self.beacon_sigma
        # Region of influence: ±3 sigma
        r = int(np.ceil(3 * sigma))
        x0 = int(px - r)
        x1 = int(px + r) + 1
        y0 = int(py - r)
        y1 = int(py + r) + 1

        # Clip to frame bounds
        x0_c = max(0, x0)
        x1_c = min(self.cam.width, x1)
        y0_c = max(0, y0)
        y1_c = min(self.cam.height, y1)
        if x1_c <= x0_c or y1_c <= y0_c:
            return frame  # Beacon fully outside frame

        # Compute Gaussian on the sub-region
        xs = np.arange(x0_c, x1_c) - px
        ys = np.arange(y0_c, y1_c) - py
        xx, yy = np.meshgrid(xs, ys)
        gaussian = self.beacon_peak * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Additive blending
        sub = frame[y0_c:y1_c, x0_c:x1_c].astype(np.float32)
        sub += gaussian[:, :, np.newaxis]
        frame[y0_c:y1_c, x0_c:x1_c] = np.clip(sub, 0, 255).astype(np.uint8)
        return frame

    def render(self, target: TargetState, cam_with_disturbances: CameraState) -> np.ndarray:
        """
        Render a full frame: background + beacon.
        cam_with_disturbances is the camera state AFTER disturbances are applied.
        """
        frame = self.render_background()

        # Draw background stars (behind the beacon)
        if self.stars_enabled:
            for sx, sy in self._star_positions:
                if 0 <= sx < self.cam.width and 0 <= sy < self.cam.height:
                    cv2.circle(frame, (int(sx), int(sy)), 1, (200, 200, 200), -1)

        # Project target into pixel coordinates using the disturbed camera
        px, py = angular_to_pixel(cam_with_disturbances,
                                  target.azimuth_deg, target.elevation_deg)

        if is_visible(cam_with_disturbances, px, py):
            frame = self.render_beacon(frame, px, py)

        return frame


# ──────────────────────────────────────────────────────────────────────────
# Simulator — ties it all together
# ──────────────────────────────────────────────────────────────────────────

class Simulator:
    """
    The main simulator. Call step() every frame to get (frame, ground_truth).

    Usage:
        sim = Simulator(config...)
        for frame_id in range(num_frames):
            frame, gt = sim.step(frame_id)
            # Pass frame to detector; pass gt to logger only
    """

    def __init__(self,
                 cam: CameraState = CameraState(),
                 motion: MotionModel = None,
                 beacon_sigma_px: float = 4.0,
                 beacon_peak: float = 255.0,
                 vibration: Optional[PlatformVibration] = None,
                 turbulence: Optional['AtmosphericTurbulence'] = None,
                 sensor_noise: Optional[SensorNoise] = None,
                 motion_blur: Optional[MotionBlur] = None,
                 initial_target: TargetState = None,
                 fps: float = 30.0):
        self.cam = cam
        self.motion = motion or CircularMotion()
        self.target = initial_target or TargetState()
        self.renderer = FrameRenderer(cam, beacon_sigma_px, beacon_peak)
        self.vibration = vibration
        self.turbulence = turbulence
        self.sensor_noise = sensor_noise
        self.motion_blur = motion_blur
        self.fps = fps
        self.dt = 1.0 / fps
        self.t = 0.0
        self.frame_id = 0

    def step(self, frame_id: int) -> tuple[np.ndarray, GroundTruth]:
        """
        Advance the simulation by one frame.

        Returns (frame, ground_truth). The frame is the synthetic image the
        detector sees. The ground_truth contains the true target position —
        it must NEVER be passed to the detector or controller.
        """
        # 1. Update target state
        self.motion.update(self.target, self.dt, self.t)

        # 2. Copy camera state (we apply disturbances to a copy, not the original)
        cam_disturbed = CameraState(
            pan_deg=self.cam.pan_deg,
            tilt_deg=self.cam.tilt_deg,
            width=self.cam.width,
            height=self.cam.height,
            h_fov_deg=self.cam.h_fov_deg,
            v_fov_deg=self.cam.v_fov_deg,
        )

        # 3. Apply camera-level disturbances to copy (never mutate original)
        if self.vibration:
            cam_disturbed = self.vibration.apply(cam_disturbed, self.t)
        if self.turbulence:
            cam_disturbed = self.turbulence.apply(cam_disturbed, self.t)

        # 4. Render the frame
        frame = self.renderer.render(self.target, cam_disturbed)

        # 5. Apply sensor-level disturbances
        if self.motion_blur:
            # Compute relative angular velocity for blur direction/magnitude
            rel_vel_az = self.target.vel_az_deg_s
            rel_vel_el = self.target.vel_el_deg_s
            blur_magnitude = np.sqrt(rel_vel_az**2 + rel_vel_el**2)
            # Scale: convert deg/s to pixels of blur
            blur_px = blur_magnitude / self.cam.deg_per_pixel_x * self.dt
            blur_angle = np.degrees(np.arctan2(rel_vel_el, rel_vel_az))
            frame = self.motion_blur.apply(frame, blur_px, blur_angle)

        if self.sensor_noise:
            frame = self.sensor_noise.apply(frame)

        # 6. Compute ground truth using the DISTURBED camera (what the detector
        #    actually sees, so the error measurement reflects real performance)
        gt_px, gt_py = angular_to_pixel(cam_disturbed,
                                        self.target.azimuth_deg,
                                        self.target.elevation_deg)
        gt_visible = is_visible(cam_disturbed, gt_px, gt_py)

        gt = GroundTruth(
            target_az_deg=self.target.azimuth_deg,
            target_el_deg=self.target.elevation_deg,
            target_pixel_x=gt_px,
            target_pixel_y=gt_py,
            target_visible=gt_visible,
            camera_pan_deg=cam_disturbed.pan_deg,
            camera_tilt_deg=cam_disturbed.tilt_deg,
            timestamp=self.t,
            frame_id=frame_id,
        )

        # 7. Advance time
        self.t += self.dt
        self.frame_id = frame_id

        return frame, gt

    def reset(self):
        """Reset to initial state."""
        self.t = 0.0
        self.frame_id = 0
        self.target = TargetState()
        self.motion.reset()

    def set_camera_pose(self, pan_deg: float, tilt_deg: float):
        """Set the camera pan/tilt (called by the controller each frame)."""
        self.cam.pan_deg = pan_deg
        self.cam.tilt_deg = tilt_deg


# ──────────────────────────────────────────────────────────────────────────
# Multi-target support
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BeaconConfig:
    """Configuration for a single beacon in a multi-target scenario."""
    id: str = "beacon_01"
    color: tuple = (255, 255, 255)  # BGR
    intensity: float = 1.0
    sigma_px: float = 4.0
    motion_model: MotionModel = None


@dataclass
class MultiGroundTruth:
    """Ground truth for all beacons in a multi-target frame."""
    beacons: dict  # {beacon_id: GroundTruth}
    camera_pan_deg: float
    camera_tilt_deg: float
    timestamp: float
    frame_id: int


class MultiTargetSimulator:
    """
    Multi-target simulator that manages multiple beacons simultaneously.

    Each beacon has its own identity, color, intensity, sigma, and motion model.
    step() renders all visible beacons into a single composite frame and returns
    a list of GroundTruth objects (one per beacon).

    Usage:
        beacons = [
            BeaconConfig(id="b1", color=(255,255,255), motion_model=CircularMotion(...)),
            BeaconConfig(id="b2", color=(0,255,255), motion_model=SinusoidalMotion(...)),
        ]
        multi_sim = MultiTargetSimulator(cam=cam, beacons=beacons)
        for frame_id in range(num_frames):
            frame, gt_list = multi_sim.step(frame_id)
    """

    def __init__(self,
                 cam: CameraState = CameraState(),
                 beacons: List[BeaconConfig] = None,
                 vibration: Optional[PlatformVibration] = None,
                 turbulence: Optional[AtmosphericTurbulence] = None,
                 sensor_noise: Optional[SensorNoise] = None,
                 motion_blur: Optional[MotionBlur] = None,
                 fps: float = 30.0):
        self.cam = cam
        self.fps = fps
        self.dt = 1.0 / fps
        self.t = 0.0
        self.frame_id = 0
        self.vibration = vibration
        self.turbulence = turbulence
        self.sensor_noise = sensor_noise
        self.motion_blur = motion_blur

        if beacons is None:
            beacons = [BeaconConfig()]

        # Internal state for each beacon
        self._beacon_configs: List[BeaconConfig] = []
        self._targets: dict = {}  # {beacon_id: TargetState}
        self._motions: dict = {}  # {beacon_id: MotionModel}
        self._renderers: dict = {}  # {beacon_id: FrameRenderer}

        for bc in beacons:
            self._beacon_configs.append(bc)
            self._targets[bc.id] = TargetState()
            self._motions[bc.id] = bc.motion_model or CircularMotion()
            self._renderers[bc.id] = FrameRenderer(
                cam,
                beacon_sigma_px=bc.sigma_px,
                beacon_peak_intensity=255.0 * bc.intensity,
            )

    def step(self, frame_id: int) -> tuple[np.ndarray, List[GroundTruth]]:
        """
        Advance the simulation by one frame.

        Returns:
            (frame, ground_truths): frame is the composite image with all
            visible beacons rendered. ground_truths is a list of GroundTruth
            objects, one per beacon (regardless of visibility).
        """
        # 1. Update all target states
        for bc in self._beacon_configs:
            self._motions[bc.id].update(self._targets[bc.id], self.dt, self.t)

        # 2. Copy camera state for disturbance application
        cam_disturbed = CameraState(
            pan_deg=self.cam.pan_deg,
            tilt_deg=self.cam.tilt_deg,
            width=self.cam.width,
            height=self.cam.height,
            h_fov_deg=self.cam.h_fov_deg,
            v_fov_deg=self.cam.v_fov_deg,
        )

        # 3. Apply camera disturbances to copy (never mutate original)
        if self.vibration:
            cam_disturbed = self.vibration.apply(cam_disturbed, self.t)
        if self.turbulence:
            cam_disturbed = self.turbulence.apply(cam_disturbed, self.t)

        # 4. Render the frame with all visible beacons
        frame = self._renderers[self._beacon_configs[0].id].render_background()
        for bc in self._beacon_configs:
            target = self._targets[bc.id]
            px, py = angular_to_pixel(cam_disturbed,
                                      target.azimuth_deg, target.elevation_deg)
            if is_visible(cam_disturbed, px, py):
                # Draw this beacon with its own color
                frame = self._render_colored_beacon(
                    frame, bc, px, py)

        # 5. Apply sensor-level disturbances
        if self.motion_blur:
            # Compute aggregate blur from the first beacon's velocity
            # (simplification for multi-target)
            first_target = self._targets[self._beacon_configs[0].id]
            rel_vel_az = first_target.vel_az_deg_s
            rel_vel_el = first_target.vel_el_deg_s
            blur_magnitude = np.sqrt(rel_vel_az**2 + rel_vel_el**2)
            blur_px = blur_magnitude / self.cam.deg_per_pixel_x * self.dt
            blur_angle = np.degrees(np.arctan2(rel_vel_el, rel_vel_az))
            frame = self.motion_blur.apply(frame, blur_px, blur_angle)

        if self.sensor_noise:
            frame = self.sensor_noise.apply(frame)

        # 6. Compute ground truth for each beacon
        ground_truths = []
        for bc in self._beacon_configs:
            target = self._targets[bc.id]
            gt_px, gt_py = angular_to_pixel(cam_disturbed,
                                            target.azimuth_deg, target.elevation_deg)
            gt_visible = is_visible(cam_disturbed, gt_px, gt_py)

            ground_truths.append(GroundTruth(
                target_az_deg=target.azimuth_deg,
                target_el_deg=target.elevation_deg,
                target_pixel_x=gt_px,
                target_pixel_y=gt_py,
                target_visible=gt_visible,
                camera_pan_deg=cam_disturbed.pan_deg,
                camera_tilt_deg=cam_disturbed.tilt_deg,
                timestamp=self.t,
                frame_id=frame_id,
            ))

        # 7. Advance time
        self.t += self.dt
        self.frame_id = frame_id

        return frame, ground_truths

    def _render_colored_beacon(self, frame: np.ndarray, bc: BeaconConfig,
                               px: float, py: float) -> np.ndarray:
        """
        Draw a 2-D Gaussian blob with the beacon's BGR color at (px, py).
        Uses the beacon's sigma and intensity.
        """
        sigma = bc.sigma_px
        intensity_peak = 255.0 * bc.intensity
        color = bc.color  # BGR tuple

        r = int(np.ceil(3 * sigma))
        x0_c = max(0, int(px - r))
        x1_c = min(self.cam.width, int(px + r) + 1)
        y0_c = max(0, int(py - r))
        y1_c = min(self.cam.height, int(py + r) + 1)
        if x1_c <= x0_c or y1_c <= y0_c:
            return frame

        xs = np.arange(x0_c, x1_c) - px
        ys = np.arange(y0_c, y1_c) - py
        xx, yy = np.meshgrid(xs, ys)
        gaussian = intensity_peak * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Apply per-channel color scaling
        sub = frame[y0_c:y1_c, x0_c:x1_c].astype(np.float32)
        for c in range(3):
            sub[:, :, c] += gaussian * (color[c] / 255.0)
        frame[y0_c:y1_c, x0_c:x1_c] = np.clip(sub, 0, 255).astype(np.uint8)
        return frame

    def reset(self):
        """Reset all beacons and time to initial state."""
        self.t = 0.0
        self.frame_id = 0
        for bc in self._beacon_configs:
            self._targets[bc.id] = TargetState()
            self._motions[bc.id].reset()

    def set_camera_pose(self, pan_deg: float, tilt_deg: float):
        """Set the camera pan/tilt (called by the controller each frame)."""
        self.cam.pan_deg = pan_deg
        self.cam.tilt_deg = tilt_deg

    @property
    def beacon_ids(self) -> List[str]:
        """Return the list of beacon IDs."""
        return [bc.id for bc in self._beacon_configs]

    def get_target_state(self, beacon_id: str) -> TargetState:
        """Return the current TargetState for a specific beacon."""
        return self._targets[beacon_id]


# ──────────────────────────────────────────────────────────────────────────
# Self-test: verify projection math
# ──────────────────────────────────────────────────────────────────────────

def _test_projection():
    """
    Verify that angular_to_pixel and pixel_to_angular are consistent inverses,
    and that a target at the boresight maps to the image center.
    """
    cam = CameraState(pan_deg=10.0, tilt_deg=5.0)

    # Test 1: target at boresight → image center
    px, py = angular_to_pixel(cam, 10.0, 5.0)
    assert abs(px - cam.cx) < 0.01, f"Expected cx={cam.cx}, got {px}"
    assert abs(py - cam.cy) < 0.01, f"Expected cy={cam.cy}, got {py}"
    print(f"  Test 1 PASS: boresight -> center ({px:.1f}, {py:.1f})")

    # Test 2: target at right edge of FOV → at width pixel
    # The H-FOV spans the full image width, so the right edge is at pan + h_fov/2
    px, py = angular_to_pixel(cam, 10.0 + cam.h_fov_deg / 2.0, 5.0)
    assert abs(px - cam.width) < 0.01, f"Expected width={cam.width}, got {px}"
    print(f"  Test 2 PASS: +half H-FOV -> right edge ({px:.1f}, {py:.1f})")

    # Test 3: target above boresight → y < cy
    px, py = angular_to_pixel(cam, 10.0, 5.0 + 5.0)
    assert py < cam.cy, f"Expected py < cy={cam.cy}, got {py}"
    print(f"  Test 3 PASS: above boresight -> y < cy ({px:.1f}, {py:.1f})")

    # Test 4: round-trip (angular → pixel → angular)
    az, el = 25.3, -7.2
    px, py = angular_to_pixel(cam, az, el)
    az2, el2 = pixel_to_angular(cam, px, py)
    assert abs(az - az2) < 1e-6, f"Azimuth round-trip failed: {az} vs {az2}"
    assert abs(el - el2) < 1e-6, f"Elevation round-trip failed: {el} vs {el2}"
    print(f"  Test 4 PASS: round-trip az={az:.3f}->{az2:.3f}, el={el:.3f}->{el2:.3f}")

    print("All projection tests passed.\n")


def _test_motion_models():
    """Verify that each motion model produces valid, changing positions."""
    print("Testing motion models...")

    models = [
        ("ConstantVelocity", ConstantVelocity(az_rate_deg_s=5.0, el_rate_deg_s=2.0)),
        ("Sinusoidal", SinusoidalMotion()),
        ("Circular", CircularMotion()),
        ("RandomManeuvering", RandomManeuvering()),
    ]

    for name, model in models:
        state = TargetState()
        positions = []
        dt = 1/30
        t = 0.0
        for _ in range(30):
            model.update(state, dt, t)
            positions.append((state.azimuth_deg, state.elevation_deg))
            t += dt

        # Check positions changed (not stuck)
        az_range = max(p[0] for p in positions) - min(p[0] for p in positions)
        assert az_range > 0.01, f"{name}: azimuth didn't change"
        print(f"  {name}: az range={az_range:.2f}°, final pos=({state.azimuth_deg:.2f}, {state.elevation_deg:.2f})")

    print("All motion model tests passed.\n")


def _test_rendering():
    """Render a few frames and save them for visual inspection."""
    print("Testing rendering...")

    cam = CameraState(pan_deg=10.0, tilt_deg=5.0)
    motion = CircularMotion(center_az_deg=10.0, center_el_deg=5.0,
                            radius_deg=8.0, angular_speed_deg_s=6.0)
    sim = Simulator(cam=cam, motion=motion, beacon_sigma_px=4.0)

    frames = []
    for i in range(90):  # 3 seconds at 30 FPS
        frame, gt = sim.step(i)
        frames.append(frame)
        if i % 30 == 0:
            print(f"  Frame {i}: target at ({gt.target_pixel_x:.1f}, {gt.target_pixel_y:.1f}), visible={gt.target_visible}")

    # Save a video for visual inspection
    import tempfile
    video_path = os.path.join(tempfile.gettempdir(), 'sim_test.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (cam.width, cam.height))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  Saved 90-frame test video to {video_path}")
    print("Rendering test passed.\n")


def _test_with_disturbances():
    """Test that disturbances don't crash and visibly affect the frame."""
    print("Testing disturbances...")

    cam = CameraState(pan_deg=0.0, tilt_deg=0.0)
    motion = CircularMotion(radius_deg=10.0, angular_speed_deg_s=5.0)
    vibration = PlatformVibration(rms_deg=0.3)
    noise = SensorNoise(sigma=12.0)
    blur = MotionBlur(max_blur_pixels=4.0)

    sim = Simulator(cam=cam, motion=motion, vibration=vibration,
                    sensor_noise=noise, motion_blur=blur)

    for i in range(30):
        frame, gt = sim.step(i)

    # Check noise was added (frame should not be a clean gradient)
    assert frame.std() > 5.0, "Frame looks too clean — noise not applied"
    print(f"  Frame std after noise+blur: {frame.std():.2f}")
    print("Disturbance test passed.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("sim.py — Self-tests")
    print("=" * 60)
    print()

    _test_projection()
    _test_motion_models()
    _test_rendering()
    _test_with_disturbances()

    print("=" * 60)
    print("All self-tests passed. sim.py is ready to use.")
    print("=" * 60)
