"""
config.py — All tunable parameters for the FSOC coarse-alignment simulator.

Edit values here to change behavior across all modules. No need to touch
sim.py, detect.py, control.py, or main.py.

Usage in other files:
    from config import Config
    config = Config()
"""

from dataclasses import dataclass


@dataclass
class Config:
    """All tunable parameters in one place."""

    # ── Simulation ──
    fps: int = 30
    duration_s: int = 60
    random_seed: int = 42

    # ── Camera ──
    width: int = 1280
    height: int = 720
    h_fov_deg: float = 60.0
    v_fov_deg: float = 40.0
    pan_range: tuple = (-180.0, 180.0)
    tilt_range: tuple = (-30.0, 90.0)
    rate_limit_deg_s: float = 60.0

    # ── Beacon ──
    beacon_sigma_px: float = 4.0
    beacon_peak: float = 255.0

    # ── Detector ──
    threshold: int = 80           # fixed brightness threshold
    min_area: int = 2             # min blob size (pixels)
    max_area: int = 500           # max blob size (pixels)

    # ── Kalman filter ──
    measurement_noise: float = 8.0  # higher = trusts detections less

    # ── PID Controller ──
    kp: float = 1.0               # proportional gain (start here)
    ki: float = 0.0               # integral gain (add only if steady-state offset)
    kd: float = 0.15              # derivative gain (damping)
    deadband_pixels: float = 3.0  # ignore errors below this

    # ── Tracker FSM ──
    verify_threshold: int = 2               # consecutive detections for CANDIDATE_VERIFICATION
    acquire_error_threshold: float = 100.0  # pixel error threshold to enter TRACKING
    acquire_threshold: int = 3              # consecutive detections to enter TRACKING
    lose_threshold: int = 5                 # consecutive misses to enter REACQUIRING
    reacquire_timeout: int = 150            # frames before returning to SEARCHING (~5s)

    # ── Clean scenario ──
    # Circular trajectory, no disturbances. Shows ideal tracking.
    clean_motion: str = "circular"
    clean_center_az: float = 5.0
    clean_center_el: float = 3.0
    clean_radius: float = 8.0
    clean_speed: float = 4.0       # deg/s along the circle
    clean_vibration_rms: float = 0.0
    clean_noise_sigma: float = 0.0
    clean_blur_pixels: float = 0.0

    # ── Hard scenario ──
    # Sinusoidal trajectory with vibration, noise, and motion blur.
    # Shows the system working under realistic disturbances.
    hard_motion: str = "sinusoidal"
    hard_az_amp: float = 10.0      # azimuth amplitude (deg)
    hard_az_freq: float = 0.2      # azimuth frequency (Hz)
    hard_el_amp: float = 6.0       # elevation amplitude (deg)
    hard_el_freq: float = 0.3      # elevation frequency (Hz)
    hard_vibration_rms: float = 0.2
    hard_noise_sigma: float = 12.0
    hard_blur_pixels: float = 3.0

    # ── v1.1 disturbances ──
    turbulence_rms_deg: float = 0.1
    turbulence_correlation_s: float = 1.0
    exposure_rate_hz: float = 0.1
    exposure_amplitude: float = 0.3
    occlusion_duration_s: float = 1.0
    occlusion_frequency_hz: float = 0.05
    false_beacon_count: int = 2
    false_beacon_brightness_range: tuple = (80, 180)

    # ── Multi-target ──
    multi_target: bool = False
    num_targets: int = 2
    target_colors: list = None  # default to [(255,255,255), (0,255,255)]

    def __post_init__(self):
        if self.target_colors is None:
            self.target_colors = [(255, 255, 255), (0, 255, 255)]
