"""
main.py — Main loop, display, logging, and plots for the
FSOC coarse-alignment simulator.

This is the entry point. Run:  python main.py

Controls:
  SPACE  — start / pause
  R      — reset
  1      — scenario: clean (circular, no disturbances)
  2      — scenario: hard (maneuvering, vibration, noise)
  G      — toggle ground-truth overlay (dev mode)
  Q/ESC  — quit

Dependencies: numpy, cv2, matplotlib
"""

import sys
import argparse
import numpy as np
import cv2
import csv
import time
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from sim import (Simulator, CameraState, TargetState,
                 ConstantVelocity, SinusoidalMotion, CircularMotion,
                 RandomManeuvering,
                 PlatformVibration, SensorNoise, MotionBlur,
                 GroundTruth,
                 AtmosphericTurbulence, ExposureVariation, Occlusion, FalseBeacons)
from detect import (BeaconDetector, KalmanFilter2D, AlphaBetaFilter,
                    Tracker, TrackerState, MultiTracker)
from control import IntegratedController, ControllerOutput, SearchPattern
from report import SummaryReporter, ReportGenerator

# Conditional imports for optional features
try:
    from ai_detector import AIDetector
except ImportError:
    AIDetector = None

try:
    from sim import MultiTargetSimulator, BeaconConfig
except ImportError:
    MultiTargetSimulator = None
    BeaconConfig = None


# ──────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────────

def _build_motion_model(config: Config, scenario_name: str):
    """Build a motion model from config based on scenario type."""
    seed = config.random_seed

    if scenario_name == "clean":
        motion_type = config.clean_motion
    elif scenario_name == "hard":
        motion_type = config.hard_motion
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    if motion_type == "circular":
        return CircularMotion(
            center_az_deg=config.clean_center_az,
            center_el_deg=config.clean_center_el,
            radius_deg=config.clean_radius,
            angular_speed_deg_s=config.clean_speed,
        )
    elif motion_type == "sinusoidal":
        return SinusoidalMotion(
            az_amp_deg=config.hard_az_amp,
            az_freq_hz=config.hard_az_freq,
            el_amp_deg=config.hard_el_amp,
            el_freq_hz=config.hard_el_freq,
        )
    elif motion_type == "constant":
        return ConstantVelocity(
            az_rate_deg_s=config.clean_az_rate,
            el_rate_deg_s=config.clean_el_rate,
        )
    elif motion_type == "random":
        return RandomManeuvering(
            max_acc_deg_s2=config.clean_max_acc,
            change_interval_s=config.clean_change_interval,
            max_vel_deg_s=config.clean_max_vel,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")


def _build_disturbances(config: Config, scenario_name: str):
    """Build v1.1 disturbance objects from config.

    Returns (vibration, noise, blur, disturbances_dict) where the first
    three are the Simulator-level disturbances and disturbances_dict holds
    the v1.1 disturbances applied in the run loop.
    """
    seed = config.random_seed

    # Select scenario-specific disturbance parameters
    if scenario_name == "clean":
        vib_rms = config.clean_vibration_rms
        noise_sigma = config.clean_noise_sigma
        blur_px = config.clean_blur_pixels
        turb_rms = config.turbulence_rms_deg
        exp_rate = config.exposure_rate_hz
        occ_dur = config.occlusion_duration_s
        fb_count = config.false_beacon_count
    elif scenario_name == "hard":
        vib_rms = config.hard_vibration_rms
        noise_sigma = config.hard_noise_sigma
        blur_px = config.hard_blur_pixels
        turb_rms = config.hard_turbulence_rms_deg
        exp_rate = config.hard_exposure_rate_hz
        occ_dur = config.hard_occlusion_duration_s
        fb_count = config.hard_false_beacon_count
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    vibration = PlatformVibration(
        rms_deg=vib_rms, seed=seed) if vib_rms > 0 else None
    noise = SensorNoise(
        sigma=noise_sigma, seed=seed + 1) if noise_sigma > 0 else None
    blur = MotionBlur(max_blur_pixels=blur_px) if blur_px > 0 else None

    # v1.1 disturbances
    turbulence = AtmosphericTurbulence(
        rms_deg=turb_rms,
        correlation_time_s=config.turbulence_correlation_s,
        seed=seed + 2,
    ) if turb_rms > 0 else None

    exposure = ExposureVariation(
        rate_hz=exp_rate,
        amplitude=config.exposure_amplitude,
        seed=seed + 5,
    ) if exp_rate > 0 else None

    occlusion = Occlusion(
        duration_s=occ_dur,
        frequency_hz=config.occlusion_frequency_hz,
        seed=seed + 3,
    ) if occ_dur > 0 else None

    false_beacons = FalseBeacons(
        count=fb_count,
        seed=seed + 4,
    ) if fb_count > 0 else None

    disturbances = {
        'turbulence': turbulence,
        'exposure': exposure,
        'occlusion': occlusion,
        'false_beacons': false_beacons,
    }

    return vibration, noise, blur, disturbances


def _build_detector(config: Config):
    """Build the detection component based on config.detector_type."""
    classical_det = BeaconDetector(
        fixed_threshold=config.threshold,
        min_area=config.min_area,
        max_area=config.max_area,
    )

    if config.detector_type == "ai":
        if AIDetector is None:
            print("  [Warning] AIDetector not available, falling back to classical")
            return classical_det
        return AIDetector(
            model_path=config.ai_model_path or None,
            fallback=classical_det,
        )
    return classical_det


def _build_filter(config: Config):
    """Build the state estimation filter based on config.filter_type."""
    if config.filter_type == "alpha_beta":
        return AlphaBetaFilter(
            alpha=config.alpha,
            beta=config.beta,
            dt=1.0 / config.fps,
        )
    return KalmanFilter2D(measurement_noise=config.measurement_noise)


def make_scenario(config: Config, scenario_name: str):
    """Create a simulator and controller for a named scenario.

    Returns
    -------
    tuple
        (sim, controller, disturbances, multi_tracker_or_None) where
        disturbances is a dict of v1.1 disturbance objects (may contain
        None values when the corresponding config parameter is zero), and
        multi_tracker_or_None is a MultiTracker when multi_target mode is
        enabled (None otherwise).
    """
    cam = CameraState(
        pan_deg=0.0, tilt_deg=0.0,
        width=config.width, height=config.height,
        h_fov_deg=config.h_fov_deg, v_fov_deg=config.v_fov_deg,
    )

    motion = _build_motion_model(config, scenario_name)
    vibration, noise, blur, disturbances = _build_disturbances(config, scenario_name)
    detector = _build_detector(config)
    kalman = _build_filter(config)

    multi_tracker = None

    if config.multi_target and MultiTargetSimulator is not None:
        # Multi-target mode: create MultiTargetSimulator + MultiTracker
        colors = config.target_colors
        beacons = [
            BeaconConfig(
                id=f"b{i}",
                color=colors[i % len(colors)] if colors else (255, 255, 255),
            )
            for i in range(config.num_targets)
        ]
        sim = MultiTargetSimulator(
            cam=cam, beacons=beacons,
            vibration=vibration,
            turbulence=disturbances.get('turbulence'),
            sensor_noise=noise, motion_blur=blur,
            fps=float(config.fps),
        )
        multi_tracker = MultiTracker(
            beacon_ids=[b.id for b in beacons],
            detector=detector,
            acquire_threshold=config.acquire_threshold,
            lose_threshold=config.lose_threshold,
            reacquire_timeout_frames=config.reacquire_timeout,
            image_width=config.width,
            image_height=config.height,
        )
        multi_tracker.start()

        # Wrap in a single-target-compatible controller using first tracker
        controller = IntegratedController(
            cam=cam, tracker=list(multi_tracker.trackers.values())[0],
            kp=config.kp, ki=config.ki, kd=config.kd,
            dt=1.0 / config.fps,
            pan_range=config.pan_range, tilt_range=config.tilt_range,
            rate_limit_deg_s=config.rate_limit_deg_s,
            deadband_pixels=config.deadband_pixels,
        )
    else:
        # Single-target mode
        sim = Simulator(
            cam=cam, motion=motion,
            beacon_sigma_px=config.beacon_sigma_px,
            beacon_peak=config.beacon_peak,
            vibration=vibration,
            turbulence=disturbances.get('turbulence'),
            sensor_noise=noise, motion_blur=blur,
            fps=config.fps,
        )
        tracker = Tracker(
            detector=detector,
            kalman=kalman,
            acquire_threshold=config.acquire_threshold,
            verify_threshold=config.verify_threshold,
            acquire_error_threshold=config.acquire_error_threshold,
            lose_threshold=config.lose_threshold,
            reacquire_timeout_frames=config.reacquire_timeout,
            image_width=config.width, image_height=config.height,
        )
        tracker.start()
        controller = IntegratedController(
            cam=cam, tracker=tracker,
            kp=config.kp, ki=config.ki, kd=config.kd,
            dt=1.0 / config.fps,
            pan_range=config.pan_range, tilt_range=config.tilt_range,
            rate_limit_deg_s=config.rate_limit_deg_s,
            deadband_pixels=config.deadband_pixels,
        )

    return sim, controller, disturbances, multi_tracker


# ──────────────────────────────────────────────────────────────────────────
# HUD overlay
# ──────────────────────────────────────────────────────────────────────────

# Colors (BGR)
COLOR_HUD_BG = (20, 20, 20)
COLOR_TEXT = (200, 200, 200)
COLOR_TEXT_BRIGHT = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 100, 0)
COLOR_GT = (0, 0, 255)       # red — ground truth (dev only)
COLOR_EST = (0, 255, 0)      # green — estimated position
COLOR_CENTER = (100, 100, 100)  # grey — center crosshair


def draw_hud(frame, state, controller_output, gt, track_output, fps, proc_time_ms,
             dev_mode, config):
    """Draw HUD overlay on the frame in-place. track_output is a TrackerOutput."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ──
    bar_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── State indicator ──
    state_str = state.value if hasattr(state, "value") else str(state)
    state_colors = {
        "IDLE": COLOR_TEXT,
        "SEARCHING": COLOR_YELLOW,
        "CANDIDATE_VERIFICATION": COLOR_YELLOW,
        "ACQUIRING": COLOR_BLUE,
        "TRACKING": COLOR_GREEN,
        "REACQUIRING": COLOR_RED,
        "FAILED": COLOR_RED,
    }
    state_color = state_colors.get(state_str, COLOR_TEXT)
    cv2.rectangle(frame, (10, 10), (280, 44), state_color, -1)
    cv2.putText(frame, f"STATE: {state_str}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Pan / Tilt ──
    cv2.putText(frame, f"PAN:   {controller_output.pan_cmd_deg:+7.2f} deg",
                (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TILT:  {controller_output.tilt_cmd_deg:+7.2f} deg",
                (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Error ──
    err_color = COLOR_GREEN if controller_output.error_pixels < 50 else \
                COLOR_YELLOW if controller_output.error_pixels < 150 else COLOR_RED
    cv2.putText(frame, f"ERROR: {controller_output.error_pixels:6.1f} px",
                (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"  ANG: {abs(controller_output.error_pan_deg):.3f} + {abs(controller_output.error_tilt_deg):.3f} deg",
                (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── FPS / processing time ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (850, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PROC: {proc_time_ms:.1f} ms", (850, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Confidence / detection ──
    if track_output is not None:
        est_x, est_y = track_output.estimated_x, track_output.estimated_y
        det_active = track_output.detected
        conf = track_output.confidence
        det_streak = track_output.consecutive_detections
        miss_streak = track_output.consecutive_misses
    else:
        est_x, est_y = 0.0, 0.0
        det_active = False
        conf = 0.0
        det_streak = 0
        miss_streak = 0
    det_str = "YES" if det_active else "NO"
    cv2.putText(frame, f"DET: {det_str}  CONF: {conf:.2f}", (1050, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"DET_STREAK: {det_streak}  MISS: {miss_streak}",
                (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Lock indicator (green border when tracking) ──
    if state == TrackerState.TRACKING:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_GREEN, 3)
    elif state == TrackerState.REACQUIRING:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RED, 3)

    # ── Center crosshair ──
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), COLOR_CENTER, 1)

    # ── Tracking target zone (central 20% = 10% radius) ──
    target_r = int(w * 0.10)
    cv2.rectangle(frame, (cx - target_r, cy - int(h * 0.10)),
                  (cx + target_r, cy + int(h * 0.10)),
                  (60, 60, 60), 1)

    # ── Estimated position (green circle) ──
    if state in (TrackerState.TRACKING, TrackerState.ACQUIRING, TrackerState.REACQUIRING):
        cv2.circle(frame, (int(est_x), int(est_y)), 8, COLOR_EST, 2)
        cv2.circle(frame, (int(est_x), int(est_y)), 2, COLOR_EST, -1)

    # ── Ground truth (dev mode only — red cross) ──
    if dev_mode and gt is not None:
        gt_list = gt if isinstance(gt, (list, tuple)) else [gt]
        for g in gt_list:
            if hasattr(g, 'target_visible') and g.target_visible:
                gtx, gty = int(g.target_pixel_x), int(g.target_pixel_y)
                cv2.drawMarker(frame, (gtx, gty), COLOR_GT, cv2.MARKER_CROSS, 20, 2)

    # ── Bottom bar: scenario + controls ──
    bot_h = 30
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - bot_h), (w, h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "SPACE:start/pause  R:reset  1:clean  2:hard  G:dev  Q:quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """Per-frame CSV logger. One row per frame."""

    FIELDS = [
        'run_id', 'timestamp', 'frame_id', 'fps',
        'target_id',
        'gt_x', 'gt_y', 'est_x', 'est_y',
        'error_px', 'angular_error_deg',
        'detection_confidence', 'detected',
        'tracker_state', 'pan_cmd', 'tilt_cmd',
        'pan_actual', 'tilt_actual',
        'processing_time_ms',
    ]

    def __init__(self, filepath, run_id, seed):
        self.filepath = filepath
        self.run_id = run_id
        self.seed = seed
        self.rows = []
        self._file = open(filepath, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        # Write header with seed metadata as comments
        self._file.write(f"# run_id={run_id}\n# seed={seed}\n# timestamp={datetime.now().isoformat()}\n")
        self._writer.writeheader()

    def log(self, frame_id, timestamp, fps, gt, track_output, controller_output,
            proc_time_ms):
        """Log one frame. track_output is a TrackerOutput (or None)."""
        if track_output is not None:
            est_x = track_output.estimated_x
            est_y = track_output.estimated_y
            det_active = track_output.detected
            conf = track_output.confidence
            tracker_state = track_output.state.value
        else:
            est_x, est_y = 0.0, 0.0
            det_active = False
            conf = 0.0
            tracker_state = 'IDLE'
        angular_err = np.sqrt(controller_output.error_pan_deg**2 +
                              controller_output.error_tilt_deg**2)

        row = {
            'run_id': self.run_id,
            'timestamp': f"{timestamp:.4f}",
            'frame_id': frame_id,
            'fps': f"{fps:.1f}",
            'target_id': 'beacon_01',
            'gt_x': f"{gt.target_pixel_x:.2f}",
            'gt_y': f"{gt.target_pixel_y:.2f}",
            'est_x': f"{est_x:.2f}",
            'est_y': f"{est_y:.2f}",
            'error_px': f"{controller_output.error_pixels:.2f}",
            'angular_error_deg': f"{angular_err:.4f}",
            'detection_confidence': f"{conf:.4f}",
            'detected': int(det_active),
            'tracker_state': tracker_state,
            'pan_cmd': f"{controller_output.pan_cmd_deg:.4f}",
            'tilt_cmd': f"{controller_output.tilt_cmd_deg:.4f}",
            'pan_actual': f"{gt.camera_pan_deg:.4f}",
            'tilt_actual': f"{gt.camera_tilt_deg:.4f}",
            'processing_time_ms': f"{proc_time_ms:.2f}",
        }
        self._writer.writerow(row)
        self.rows.append(row)

    def close(self):
        self._file.close()

    def summary(self):
        """Compute and return summary statistics."""
        if not self.rows:
            return {}

        errors = [float(r['error_px']) for r in self.rows]
        ang_errors = [float(r['angular_error_deg']) for r in self.rows]
        fps_vals = [float(r['fps']) for r in self.rows]
        proc_times = [float(r['processing_time_ms']) for r in self.rows]
        states = [r['tracker_state'] for r in self.rows]
        detected = [int(r['detected']) for r in self.rows]

        # Acquisition time: first frame in TRACKING
        acq_frames = None
        for i, s in enumerate(states):
            if s == 'TRACKING':
                acq_frames = i
                break

        # Lock retention: percentage of TRACKING frames
        tracking_count = sum(1 for s in states if s == 'TRACKING')

        # Loss events: transitions from TRACKING to REACQUIRING
        losses = 0
        for i in range(1, len(states)):
            if states[i-1] == 'TRACKING' and states[i] == 'REACQUIRING':
                losses += 1

        # Use actual FPS from data (not hardcoded 30.0)
        nominal_fps = float(fps_vals[0]) if fps_vals else 30.0
        if nominal_fps <= 0:
            nominal_fps = 30.0

        # Steady-state error: mean error over last 20% of frames
        ss_start = int(len(errors) * 0.8)
        ss_errors = errors[ss_start:] if ss_start < len(errors) else errors
        steady_state_error = float(np.mean(ss_errors)) if ss_errors else 0.0

        return {
            'duration_s': len(self.rows) / nominal_fps,
            'num_frames': len(self.rows),
            'mean_fps': np.mean(fps_vals),
            'min_fps': np.min(fps_vals),
            'mean_proc_ms': np.mean(proc_times),
            'max_proc_ms': np.max(proc_times),
            'acquisition_time_s': acq_frames / nominal_fps if acq_frames else None,
            'detection_rate': np.mean(detected),
            'mean_error_px': np.mean(errors),
            'max_error_px': np.max(errors),
            'steady_state_error_px': steady_state_error,
            'rms_angular_error_deg': np.sqrt(np.mean(np.square(ang_errors))),
            'lock_retention_pct': tracking_count / len(self.rows) * 100,
            'num_losses': losses,
            'mean_confidence': np.mean([float(r['detection_confidence']) for r in self.rows]),
        }


# ──────────────────────────────────────────────────────────────────────────
# Plot generation
# ──────────────────────────────────────────────────────────────────────────

def generate_plots(csv_path, output_path, scenario_name):
    """Generate an error-over-time plot from a CSV log."""
    timestamps = []
    errors = []
    states = []
    detected = []

    with open(csv_path, 'r') as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith('#')]
    reader = csv.DictReader(lines)
    for row in reader:
        timestamps.append(float(row['timestamp']))
        errors.append(float(row['error_px']))
        states.append(row['tracker_state'])
        detected.append(int(row['detected']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Error plot
    ax1.plot(timestamps, errors, color='#2E86AB', linewidth=1.2, label='Tracking error')
    ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50px threshold')
    ax1.axhline(y=128, color='orange', linestyle='--', alpha=0.5, label='Central 20% boundary')
    ax1.set_ylabel('Pixel Error')
    ax1.set_title(f'{scenario_name.upper()} Scenario — Tracking Error Over Time')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(max(errors) * 1.1, 200))
    ax1.grid(True, alpha=0.3)

    # Color background by state
    state_colors = {
        'SEARCHING': '#FFF3CD',
        'ACQUIRING': '#FFE0B2',
        'TRACKING': '#C8E6C9',
        'REACQUIRING': '#FFCDD2',
    }
    for i in range(len(states)):
        color = state_colors.get(states[i], 'white')
        ax1.axvspan(timestamps[i] - 0.016, timestamps[i] + 0.016, alpha=0.3, color=color)

    # Detection bar
    ax2.fill_between(timestamps, 0, detected, color='#2E86AB', alpha=0.7)
    ax2.set_ylabel('Detected')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────

def run_scenario(scenario_name, config, output_dir, headless=False):
    """Run one scenario end-to-end and produce CSV + plot."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name.upper()}")
    print(f"{'='*60}")

    sim, controller, disturbances, multi_tracker = make_scenario(config, scenario_name)
    num_frames = int(config.duration_s * config.fps)

    run_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = os.path.join(output_dir, f"{run_id}.csv")
    plot_path = os.path.join(output_dir, f"{run_id}_plot.png")
    frames_dir = os.path.join(output_dir, f"{run_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    logger = CSVLogger(csv_path, run_id, config.random_seed)

    print(f"  Duration: {config.duration_s}s ({num_frames} frames)")
    print(f"  Output: {csv_path}")

    running = True
    paused = False
    dev_mode = False
    frame_id = 0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    while frame_id < num_frames:
        if not headless and not paused:
            # Check for key presses (non-blocking)
            pass

        if not running:
            break

        t0 = time.time()

        if not paused:
            # ── The main loop ──
            # Camera-level disturbances (vibration, turbulence) are now applied
            # inside sim.step() on a copy of the camera state (never on sim.cam).
            frame, gt = sim.step(frame_id)
            gt_primary = gt[0] if isinstance(gt, (list, tuple)) else gt
            ts = gt_primary.timestamp if hasattr(gt_primary, 'timestamp') else (frame_id / config.fps)

            # ── Apply v1.1 frame-level disturbances after rendering ──
            if disturbances.get('occlusion'):
                frame = disturbances['occlusion'].apply(frame, ts)
            if disturbances.get('false_beacons'):
                frame = disturbances['false_beacons'].apply(frame, ts)
            if disturbances.get('exposure'):
                frame = disturbances['exposure'].apply(frame, ts)

            if multi_tracker is not None:
                # Multi-target mode: run multi_tracker, then compute camera
                # commands from the first sub-tracker's output.
                mt_output = multi_tracker.track(frame)
                first_bid = multi_tracker.beacon_ids[0]
                first_tracklet = mt_output.tracklets[first_bid]

                if first_tracklet.state == TrackerState.SEARCHING:
                    new_pan, new_tilt = controller.search.next_command(
                        sim.cam.pan_deg, sim.cam.tilt_deg, controller.dt)
                    cmd = ControllerOutput(
                        pan_cmd_deg=new_pan, tilt_cmd_deg=new_tilt,
                        pan_rate_cmd_deg_s=(new_pan - sim.cam.pan_deg) / controller.dt,
                        tilt_rate_cmd_deg_s=(new_tilt - sim.cam.tilt_deg) / controller.dt,
                        error_pan_deg=0.0, error_tilt_deg=0.0,
                        error_pixels=0.0, saturated=False,
                    )
                else:
                    cmd = controller.controller.compute(
                        first_tracklet.estimated_x, first_tracklet.estimated_y,
                        sim.cam.pan_deg, sim.cam.tilt_deg)

                # Sync the first sub-tracker's state for HUD display
                controller.tracker = multi_tracker.trackers[first_bid]
            else:
                cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)

            sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)

            proc_time_ms = (time.time() - t0) * 1000

            # Log
            logger.log(frame_id, ts, current_fps, gt_primary,
                       controller.last_track_output, cmd, proc_time_ms)

            # Print milestones
            if frame_id == 0:
                print(f"  Frame 0: state={controller.tracker.state.value}")
            if frame_id == 2 and controller.tracker.state.value == 'TRACKING':
                print(f"  Frame {frame_id}: entered TRACKING ({frame_id/config.fps:.1f}s)")

            if frame_id % 150 == 0 and frame_id > 0:
                print(f"  Frame {frame_id} ({frame_id/config.fps:.1f}s): "
                      f"state={controller.tracker.state.value}, "
                      f"error={cmd.error_pixels:.1f}px")

            frame_id += 1
        else:
            proc_time_ms = 0.0

        # ── Save & Display ──
        if not paused:
            save_frame = frame.copy()
            track_out = controller.last_track_output
            hud_state = track_out.state if track_out else controller.tracker.state
            draw_hud(save_frame, hud_state, cmd, gt,
                     track_out, current_fps, proc_time_ms,
                     dev_mode, config)
            if config.save_frames:
                frame_path = os.path.join(frames_dir, f"frame_{frame_id - 1:05d}.jpg")
                cv2.imwrite(frame_path, save_frame)

        if not headless:
            if paused:
                display_frame = (frame.copy() * 0.5).astype(np.uint8)
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "PAUSED", (w//2 - 50, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT_BRIGHT, 3)
            else:
                display_frame = save_frame

            cv2.imshow("FSOC Virtual Camera Tracking", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                running = False
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                sim.reset()
                controller.reset()
                frame_id = 0
                logger.close()
                logger = CSVLogger(csv_path, run_id, config.random_seed)
                print("  Reset.")
            elif key == ord('g'):
                if not headless:
                    dev_mode = not dev_mode
                    print(f"  Dev mode: {'ON' if dev_mode else 'OFF'}")

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

    # ── Post-run ──
    logger.close()
    print(f"\n  Run complete: {frame_id} frames")

    stats = logger.summary()
    print(f"\n  Summary:")
    print(f"    Duration:            {stats.get('duration_s', 0):.1f}s")
    print(f"    Mean FPS:            {stats.get('mean_fps', 0):.1f}")
    print(f"    Mean processing:     {stats.get('mean_proc_ms', 0):.2f} ms")
    print(f"    Acquisition time:    {stats.get('acquisition_time_s', 'N/A')}")
    if stats.get('acquisition_time_s'):
        print(f"                         {stats['acquisition_time_s']:.2f}s")
    print(f"    Detection rate:      {stats.get('detection_rate', 0)*100:.1f}%")
    print(f"    Mean error:          {stats.get('mean_error_px', 0):.1f} px")
    print(f"    Max error:           {stats.get('max_error_px', 0):.1f} px")
    print(f"    RMS angular error:   {stats.get('rms_angular_error_deg', 0):.4f} deg")
    print(f"    Lock retention:      {stats.get('lock_retention_pct', 0):.1f}%")
    print(f"    Loss events:         {stats.get('num_losses', 0)}")
    print(f"    Mean confidence:     {stats.get('mean_confidence', 0):.3f}")

    # Generate plot
    generate_plots(csv_path, plot_path, scenario_name)
    print(f"  Frames saved to {frames_dir}")

    # Generate structured JSON summary and HTML report
    try:
        reporter = SummaryReporter()
        config_dict = {
            'fps': config.fps, 'duration_s': config.duration_s,
            'random_seed': config.random_seed,
            'width': config.width, 'height': config.height,
            'kp': config.kp, 'ki': config.ki, 'kd': config.kd,
            'threshold': config.threshold,
            'scenario': scenario_name,
        }
        summary = reporter.generate(csv_path, output_dir, config=config_dict)
        ReportGenerator.generate_html(csv_path, summary, output_dir)
    except Exception as e:
        print(f"  [Report generation failed: {e}]")

    if not headless:
        cv2.destroyAllWindows()

    return stats, csv_path, plot_path


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FSOC Virtual Camera Tracking System")
    print("  AI-Based Coarse-Alignment Simulator")
    print("=" * 60)
    print()

    # ── CLI argument parsing with argparse ───────────────────────────
    parser = argparse.ArgumentParser(description='FSOC Coarse-Alignment Simulator')
    parser.add_argument('--scenario', choices=['clean', 'hard'], default='clean',
                        help='Scenario to run (default: clean)')
    parser.add_argument('--yaml', type=str, default=None,
                        help='Path to YAML scenario config file')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation framework (headless)')
    parser.add_argument('--headless', action='store_true',
                        help='Run headless (no display)')
    parser.add_argument('--gui', action='store_true',
                        help='Launch PySide6 GUI')
    parser.add_argument('--multi-target', action='store_true',
                        help='Run multi-target scenario')
    parser.add_argument('--detector', choices=['classical', 'ai'], default=None,
                        help='Detector type (overrides config)')
    parser.add_argument('--filter', choices=['kalman', 'alpha_beta'], default=None,
                        help='Filter type (overrides config)')
    parser.add_argument('--model', type=str, default='',
                        help='Path to ONNX model for AI detector')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration override for eval mode')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for eval mode')
    args = parser.parse_args()

    # ── Headless evaluation mode ──────────────────────────────────────
    if args.eval:
        from evaluation import (
            run_full_evaluation,
            CATEGORIES,
            DEFAULT_SEEDS,
        )
        print("  Running evaluation framework...")
        print()

        eval_output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results")
        os.makedirs(eval_output, exist_ok=True)

        duration_s = args.duration if args.duration is not None else 10
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else DEFAULT_SEEDS

        report = run_full_evaluation(
            seeds=seeds,
            duration_s=duration_s,
            output_dir=eval_output,
            quiet=True,
        )

        # Print the evaluation matrix
        print("\n" + "=" * 60)
        print("  FSOC EVALUATION MATRIX")
        print("=" * 60)
        for cat_name in CATEGORIES:
            score = report.get(cat_name, 0.0)
            print(f"  {cat_name:<24} {score:>6.1f} / 100")
        print("-" * 60)
        print(f"  {'OVERALL':<24} {report.get('overall', 0.0):>6.1f} / 100")
        print("=" * 60)

        # Save results as JSON
        import json
        results_path = os.path.join(eval_output, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_path}")
        return

    # ── GUI mode ─────────────────────────────────────────────────────
    if args.gui:
        try:
            from gui.main_window import MainWindow
            from PySide6.QtWidgets import QApplication
            app = QApplication(sys.argv)
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
        except ImportError:
            print("Error: PySide6 is required for GUI mode.")
            print("Install with: pip install PySide6")
            sys.exit(1)

    print("  Controls:")
    print("    SPACE — start/pause")
    print("    R     — reset")
    print("    1     — clean scenario")
    print("    2     — hard scenario")
    print("    G     — toggle ground-truth overlay")
    print("    Q/ESC — quit")
    print()

    # ── Load config from YAML or use defaults ────────────────────────
    if args.yaml:
        from yaml_config import load_scenario, scenario_to_config
        scenario_dict = load_scenario(args.yaml)
        config = scenario_to_config(scenario_dict)
        print(f"  Loaded config from: {args.yaml}")
    else:
        config = Config()

    # ── Apply CLI overrides to config ────────────────────────────────
    if args.filter:
        config.filter_type = args.filter
    if args.detector:
        config.detector_type = args.detector
    if args.model:
        config.ai_model_path = args.model
    if args.multi_target:
        config.multi_target = True
    if args.duration is not None:
        config.duration_s = args.duration

    # Shorten duration for testing; restore to 60 for the real demo
    if os.environ.get('FSOC_TEST_MODE', '0') == '1':
        config.duration_s = 10

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    scenario = args.scenario

    # ── Headless detection ────────────────────────────────────────────
    headless = args.headless
    if not headless:
        display_val = os.environ.get('DISPLAY', '')
        if display_val == '' or display_val == 'needs-to-be-defined':
            headless = True
        else:
            try:
                test = np.zeros((1, 1, 3), dtype=np.uint8)
                cv2.imshow("test", test)
                cv2.waitKey(1)
                cv2.destroyWindow("test")
            except Exception:
                headless = True

    if headless:
        print("  [Headless mode - no display available]")
        print("  Running both scenarios automatically...\n")

    if headless:
        # If a specific scenario was requested, only run that one
        if args.scenario:
            scenarios_to_run = [args.scenario]
        else:
            scenarios_to_run = ["clean", "hard"]

        # Run scenarios and produce reports
        all_stats = {}
        for sc in scenarios_to_run:
            stats, csv_path, plot_path = run_scenario(sc, config, output_dir, headless=True)
            all_stats[sc] = (stats, csv_path, plot_path)

        # Print comparison (only when running multiple scenarios)
        if len(scenarios_to_run) > 1:
            print("\n" + "=" * 60)
            print("  COMPARISON")
            print("=" * 60)
            print(f"{'Metric':<25} {'Clean':>12} {'Hard':>12}")
            print("-" * 50)
            for key in ['acquisition_time_s', 'mean_error_px', 'max_error_px',
                         'lock_retention_pct', 'num_losses', 'detection_rate',
                         'rms_angular_error_deg', 'mean_fps']:
                clean_val = all_stats.get('clean', ({}))[0].get(key, 'N/A') if 'clean' in all_stats else 'N/A'
                hard_val = all_stats.get('hard', ({}))[0].get(key, 'N/A') if 'hard' in all_stats else 'N/A'
                if isinstance(clean_val, float) and isinstance(hard_val, float):
                    print(f"{key:<25} {clean_val:>12.2f} {hard_val:>12.2f}")
                else:
                    print(f"{key:<25} {str(clean_val):>12} {str(hard_val):>12}")

        print(f"\n  CSV logs and plots saved to: {output_dir}/")
        return

    # Interactive mode
    print(f"  Starting scenario: {scenario.upper()}")
    print(f"  Output directory: {output_dir}/")
    print()

    while True:
        run_scenario(scenario, config, output_dir, headless=False)
        print("\n  Press 1 for clean, 2 for hard, or Q to quit.")
        # Wait for next scenario selection
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                scenario = "clean"
                break
            elif key == ord('2'):
                scenario = "hard"
                break
            elif key == ord('q') or key == 27:
                return


if __name__ == "__main__":
    main()
