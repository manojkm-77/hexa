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
                 GroundTruth)
from detect import BeaconDetector, KalmanFilter2D, Tracker, TrackerState
from control import IntegratedController, ControllerOutput
from report import SummaryReporter, ReportGenerator


# ──────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────────

def make_scenario(config: Config, scenario_name: str):
    """Create a simulator and controller for a named scenario."""
    cam = CameraState(
        pan_deg=0.0, tilt_deg=0.0,
        width=config.width, height=config.height,
        h_fov_deg=config.h_fov_deg, v_fov_deg=config.v_fov_deg,
    )

    if scenario_name == "clean":
        motion = CircularMotion(
            center_az_deg=config.clean_center_az,
            center_el_deg=config.clean_center_el,
            radius_deg=config.clean_radius,
            angular_speed_deg_s=config.clean_speed,
        )
        vibration = None
        noise = None
        blur = None

    elif scenario_name == "hard":
        motion = SinusoidalMotion(
            az_amp_deg=config.hard_az_amp,
            az_freq_hz=config.hard_az_freq,
            el_amp_deg=config.hard_el_amp,
            el_freq_hz=config.hard_el_freq,
        )
        vibration = PlatformVibration(
            rms_deg=config.hard_vibration_rms, seed=config.random_seed)
        noise = SensorNoise(
            sigma=config.hard_noise_sigma, seed=config.random_seed + 1)
        blur = MotionBlur(max_blur_pixels=config.hard_blur_pixels)

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    sim = Simulator(
        cam=cam, motion=motion,
        beacon_sigma_px=config.beacon_sigma_px,
        beacon_peak=config.beacon_peak,
        vibration=vibration, sensor_noise=noise, motion_blur=blur,
        fps=config.fps,
    )

    tracker = Tracker(
        detector=BeaconDetector(
            fixed_threshold=config.threshold,
            min_area=config.min_area, max_area=config.max_area,
        ),
        kalman=KalmanFilter2D(measurement_noise=config.measurement_noise),
        acquire_threshold=config.acquire_threshold,
        lose_threshold=config.lose_threshold,
        reacquire_timeout_frames=config.reacquire_timeout,
        image_width=config.width, image_height=config.height,
    )

    controller = IntegratedController(
        cam=cam, tracker=tracker,
        kp=config.kp, ki=config.ki, kd=config.kd,
        dt=1.0 / config.fps,
        pan_range=config.pan_range, tilt_range=config.tilt_range,
        rate_limit_deg_s=config.rate_limit_deg_s,
        deadband_pixels=config.deadband_pixels,
    )

    return sim, controller


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


def draw_hud(frame, state, controller_output, gt, tracker, fps, proc_time_ms,
             dev_mode, config):
    """Draw HUD overlay on the frame in-place."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ──
    bar_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── State indicator ──
    state_colors = {
        TrackerState.SEARCHING: COLOR_YELLOW,
        TrackerState.ACQUIRING: COLOR_YELLOW,
        TrackerState.TRACKING: COLOR_GREEN,
        TrackerState.REACQUIRING: COLOR_RED,
    }
    state_color = state_colors.get(state, COLOR_TEXT)
    cv2.rectangle(frame, (10, 10), (280, 44), state_color, -1)
    cv2.putText(frame, f"STATE: {state.value}", (20, 35),
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
    est_x, est_y = tracker.kalman.get_position()
    det_active = tracker.last_detection is not None and tracker.last_detection.detected
    conf = tracker.last_detection.confidence if tracker.last_detection else 0.0
    det_str = "YES" if det_active else "NO"
    cv2.putText(frame, f"DET: {det_str}  CONF: {conf:.2f}", (1050, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"DET_STREAK: {tracker.consecutive_detections}  MISS: {tracker.consecutive_misses}",
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
    if dev_mode and gt.target_visible:
        gtx, gty = int(gt.target_pixel_x), int(gt.target_pixel_y)
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

    def log(self, frame_id, timestamp, fps, gt, tracker, controller_output,
            proc_time_ms):
        """Log one frame."""
        est_x, est_y = tracker.kalman.get_position()
        angular_err = np.sqrt(controller_output.error_pan_deg**2 +
                              controller_output.error_tilt_deg**2)
        det_active = (tracker.last_detection is not None and
                      tracker.last_detection.detected)
        conf = tracker.last_detection.confidence if tracker.last_detection else 0.0

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
            'tracker_state': tracker.state.value,
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

        return {
            'duration_s': len(self.rows) / 30.0,
            'num_frames': len(self.rows),
            'mean_fps': np.mean(fps_vals),
            'min_fps': np.min(fps_vals),
            'mean_proc_ms': np.mean(proc_times),
            'max_proc_ms': np.max(proc_times),
            'acquisition_time_s': acq_frames / 30.0 if acq_frames else None,
            'detection_rate': np.mean(detected),
            'mean_error_px': np.mean(errors),
            'max_error_px': np.max(errors),
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

    sim, controller = make_scenario(config, scenario_name)
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
            frame, gt = sim.step(frame_id)
            cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
            sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)

            proc_time_ms = (time.time() - t0) * 1000

            # Log
            logger.log(frame_id, gt.timestamp, current_fps, gt,
                       controller.tracker, cmd, proc_time_ms)

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
            draw_hud(save_frame, controller.tracker.state, cmd, gt,
                     controller.tracker, current_fps, proc_time_ms,
                     dev_mode, config)
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
    print("  Controls:")
    print("    SPACE — start/pause")
    print("    R     — reset")
    print("    1     — clean scenario")
    print("    2     — hard scenario")
    print("    G     — toggle ground-truth overlay")
    print("    Q/ESC — quit")
    print()

    config = Config()
    # Shorten duration for testing; restore to 60 for the real demo
    if os.environ.get('FSOC_TEST_MODE', '0') == '1':
        config.duration_s = 10
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Default scenario
    scenario = "clean"

    # Check if running headless (no display)
    headless = False
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
        # Run both scenarios and produce reports
        all_stats = {}
        for sc in ["clean", "hard"]:
            stats, csv_path, plot_path = run_scenario(sc, config, output_dir, headless=True)
            all_stats[sc] = (stats, csv_path, plot_path)

        # Print comparison
        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<25} {'Clean':>12} {'Hard':>12}")
        print("-" * 50)
        for key in ['acquisition_time_s', 'mean_error_px', 'max_error_px',
                     'lock_retention_pct', 'num_losses', 'detection_rate',
                     'rms_angular_error_deg', 'mean_fps']:
            clean_val = all_stats['clean'][0].get(key, 'N/A')
            hard_val = all_stats['hard'][0].get(key, 'N/A')
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
