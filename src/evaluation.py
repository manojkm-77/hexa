"""
evaluation.py -- Evaluation framework for the FSOC coarse-alignment simulator.

Provides:
  - MetricsCalculator: compute all PRD metrics from a CSV log
  - ScenarioRunner:    run a scenario N times with different random seeds
  - EvaluationMatrix:  8-category weighted scoring matrix

Usage:
    from evaluation import MetricsCalculator, ScenarioRunner, EvaluationMatrix

    # Compute metrics from an existing log
    mc = MetricsCalculator()
    metrics = mc.compute("results/clean_run.csv")

    # Run a scenario with multiple seeds
    runner = ScenarioRunner(output_dir="results")
    results = runner.run_scenario("Nominal", seeds=[42, 123, 456], duration_s=30)

    # Build the evaluation matrix
    matrix = EvaluationMatrix()
    report = matrix.evaluate({"Nominal": results["metrics"], "Motion": ...})
"""

import csv
import os
import time
import numpy as np
from typing import Optional

from config import Config
from sim import (
    Simulator, CameraState, TargetState,
    CircularMotion, SinusoidalMotion, RandomManeuvering, ConstantVelocity,
    PlatformVibration, SensorNoise, MotionBlur, AtmosphericTurbulence,
    ExposureVariation, Occlusion, FalseBeacons,
)
from detect import BeaconDetector, KalmanFilter2D, Tracker
from control import IntegratedController


# ---------------------------------------------------------------------------
# Evaluation categories and scoring (PRD sections 20 and 25)
# ---------------------------------------------------------------------------

CATEGORIES = {
    'nominal': {
        'weight': 0.15,
        'metrics': ['acquisition_time_s', 'mean_pixel_error'],
    },
    'motion': {
        'weight': 0.15,
        'metrics': ['lock_retention_pct', 'max_pixel_error'],
    },
    'vibration': {
        'weight': 0.12,
        'metrics': ['rms_angular_error_deg', 'lock_retention_pct'],
    },
    'turbulence': {
        'weight': 0.10,
        'metrics': ['lock_retention_pct', 'mean_pixel_error'],
    },
    'sensor_degradation': {
        'weight': 0.12,
        'metrics': ['detection_rate_pct', 'mean_fps'],
    },
    'clutter': {
        'weight': 0.10,
        'metrics': ['detection_rate_pct', 'lock_retention_pct'],
    },
    'occlusion': {
        'weight': 0.12,
        'metrics': ['reacquisition_time_s', 'lock_retention_pct'],
    },
    'multitarget': {
        'weight': 0.14,
        'metrics': ['lock_retention_pct', 'mean_pixel_error'],
    },
}

# Per-metric scoring parameters.
#   target  -- ideal value that maps to score 100
#   baseline -- value that maps to score 0
#   higher_better -- True means larger values score higher
_METRIC_SCORING = {
    'acquisition_time_s':  {'target': 0.5,   'baseline': 5.0,   'higher_better': False},
    'mean_pixel_error':    {'target': 25.0,  'baseline': 200.0, 'higher_better': False},
    'max_pixel_error':     {'target': 100.0, 'baseline': 500.0, 'higher_better': False},
    'steady_state_error_px': {'target': 10.0, 'baseline': 100.0, 'higher_better': False},
    'lock_retention_pct':  {'target': 95.0,  'baseline': 50.0,  'higher_better': True},
    'detection_rate_pct':  {'target': 100.0, 'baseline': 60.0,  'higher_better': True},
    'rms_angular_error_deg': {'target': 0.5, 'baseline': 5.0,   'higher_better': False},
    'loss_events':         {'target': 0,     'baseline': 10,    'higher_better': False},
    'mean_fps':            {'target': 30.0,  'baseline': 10.0,  'higher_better': True},
    'mean_processing_time_ms': {'target': 15.0, 'baseline': 50.0, 'higher_better': False},
    'reacquisition_time_s': {'target': 0.2,  'baseline': 3.0,   'higher_better': False},
}

# Default seeds (PRD section 25.5: at least 3 different seeds)
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Pre-defined scenario configurations (PRD section 25.2)
_SCENARIO_CONFIGS = {
    'nominal': {
        'motion': 'circular',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
    'motion': {
        'motion': 'random',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
    'vibration': {
        'motion': 'circular',
        'vibration_rms': 0.2, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
    'turbulence': {
        'motion': 'circular',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.1,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
    'sensor_degradation': {
        'motion': 'circular',
        'vibration_rms': 0.0, 'noise_sigma': 12.0,
        'blur_pixels': 3.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': True,
    },
    'clutter': {
        'motion': 'circular',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 2,
        'exposure_variation': False,
    },
    'occlusion': {
        'motion': 'circular',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': True, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
    'multitarget': {
        'motion': 'sinusoidal',
        'vibration_rms': 0.0, 'noise_sigma': 0.0,
        'blur_pixels': 0.0, 'turbulence_rms': 0.0,
        'occlusion_enabled': False, 'false_beacon_count': 0,
        'exposure_variation': False,
    },
}


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """Compute all PRD metrics from a CSV log produced by CSVLogger.

    Metrics:
        acquisition_time_s, mean_pixel_error, steady_state_error_px,
        max_pixel_error, lock_retention_pct, detection_rate_pct,
        rms_angular_error_deg, loss_events, reacquisition_time_s,
        mean_fps, mean_processing_time_ms, num_frames, duration_s
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

    def compute(self, csv_path: str) -> dict:
        """Read a CSV log and compute all metrics.

        Returns a dict with keys matching the metric names used in CATEGORIES
        and _METRIC_SCORING.
        """
        rows = self._read_csv(csv_path)
        if not rows:
            return self._empty_metrics()

        errors = np.array([float(r['error_px']) for r in rows])
        ang_errors = np.array([float(r['angular_error_deg']) for r in rows])
        fps_vals = np.array([float(r['fps']) for r in rows])
        proc_times = np.array([float(r['processing_time_ms']) for r in rows])
        states = [r['tracker_state'] for r in rows]
        detected = np.array([int(r['detected']) for r in rows])

        num_frames = len(rows)

        # -- Acquisition time: first frame entering TRACKING --
        acquisition_frame = None
        for i, s in enumerate(states):
            if s == 'TRACKING':
                acquisition_frame = i
                break
        acquisition_time_s = (
            acquisition_frame / self.fps if acquisition_frame is not None
            else None
        )

        # -- Lock retention: fraction of TRACKING frames --
        tracking_count = sum(1 for s in states if s == 'TRACKING')
        lock_retention_pct = tracking_count / num_frames * 100.0

        # -- Loss events: TRACKING -> REACQUIRING transitions --
        loss_events = 0
        for i in range(1, num_frames):
            if states[i - 1] == 'TRACKING' and states[i] == 'REACQUIRING':
                loss_events += 1

        # -- Reacquisition time: mean duration of REACQUIRING episodes --
        reacq_times: list[float] = []
        in_loss = False
        loss_start = 0
        for i, s in enumerate(states):
            if s == 'REACQUIRING' and not in_loss:
                if i > 0 and states[i - 1] == 'TRACKING':
                    in_loss = True
                    loss_start = i
            elif s == 'TRACKING' and in_loss:
                reacq_times.append((i - loss_start) / self.fps)
                in_loss = False
        reacquisition_time_s = (
            float(np.mean(reacq_times)) if reacq_times else None
        )

        # -- Steady-state error: mean of last 30 frames --
        n_steady = min(30, num_frames)
        steady_state_error_px = float(np.mean(errors[-n_steady:]))

        # -- Detection rate --
        detection_rate_pct = float(np.mean(detected)) * 100.0

        # -- RMS angular error --
        rms_angular_error_deg = float(np.sqrt(np.mean(ang_errors ** 2)))

        return {
            'acquisition_time_s': acquisition_time_s,
            'mean_pixel_error': float(np.mean(errors)),
            'steady_state_error_px': steady_state_error_px,
            'max_pixel_error': float(np.max(errors)),
            'lock_retention_pct': lock_retention_pct,
            'detection_rate_pct': detection_rate_pct,
            'rms_angular_error_deg': rms_angular_error_deg,
            'loss_events': loss_events,
            'reacquisition_time_s': reacquisition_time_s,
            'mean_fps': float(np.mean(fps_vals)),
            'mean_processing_time_ms': float(np.mean(proc_times)),
            'num_frames': num_frames,
            'duration_s': num_frames / self.fps,
        }

    # -- helpers --

    @staticmethod
    def _read_csv(csv_path: str) -> list[dict]:
        """Read CSV, skipping comment header lines written by CSVLogger."""
        with open(csv_path, 'r', newline='') as f:
            lines = [line for line in f if not line.startswith('#')]
        reader = csv.DictReader(lines)
        return list(reader)

    @staticmethod
    def _empty_metrics() -> dict:
        """Return a metrics dict with zero / None values."""
        return {
            'acquisition_time_s': None,
            'mean_pixel_error': 0.0,
            'steady_state_error_px': 0.0,
            'max_pixel_error': 0.0,
            'lock_retention_pct': 0.0,
            'detection_rate_pct': 0.0,
            'rms_angular_error_deg': 0.0,
            'loss_events': 0,
            'reacquisition_time_s': None,
            'mean_fps': 0.0,
            'mean_processing_time_ms': 0.0,
            'num_frames': 0,
            'duration_s': 0.0,
        }


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------

class ScenarioRunner:
    """Run a named scenario with multiple random seeds.

    Usage:
        runner = ScenarioRunner(output_dir="results")
        results = runner.run_scenario("Nominal", seeds=[42, 123], duration_s=10)
    """

    def __init__(self, output_dir: str = 'results', fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.calculator = MetricsCalculator(fps=fps)

    def run_scenario(self, scenario_name: str, seeds: list[int],
                     duration_s: int = 10,
                     quiet: bool = True) -> dict:
        """Run a scenario with each seed and return per-run metrics.

        Args:
            scenario_name: Key into _SCENARIO_CONFIGS (e.g. 'Nominal').
            seeds: Random seeds, one per run.
            duration_s: Duration of each run in seconds.
            quiet: Suppress per-frame print output.

        Returns:
            {
                'scenario': scenario_name,
                'metrics': [metrics_dict_per_seed, ...],
                'aggregated': {metric: {'mean': ..., 'std': ...}, ...},
            }
        """
        os.makedirs(self.output_dir, exist_ok=True)

        all_metrics = []
        for seed in seeds:
            m = self._run_single(scenario_name, seed, duration_s, quiet)
            all_metrics.append(m)

        aggregated = _aggregate_runs(all_metrics)
        return {
            'scenario': scenario_name,
            'metrics': all_metrics,
            'aggregated': aggregated,
        }

    # -- internal --

    def _run_single(self, scenario_name: str, seed: int,
                    duration_s: int, quiet: bool) -> dict:
        """Run a single scenario with a specific seed."""
        from main import CSVLogger

        cfg = _SCENARIO_CONFIGS[scenario_name]
        sim, controller, disturbances = _build_simulator(cfg, seed, self.fps)
        num_frames = int(duration_s * self.fps)

        run_id = f"eval_{scenario_name}_s{seed}"
        csv_path = os.path.join(self.output_dir, f"{run_id}.csv")
        logger = CSVLogger(csv_path, run_id, seed)

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        for frame_id in range(num_frames):
            t0 = time.time()
            frame, gt = sim.step(frame_id)

            # Apply v1.1 frame-level disturbances (turbulence is now inside step())
            if disturbances.get('occlusion'):
                frame = disturbances['occlusion'].apply(frame, sim.t)
            if disturbances.get('false_beacons'):
                frame = disturbances['false_beacons'].apply(frame, sim.t)
            if disturbances.get('exposure'):
                frame = disturbances['exposure'].apply(frame, sim.t)

            cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
            sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)
            proc_time_ms = (time.time() - t0) * 1000

            logger.log(frame_id, gt.timestamp, current_fps, gt,
                       controller.last_track_output, cmd, proc_time_ms)

            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            if not quiet and frame_id % 150 == 0 and frame_id > 0:
                print(f"    Frame {frame_id}: state={controller.tracker.state.value}, "
                      f"error={cmd.error_pixels:.1f}px")

        logger.close()
        metrics = self.calculator.compute(csv_path)
        metrics['seed'] = seed
        metrics['csv_path'] = csv_path
        return metrics


# ---------------------------------------------------------------------------
# Helper: build a simulator + controller from a scenario config dict
# ---------------------------------------------------------------------------

def _build_simulator(cfg: dict, seed: int, fps: int = 30) -> tuple:
    """Create Simulator, IntegratedController from a scenario config dict."""
    cam = CameraState(
        pan_deg=0.0, tilt_deg=0.0,
        width=1280, height=720,
        h_fov_deg=60.0, v_fov_deg=40.0,
    )

    # -- Motion model --
    motion_type = cfg.get('motion', 'circular')
    if motion_type == 'circular':
        motion = CircularMotion(
            center_az_deg=5.0, center_el_deg=3.0,
            radius_deg=8.0, angular_speed_deg_s=4.0,
        )
    elif motion_type == 'sinusoidal':
        motion = SinusoidalMotion(
            az_amp_deg=10.0, az_freq_hz=0.2,
            el_amp_deg=6.0, el_freq_hz=0.3,
        )
    elif motion_type == 'random':
        motion = RandomManeuvering(
            max_acc_deg_s2=10.0, change_interval_s=2.0,
            max_vel_deg_s=20.0, seed=seed,
        )
    elif motion_type == 'constant':
        motion = ConstantVelocity(az_rate_deg_s=5.0, el_rate_deg_s=2.0)
    else:
        raise ValueError(f"Unknown motion model: {motion_type}")

    # -- Disturbances --
    vib_rms = cfg.get('vibration_rms', 0.0)
    vibration = (PlatformVibration(rms_deg=vib_rms, seed=seed)
                 if vib_rms > 0 else None)

    noise_sigma = cfg.get('noise_sigma', 0.0)
    sensor_noise = (SensorNoise(sigma=noise_sigma, seed=seed + 1)
                    if noise_sigma > 0 else None)

    blur_px = cfg.get('blur_pixels', 0.0)
    motion_blur = (MotionBlur(max_blur_pixels=blur_px)
                   if blur_px > 0 else None)

    # Build turbulence before Simulator so it can be passed as constructor param
    turb_rms = cfg.get('turbulence_rms', 0.0)
    turbulence = (AtmosphericTurbulence(rms_deg=turb_rms, seed=seed + 2)
                  if turb_rms > 0 else None)

    sim = Simulator(
        cam=cam, motion=motion,
        beacon_sigma_px=4.0, beacon_peak=255.0,
        vibration=vibration, turbulence=turbulence,
        sensor_noise=sensor_noise,
        motion_blur=motion_blur, fps=float(fps),
    )

    occlusion = (Occlusion(seed=seed + 3)
                 if cfg.get('occlusion_enabled', False) else None)

    fb_count = cfg.get('false_beacon_count', 0)
    false_beacons = (FalseBeacons(count=fb_count, seed=seed + 4)
                     if fb_count > 0 else None)

    exposure = (ExposureVariation(seed=seed + 5)
                if cfg.get('exposure_variation', False) else None)

    disturbances = {
        'turbulence': turbulence,
        'occlusion': occlusion,
        'false_beacons': false_beacons,
        'exposure': exposure,
    }

    # -- Tracker and controller --
    tracker = Tracker(
        detector=BeaconDetector(
            fixed_threshold=80, min_area=2, max_area=500,
        ),
        kalman=KalmanFilter2D(measurement_noise=8.0),
        acquire_threshold=3, lose_threshold=5,
        reacquire_timeout_frames=150,
        image_width=1280, image_height=720,
    )
    tracker.start()
    controller = IntegratedController(
        cam=cam, tracker=tracker,
        kp=1.0, ki=0.0, kd=0.15,
        dt=1.0 / float(fps),
        pan_range=(-180.0, 180.0), tilt_range=(-30.0, 90.0),
        rate_limit_deg_s=60.0, deadband_pixels=3.0,
    )

    return sim, controller, disturbances


# ---------------------------------------------------------------------------
# EvaluationMatrix
# ---------------------------------------------------------------------------

class EvaluationMatrix:
    """8-category evaluation matrix with weighted scoring.

    Each category maps to one or more metrics.  Each metric is scored on a
    0--100 scale using linear interpolation between a *baseline* value
    (score 0) and a *target* value (score 100).  The category score is the
    mean of its metrics' scores.  The overall score is the weighted sum of
    category scores.

    Usage:
        matrix = EvaluationMatrix()
        report = matrix.evaluate({
            'nominal': [metrics_dict, ...],
            'motion':  [metrics_dict, ...],
            ...
        })
    """

    def evaluate(self, results: dict[str, list[dict]]) -> dict:
        """Evaluate all categories.

        Args:
            results: ``{category_name: [per_run_metrics, ...]}``.
                Per-run metrics dicts are the output of
                ``MetricsCalculator.compute()``.

        Returns:
            ``{category_name: score, 'overall': weighted_score}``
            where each category score is in [0, 100] and the overall score
            is the weight-sum across categories.
        """
        category_scores: dict[str, float] = {}

        for cat_name, cat_cfg in CATEGORIES.items():
            runs = results.get(cat_name, [])
            if not runs:
                category_scores[cat_name] = 0.0
                continue

            # Aggregate across runs: mean of each metric
            agg = _aggregate_runs(runs)

            # Score each metric referenced by this category
            metric_scores: list[float] = []
            for metric_name in cat_cfg['metrics']:
                if metric_name not in agg:
                    continue
                mean_val = agg[metric_name]['mean']
                scoring = _METRIC_SCORING.get(metric_name)
                if scoring is None:
                    continue
                metric_scores.append(_score_metric(
                    mean_val,
                    scoring['target'],
                    scoring['baseline'],
                    scoring['higher_better'],
                ))

            category_scores[cat_name] = (
                float(np.mean(metric_scores)) if metric_scores else 0.0
            )

        # Overall weighted score
        overall = sum(
            category_scores.get(cat, 0.0) * cat_cfg['weight']
            for cat, cat_cfg in CATEGORIES.items()
        )

        result = dict(category_scores)
        result['overall'] = overall
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _aggregate_runs(runs: list[dict]) -> dict[str, dict]:
    """Aggregate multiple per-run metric dicts into per-metric stats."""
    agg: dict[str, dict] = {}
    numeric_keys = [
        'acquisition_time_s', 'mean_pixel_error', 'steady_state_error_px',
        'max_pixel_error', 'lock_retention_pct', 'detection_rate_pct',
        'rms_angular_error_deg', 'loss_events', 'reacquisition_time_s',
        'mean_fps', 'mean_processing_time_ms',
    ]
    for key in numeric_keys:
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    return agg


def _score_metric(value: float, target: float, baseline: float,
                  higher_better: bool) -> float:
    """Score a metric on a 0--100 scale via linear interpolation.

    - ``higher_better=True``:  score = (value - baseline) / (target - baseline) * 100
    - ``higher_better=False``: score = (baseline - value) / (baseline - target) * 100
    """
    if target == baseline:
        return 100.0 if value == target else 0.0

    if higher_better:
        score = (value - baseline) / (target - baseline) * 100.0
    else:
        score = (baseline - value) / (baseline - target) * 100.0

    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Convenience: run the full evaluation matrix
# ---------------------------------------------------------------------------

def run_full_evaluation(
    seeds: Optional[list[int]] = None,
    duration_s: int = 10,
    output_dir: str = 'results',
    quiet: bool = True,
    categories: Optional[list[str]] = None,
) -> dict:
    """Run every evaluation category, compute the matrix, and return results."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if categories is None:
        categories = list(CATEGORIES.keys())

    runner = ScenarioRunner(output_dir=output_dir)
    matrix = EvaluationMatrix()

    all_results: dict[str, list[dict]] = {}
    for cat_name in categories:
        print(f"  Running category: {cat_name}")
        result = runner.run_scenario(cat_name, seeds,
                                     duration_s=duration_s, quiet=quiet)
        all_results[cat_name] = result['metrics']

        agg = result['aggregated']
        for metric in ['mean_pixel_error', 'lock_retention_pct',
                       'detection_rate_pct']:
            if metric in agg:
                print(f"    {metric}: {agg[metric]['mean']:.2f} "
                      f"(std={agg[metric]['std']:.2f})")
        print()

    scores = matrix.evaluate(all_results)
    print(f"\n  Overall Score: {scores['overall']:.1f} / 100\n")
    return scores


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('evaluation.py -- Self-test')
    print('=' * 60)
    report = run_full_evaluation(
        seeds=[42, 123], duration_s=5,
        output_dir='results', quiet=True,
        categories=['nominal'],
    )
    print('=' * 60)
    print('evaluation.py self-test complete.')
    print('=' * 60)
