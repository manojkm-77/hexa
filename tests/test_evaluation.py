"""
test_evaluation.py -- Tests for the evaluation framework.

Covers:
  1. MetricsCalculator.compute() with a synthetic CSV
  2. MetricsCalculator edge cases (empty CSV, reacquisition time)
  3. EvaluationMatrix scoring with synthetic metrics
  4. EvaluationMatrix with missing / partial data
  5. Weights sum to 1.0
  6. All category metrics have scoring definitions
"""

import csv
import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import (
    MetricsCalculator,
    EvaluationMatrix,
    CATEGORIES,
    _METRIC_SCORING,
    _aggregate_runs,
    _score_metric,
)


# ---------------------------------------------------------------------------
# Helper: write a synthetic CSV log that mimics CSVLogger output
# ---------------------------------------------------------------------------

def _write_sample_csv(filepath: str, num_frames: int = 100,
                      tracking_start: int = 5, fps: float = 30.0,
                      loss_events: list[tuple[int, int]] | None = None):
    """Write a minimal CSV log for testing MetricsCalculator.

    Args:
        filepath:          Output path.
        num_frames:        Number of data rows to write.
        tracking_start:    Frame at which TRACKING first appears.
        fps:               Nominal FPS (written into the CSV).
        loss_events:       Optional list of (start, end) frame pairs that
                           will be set to REACQUIRING.  Default: one loss
                           event at frames 71--72.
    """
    if loss_events is None:
        loss_events = [(71, 72)]

    fields = [
        'run_id', 'timestamp', 'frame_id', 'fps',
        'target_id',
        'gt_x', 'gt_y', 'est_x', 'est_y',
        'error_px', 'angular_error_deg',
        'detection_confidence', 'detected',
        'tracker_state', 'pan_cmd', 'tilt_cmd',
        'pan_actual', 'tilt_actual',
        'processing_time_ms',
    ]

    # Build a quick lookup set for loss frames
    loss_set: set[int] = set()
    for start, end in loss_events:
        for f in range(start, end + 1):
            loss_set.add(f)

    with open(filepath, 'w', newline='') as f:
        f.write('# run_id=test_run\n')
        f.write('# seed=42\n')
        f.write('# timestamp=2026-01-01T00:00:00\n')
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(num_frames):
            # Determine state
            if i < tracking_start:
                state = 'SEARCHING'
                detected = 0
                conf = 0.0
            elif i in loss_set:
                state = 'REACQUIRING'
                detected = 0
                conf = 0.0
            else:
                state = 'TRACKING'
                detected = 1
                conf = 0.9

            # Error: high during SEARCHING / REACQUIRING, low during TRACKING
            if state == 'SEARCHING':
                error = 200.0
            elif state == 'REACQUIRING':
                error = 80.0
            else:
                error = 10.0 + 2.0 * np.sin(i * 0.1)

            angular_err = error * 0.047
            gt_x = 640.0 + error * 0.7
            gt_y = 360.0 + error * 0.7
            est_x = gt_x + np.random.randn() * 2
            est_y = gt_y + np.random.randn() * 2

            row = {
                'run_id': 'test_run',
                'timestamp': f'{i / fps:.4f}',
                'frame_id': str(i),
                'fps': f'{fps:.1f}',
                'target_id': 'beacon_01',
                'gt_x': f'{gt_x:.2f}',
                'gt_y': f'{gt_y:.2f}',
                'est_x': f'{est_x:.2f}',
                'est_y': f'{est_y:.2f}',
                'error_px': f'{error:.2f}',
                'angular_error_deg': f'{angular_err:.4f}',
                'detection_confidence': f'{conf:.4f}',
                'detected': str(detected),
                'tracker_state': state,
                'pan_cmd': '0.0000',
                'tilt_cmd': '0.0000',
                'pan_actual': '0.0000',
                'tilt_actual': '0.0000',
                'processing_time_ms': f'{15.0 + np.random.randn() * 3:.2f}',
            }
            writer.writerow(row)


# ===================================================================
# MetricsCalculator tests
# ===================================================================

class TestMetricsCalculator:
    """MetricsCalculator.compute() with synthetic CSV data."""

    def test_returns_all_expected_keys(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        expected = {
            'acquisition_time_s', 'mean_pixel_error', 'steady_state_error_px',
            'max_pixel_error', 'lock_retention_pct', 'detection_rate_pct',
            'rms_angular_error_deg', 'loss_events', 'reacquisition_time_s',
            'mean_fps', 'mean_processing_time_ms', 'num_frames', 'duration_s',
        }
        assert expected.issubset(set(metrics.keys())), \
            f"Missing keys: {expected - set(metrics.keys())}"

    def test_acquisition_time(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        tracking_start = 5
        _write_sample_csv(csv_path, num_frames=100, tracking_start=tracking_start)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        expected_time = tracking_start / 30.0
        assert metrics['acquisition_time_s'] is not None
        assert abs(metrics['acquisition_time_s'] - expected_time) < 0.01

    def test_lock_retention(self, tmp_path):
        """95 of 100 frames are TRACKING (1 SEARCHING start + 2 REACQUIRING)."""
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        # SEARCHING: frames 0-4 = 5 frames
        # REACQUIRING: frames 71-72 = 2 frames
        # TRACKING: 100 - 5 - 2 = 93 frames
        assert metrics['lock_retention_pct'] == pytest.approx(93.0, abs=0.1)

    def test_loss_events(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['loss_events'] == 1

    def test_detection_rate(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        # detected=1 for frames 5-70 and 73-99 = 93 frames
        # detected=1 for frame 70, 0 for 71-72
        # So: 5-70 = 66 frames, 73-99 = 27 frames, total = 93
        assert metrics['detection_rate_pct'] == pytest.approx(93.0, abs=0.1)

    def test_steady_state_error(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['steady_state_error_px'] < 20.0
        assert metrics['steady_state_error_px'] > 0.0

    def test_max_pixel_error(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['max_pixel_error'] >= 200.0

    def test_rms_angular_error_positive(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['rms_angular_error_deg'] > 0.0

    def test_reacquisition_time_computed(self, tmp_path):
        """With one loss event, reacquisition time should be a short float."""
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5,
                          loss_events=[(71, 72)])
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['reacquisition_time_s'] is not None
        # 2 frames / 30 fps = ~0.067s
        assert abs(metrics['reacquisition_time_s'] - 2 / 30.0) < 0.01

    def test_multiple_loss_events(self, tmp_path):
        """Two distinct loss events should yield loss_events == 2."""
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=100, tracking_start=5,
                          loss_events=[(20, 22), (71, 73)])
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        assert metrics['loss_events'] == 2

    def test_empty_csv(self, tmp_path):
        csv_path = str(tmp_path / 'empty.csv')
        with open(csv_path, 'w') as f:
            f.write('# comment\n')
            f.write('run_id,timestamp,frame_id,fps,target_id,'
                    'gt_x,gt_y,est_x,est_y,error_px,angular_error_deg,'
                    'detection_confidence,detected,tracker_state,'
                    'pan_cmd,tilt_cmd,pan_actual,tilt_actual,'
                    'processing_time_ms\n')
        calc = MetricsCalculator()
        metrics = calc.compute(csv_path)

        assert metrics['num_frames'] == 0
        assert metrics['mean_pixel_error'] == 0.0

    def test_frame_count_and_duration(self, tmp_path):
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=50, tracking_start=3)
        calc = MetricsCalculator()
        metrics = calc.compute(csv_path)

        assert metrics['num_frames'] == 50
        assert abs(metrics['duration_s'] - 50 / 30.0) < 0.01


# ===================================================================
# EvaluationMatrix tests
# ===================================================================

class TestEvaluationMatrix:
    """EvaluationMatrix scoring with synthetic metric data."""

    def test_perfect_score(self):
        """Score should be 100 when value equals the target."""
        score = _score_metric(value=10.0, target=10.0, baseline=100.0,
                              higher_better=False)
        assert score == pytest.approx(100.0)

    def test_zero_score(self):
        """Score should be 0 when value equals the baseline."""
        score = _score_metric(value=100.0, target=10.0, baseline=100.0,
                              higher_better=False)
        assert score == pytest.approx(0.0)

    def test_clamped_above_target(self):
        """Better-than-target values should clamp to 100."""
        score = _score_metric(value=5.0, target=10.0, baseline=100.0,
                              higher_better=False)
        assert score == 100.0

    def test_clamped_below_baseline(self):
        """Worse-than-baseline values should clamp to 0."""
        score = _score_metric(value=150.0, target=10.0, baseline=100.0,
                              higher_better=False)
        assert score == 0.0

    def test_higher_better_increases(self):
        """Score should increase with value when higher_better=True."""
        s1 = _score_metric(20.0, target=30.0, baseline=15.0,
                           higher_better=True)
        s2 = _score_metric(25.0, target=30.0, baseline=15.0,
                           higher_better=True)
        assert s2 > s1

    def test_lower_better_increases(self):
        """Score should increase as value decreases when higher_better=False."""
        s1 = _score_metric(50.0, target=10.0, baseline=100.0,
                           higher_better=False)
        s2 = _score_metric(30.0, target=10.0, baseline=100.0,
                           higher_better=False)
        assert s2 > s1

    def test_evaluate_with_real_csv(self, tmp_path):
        """Feed MetricsCalculator output into EvaluationMatrix for scoring."""
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=60, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(csv_path)

        matrix = EvaluationMatrix()
        report = matrix.evaluate({'nominal': [metrics]})

        assert 0 <= report['nominal'] <= 100
        assert 0 <= report['overall'] <= 100

    def test_evaluate_empty_results(self):
        """Evaluating with no data should return 0 scores."""
        matrix = EvaluationMatrix()
        report = matrix.evaluate({})

        for cat_name in CATEGORIES:
            assert report[cat_name] == 0.0
        assert report['overall'] == 0.0

    def test_evaluate_missing_category(self):
        """Missing categories should get zero scores."""
        matrix = EvaluationMatrix()
        report = matrix.evaluate({'nominal': []})

        assert report['nominal'] == 0.0

    def test_evaluate_multiple_runs(self, tmp_path):
        """Multiple runs should aggregate correctly and yield valid scores."""
        csv_path = str(tmp_path / 'test.csv')
        _write_sample_csv(csv_path, num_frames=60, tracking_start=5)
        calc = MetricsCalculator(fps=30)
        m1 = calc.compute(csv_path)
        m2 = calc.compute(csv_path)

        matrix = EvaluationMatrix()
        report = matrix.evaluate({'nominal': [m1, m2]})

        assert 0 <= report['overall'] <= 100

    def test_all_categories_scoring_present(self):
        """Every metric referenced in CATEGORIES has a scoring definition."""
        for cat_name, cat_cfg in CATEGORIES.items():
            for metric_name in cat_cfg['metrics']:
                assert metric_name in _METRIC_SCORING, \
                    f"Metric '{metric_name}' in category '{cat_name}' " \
                    f"has no scoring definition"


# ===================================================================
# Weights and configuration tests
# ===================================================================

class TestWeightsAndConfig:
    """Verify structural invariants in the evaluation framework."""

    def test_category_weights_sum_to_one(self):
        total = sum(cat_cfg['weight'] for cat_cfg in CATEGORIES.values())
        assert abs(total - 1.0) < 1e-6, \
            f"Category weights sum to {total}, expected 1.0"

    def test_exactly_eight_categories(self):
        assert len(CATEGORIES) == 8

    def test_all_weights_positive(self):
        for name, cat_cfg in CATEGORIES.items():
            assert cat_cfg['weight'] > 0, \
                f"Weight for '{name}' is not positive: {cat_cfg['weight']}"

    def test_all_categories_have_metrics(self):
        for name, cat_cfg in CATEGORIES.items():
            assert len(cat_cfg['metrics']) > 0, \
                f"Category '{name}' has no metrics"

    def test_all_category_metrics_exist_in_scoring(self):
        for cat_name, cat_cfg in CATEGORIES.items():
            for metric_name in cat_cfg['metrics']:
                assert metric_name in _METRIC_SCORING, \
                    f"Category '{cat_name}' references metric '{metric_name}' " \
                    f"which is not defined in _METRIC_SCORING"


# Keep _METRIC_SCORING accessible for tests
from evaluation import _METRIC_SCORING
