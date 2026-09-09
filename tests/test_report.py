"""Tests for the report module: SummaryReporter and ReportGenerator."""

import csv
import json
import os
import sys
import tempfile

import pytest

from report import SummaryReporter, ReportGenerator


# ---------------------------------------------------------------------------
# Helpers: minimal CSV fixture
# ---------------------------------------------------------------------------

CSV_HEADER = [
    'run_id', 'timestamp', 'frame_id', 'fps',
    'target_id',
    'gt_x', 'gt_y', 'est_x', 'est_y',
    'error_px', 'angular_error_deg',
    'detection_confidence', 'detected',
    'tracker_state', 'pan_cmd', 'tilt_cmd',
    'pan_actual', 'tilt_actual',
    'processing_time_ms',
]


def _make_row(frame_id: int, *, state='TRACKING', error=10.0, detected=1,
              gt_x=640.0, gt_y=360.0, proc_ms=5.0) -> dict:
    """Build one CSV row dict with sensible defaults."""
    t = frame_id / 30.0
    return {
        'run_id': 'test_run_001',
        'timestamp': f'{t:.4f}',
        'frame_id': str(frame_id),
        'fps': '30.0',
        'target_id': 'beacon_01',
        'gt_x': f'{gt_x:.2f}',
        'gt_y': f'{gt_y:.2f}',
        'est_x': f'{gt_x - error * 0.3:.2f}',
        'est_y': f'{gt_y - error * 0.4:.2f}',
        'error_px': f'{error:.2f}',
        'angular_error_deg': f'{error * 0.01:.4f}',
        'detection_confidence': '0.9000',
        'detected': str(detected),
        'tracker_state': state,
        'pan_cmd': '0.5000',
        'tilt_cmd': '0.3000',
        'pan_actual': '0.5100',
        'tilt_actual': '0.3100',
        'processing_time_ms': f'{proc_ms:.2f}',
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write a CSV file from a list of row dicts."""
    with open(path, 'w', newline='') as f:
        f.write(f"# run_id={rows[0]['run_id']}\n")
        f.write(f"# seed=42\n")
        f.write(f"# timestamp=2025-01-01T00:00:00\n")
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal 30-frame CSV covering SEARCHING -> TRACKING -> REACQUIRING -> TRACKING."""
    frames = []
    for i in range(30):
        if i < 5:
            state = 'SEARCHING'
            error = 300.0
            detected = 0
        elif i < 10:
            state = 'ACQUIRING'
            error = 150.0
            detected = 1
        elif i < 15:
            state = 'TRACKING'
            error = 10.0
            detected = 1
        elif i < 20:
            state = 'REACQUIRING'
            error = 200.0
            detected = 0
        else:
            state = 'TRACKING'
            error = 8.0
            detected = 1
        frames.append(_make_row(i, state=state, error=error, detected=detected))

    csv_path = str(tmp_path / "test_run_001.csv")
    _write_csv(csv_path, frames)
    return csv_path


# ---------------------------------------------------------------------------
# SummaryReporter tests
# ---------------------------------------------------------------------------

class TestSummaryReporter:
    """SummaryReporter should produce a valid JSON summary with all PRD fields."""

    def test_returns_all_required_fields(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "output"))

        for field in SummaryReporter.REQUIRED_FIELDS:
            assert field in summary, f"Missing required field: {field}"

    def test_json_is_written(self, sample_csv, tmp_path):
        output_dir = str(tmp_path / "output")
        reporter = SummaryReporter()
        reporter.generate(sample_csv, output_dir)

        expected_json = os.path.join(output_dir, "test_run_001_summary.json")
        assert os.path.isfile(expected_json), f"JSON not created at {expected_json}"

        with open(expected_json, 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_simulation_duration(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        # 30 frames at 30 FPS = 1.0s
        assert summary['simulation_duration_s'] == pytest.approx(1.0, abs=0.01)

    def test_num_frames(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        assert summary['num_frames'] == 30

    def test_fps_stats(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        # All frames logged as 30 FPS
        assert summary['mean_fps'] == pytest.approx(30.0, abs=0.01)
        assert summary['min_fps'] == pytest.approx(30.0, abs=0.01)
        assert summary['max_fps'] == pytest.approx(30.0, abs=0.01)

    def test_acquisition_time(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        # First TRACKING at frame 10 -> 10/30 = 0.333s
        assert summary['acquisition_time_s'] == pytest.approx(10 / 30, abs=0.01)

    def test_lock_retention(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        # TRACKING frames: 10-14 and 20-29 = 15 frames out of 30 = 50%
        assert summary['lock_retention_pct'] == pytest.approx(50.0, abs=0.1)

    def test_loss_events(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        # One loss: TRACKING (frame 14) -> REACQUIRING (frame 15)
        assert summary['loss_events'] >= 1

    def test_run_id_in_summary(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        assert summary['run_id'] == 'test_run_001'

    def test_pixel_error_values(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        assert summary['mean_pixel_error'] > 0
        assert summary['max_pixel_error'] >= summary['mean_pixel_error']

    def test_detection_rates(self, sample_csv, tmp_path):
        reporter = SummaryReporter()
        summary = reporter.generate(sample_csv, str(tmp_path / "out"))
        assert 0.0 <= summary['detection_precision'] <= 1.0
        assert 0.0 <= summary['detection_recall'] <= 1.0

    def test_json_validity(self, sample_csv, tmp_path):
        """The saved JSON must round-trip through loads."""
        output_dir = str(tmp_path / "output")
        reporter = SummaryReporter()
        reporter.generate(sample_csv, output_dir)

        json_path = os.path.join(output_dir, "test_run_001_summary.json")
        with open(json_path, 'r') as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        # Ensure it has at least the key summary fields
        assert 'simulation_duration_s' in loaded
        assert 'loss_events' in loaded

    def test_empty_csv_raises(self, tmp_path):
        """An empty CSV should raise ValueError."""
        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, 'w') as f:
            f.write("# run_id=empty\n")
            f.write("# seed=42\n")
            f.write("# timestamp=2025-01-01\n")
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            # No data rows

        reporter = SummaryReporter()
        with pytest.raises(ValueError, match="No data rows"):
            reporter.generate(csv_path, str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    """ReportGenerator should produce valid HTML from summary data."""

    @pytest.fixture
    def sample_summary(self):
        return {
            'run_id': 'test_report_001',
            'simulation_duration_s': 1.0,
            'num_frames': 30,
            'mean_fps': 30.0,
            'min_fps': 28.0,
            'max_fps': 32.0,
            'acquisition_time_s': 0.33,
            'detection_precision': 0.95,
            'detection_recall': 0.80,
            'mean_pixel_error': 12.5,
            'max_pixel_error': 180.0,
            'steady_state_error_px': 5.0,
            'rms_angular_error_deg': 0.15,
            'lock_retention_pct': 85.0,
            'loss_events': 2,
            'mean_loss_duration_s': 0.25,
            'reacquisition_time_s': 0.50,
            'time_in_center_pct': 60.0,
            'controller_saturation_pct': 3.5,
            'mean_processing_time_ms': 8.0,
            'max_processing_time_ms': 25.0,
            'config': {'fps': 30, 'duration_s': 60, 'kp': 1.0},
        }

    def test_html_file_created(self, sample_summary, tmp_path):
        csv_path = str(tmp_path / "test_report_001.csv")
        # Create a dummy CSV so the plot lookup doesn't crash
        _write_csv(csv_path, [_make_row(0)])

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, sample_summary, output_dir)
        assert os.path.isfile(path)
        assert path.endswith(".html")

    def test_html_is_valid_structure(self, sample_summary, tmp_path):
        csv_path = str(tmp_path / "dummy.csv")
        _write_csv(csv_path, [_make_row(0)])

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, sample_summary, output_dir)

        with open(path, 'r') as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        assert "test_report_001" in content

    def test_html_contains_metrics(self, sample_summary, tmp_path):
        csv_path = str(tmp_path / "dummy.csv")
        _write_csv(csv_path, [_make_row(0)])

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, sample_summary, output_dir)

        with open(path, 'r') as f:
            content = f.read()

        # Verify key metrics appear in the HTML
        assert "Mean FPS" in content
        assert "30.0" in content
        assert "85.0%" in content  # lock retention
        assert "Loss Events" in content

    def test_html_contains_config(self, sample_summary, tmp_path):
        csv_path = str(tmp_path / "dummy.csv")
        _write_csv(csv_path, [_make_row(0)])

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, sample_summary, output_dir)

        with open(path, 'r') as f:
            content = f.read()

        assert "Configuration" in content
        assert "kp" in content

    def test_html_contains_limitations(self, sample_summary, tmp_path):
        csv_path = str(tmp_path / "dummy.csv")
        _write_csv(csv_path, [_make_row(0)])

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, sample_summary, output_dir)

        with open(path, 'r') as f:
            content = f.read()

        assert "Known Limitations" in content
        assert "Single-target" in content

    def test_html_with_no_config(self, tmp_path):
        csv_path = str(tmp_path / "dummy.csv")
        _write_csv(csv_path, [_make_row(0)])

        summary = {
            'run_id': 'no_config',
            'simulation_duration_s': 1.0,
            'num_frames': 30,
            'mean_fps': 30.0,
            'min_fps': 30.0,
            'max_fps': 30.0,
            'mean_pixel_error': 5.0,
            'max_pixel_error': 10.0,
            'steady_state_error_px': 3.0,
            'rms_angular_error_deg': 0.05,
            'lock_retention_pct': 90.0,
            'loss_events': 0,
            'mean_loss_duration_s': 0.0,
            'mean_processing_time_ms': 5.0,
            'max_processing_time_ms': 10.0,
            'detection_precision': 1.0,
            'detection_recall': 1.0,
            'time_in_center_pct': 80.0,
            'controller_saturation_pct': 0.0,
        }

        output_dir = str(tmp_path / "output")
        path = ReportGenerator.generate_html(csv_path, summary, output_dir)
        assert os.path.isfile(path)
