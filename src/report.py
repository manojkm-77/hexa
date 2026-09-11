"""
report.py — Structured JSON summary report and HTML report generation
for the FSOC coarse-alignment simulator.

Implements v1.1 reporting per PRD section 19:
  - SummaryReporter: reads CSV, computes all PRD metrics, outputs JSON
  - ReportGenerator: produces a self-contained HTML report

Usage:
    from report import SummaryReporter, ReportGenerator
    reporter = SummaryReporter()
    summary = reporter.generate(csv_path, output_dir, config=config)
    html_path = ReportGenerator.generate_html(csv_path, summary, output_dir)
"""

import csv
import json
import os
import base64
from typing import Optional


class SummaryReporter:
    """Reads a per-frame CSV log and produces a structured JSON summary
    containing all metrics defined in PRD section 19.2."""

    REQUIRED_FIELDS = [
        'simulation_duration_s', 'mean_fps', 'min_fps', 'max_fps',
        'acquisition_time_s',
        'detection_precision', 'detection_recall',
        'mean_pixel_error', 'max_pixel_error', 'steady_state_error_px',
        'rms_angular_error_deg',
        'lock_retention_pct',
        'loss_events', 'mean_loss_duration_s',
        'reacquisition_time_s',
        'time_in_center_pct',
        'controller_saturation_pct',
        'mean_processing_time_ms', 'max_processing_time_ms',
    ]

    def generate(self, csv_path: str, output_dir: str,
                 config: Optional[dict] = None) -> dict:
        """Read the CSV log and compute a structured summary.

        Args:
            csv_path: Path to the per-frame CSV log.
            output_dir: Directory to write the JSON summary into.
            config: Optional dictionary of simulation configuration values.

        Returns:
            Dictionary containing all PRD metrics.
        """
        rows = self._read_csv(csv_path)
        if not rows:
            raise ValueError(f"No data rows found in {csv_path}")

        summary = self._compute_metrics(rows)

        # Attach metadata
        run_id = rows[0].get('run_id', 'unknown')
        summary['run_id'] = run_id
        if config:
            summary['config'] = config

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{run_id}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)

        # Print human-readable summary
        self._print_summary(summary)

        return summary

    # ------------------------------------------------------------------
    # CSV reading
    # ------------------------------------------------------------------

    def _read_csv(self, csv_path: str) -> list[dict]:
        """Read CSV, skipping comment header lines."""
        rows = []
        with open(csv_path, 'r') as f:
            lines = [line for line in f if not line.startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, rows: list[dict]) -> dict:
        """Compute all PRD metrics from per-frame data."""
        n = len(rows)

        # --- Basic arrays ---
        fps_vals = [float(r['fps']) for r in rows]
        proc_times = [float(r['processing_time_ms']) for r in rows]
        errors_px = [float(r['error_px']) for r in rows]
        ang_errors = [float(r['angular_error_deg']) for r in rows]
        pan_cmds = [float(r['pan_cmd']) for r in rows]
        tilt_cmds = [float(r['tilt_cmd']) for r in rows]
        detected_flags = [int(r['detected']) for r in rows]
        states = [r['tracker_state'] for r in rows]

        fps = float(rows[0].get('fps', 30))  # nominal FPS for time conversion
        if fps <= 0:
            fps = 30.0  # fallback for frame-0 where FPS counter hasn't updated yet

        # --- Duration ---
        simulation_duration_s = n / fps

        # --- FPS stats ---
        mean_fps = float(_mean(fps_vals))
        min_fps = float(_min(fps_vals))
        max_fps = float(_max(fps_vals))

        # --- Processing time ---
        mean_processing_time_ms = float(_mean(proc_times))
        max_processing_time_ms = float(_max(proc_times))

        # --- Acquisition time: first frame entering TRACKING ---
        acquisition_frame = None
        for i, s in enumerate(states):
            if s == 'TRACKING':
                acquisition_frame = i
                break
        acquisition_time_s = acquisition_frame / fps if acquisition_frame is not None else None

        # --- Detection precision / recall ---
        # Precision: fraction of detected frames where detected == 1 and error < 128px
        # Recall: fraction of all frames that are detected (detected == 1)
        det_frame_count = 0
        true_pos_count = 0
        for i in range(n):
            if detected_flags[i] == 1:
                det_frame_count += 1
                if errors_px[i] < 128.0:
                    true_pos_count += 1
        detection_precision = (true_pos_count / det_frame_count) if det_frame_count > 0 else 0.0
        detection_recall = (det_frame_count / n) if n > 0 else 0.0

        # --- Pixel error stats ---
        mean_pixel_error = float(_mean(errors_px))
        max_pixel_error = float(_max(errors_px))

        # Steady-state error: mean error over the last 20% of frames
        ss_start = int(n * 0.8)
        ss_errors = errors_px[ss_start:] if ss_start < n else errors_px
        steady_state_error_px = float(_mean(ss_errors))

        # --- RMS angular error ---
        rms_angular_error_deg = float((_mean([e ** 2 for e in ang_errors])) ** 0.5)

        # --- Lock retention ---
        tracking_count = sum(1 for s in states if s == 'TRACKING')
        lock_retention_pct = (tracking_count / n * 100) if n > 0 else 0.0

        # --- Loss events and mean loss duration ---
        loss_events, total_loss_frames = self._count_loss_events(states)
        mean_loss_duration_s = (total_loss_frames / fps / loss_events) if loss_events > 0 else 0.0

        # --- Reacquisition time ---
        reacquisition_time_s = self._compute_reacquisition_time(states, fps)

        # --- Time in center (central 20%) ---
        center_x, center_y = 640.0, 360.0  # 1280x720 default resolution
        # Central 20% region: x in [256, 1024], y in [144, 576]
        center_tolerance_x = 1280 * 0.10  # 10% radius from center
        center_tolerance_y = 720 * 0.10
        in_center_count = 0
        for i in range(n):
            # Use ground truth relative to frame center
            try:
                gtx = float(rows[i]['gt_x'])
                gty = float(rows[i]['gt_y'])
                err_x = abs(gtx - center_x)
                err_y = abs(gty - center_y)
                if err_x <= center_tolerance_x and err_y <= center_tolerance_y:
                    in_center_count += 1
            except (KeyError, ValueError):
                pass
        time_in_center_pct = (in_center_count / n * 100) if n > 0 else 0.0

        # --- Controller saturation ---
        # Saturation = pan or tilt at rate limit; approximate via large deltas
        rate_limit = 60.0  # deg/s default
        sat_count = 0
        for i in range(1, n):
            dt_frame = 1.0 / fps
            dpan = abs(pan_cmds[i] - pan_cmds[i - 1])
            dtilt = abs(tilt_cmds[i] - tilt_cmds[i - 1])
            if dpan / dt_frame >= rate_limit - 0.01 or dtilt / dt_frame >= rate_limit - 0.01:
                sat_count += 1
        controller_saturation_pct = (sat_count / max(n - 1, 1)) * 100

        return {
            'simulation_duration_s': round(simulation_duration_s, 2),
            'num_frames': n,
            'mean_fps': round(mean_fps, 2),
            'min_fps': round(min_fps, 2),
            'max_fps': round(max_fps, 2),
            'acquisition_time_s': round(acquisition_time_s, 4) if acquisition_time_s is not None else None,
            'detection_precision': round(detection_precision, 4),
            'detection_recall': round(detection_recall, 4),
            'mean_pixel_error': round(mean_pixel_error, 2),
            'max_pixel_error': round(max_pixel_error, 2),
            'steady_state_error_px': round(steady_state_error_px, 2),
            'rms_angular_error_deg': round(rms_angular_error_deg, 6),
            'lock_retention_pct': round(lock_retention_pct, 2),
            'loss_events': loss_events,
            'mean_loss_duration_s': round(mean_loss_duration_s, 4),
            'reacquisition_time_s': round(reacquisition_time_s, 4) if reacquisition_time_s is not None else None,
            'time_in_center_pct': round(time_in_center_pct, 2),
            'controller_saturation_pct': round(controller_saturation_pct, 2),
            'mean_processing_time_ms': round(mean_processing_time_ms, 2),
            'max_processing_time_ms': round(max_processing_time_ms, 2),
        }

    # ------------------------------------------------------------------
    # Loss / reacquisition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_loss_events(states: list[str]) -> tuple[int, int]:
        """Count transitions from TRACKING to REACQUIRING and total lost frames."""
        losses = 0
        total_loss_frames = 0
        in_loss = False
        loss_start = 0

        for i, s in enumerate(states):
            if s == 'TRACKING':
                if in_loss:
                    total_loss_frames += (i - loss_start)
                    in_loss = False
            elif s == 'REACQUIRING':
                if not in_loss:
                    losses += 1
                    loss_start = i
                    in_loss = True

        # If still in loss at end
        if in_loss:
            total_loss_frames += (len(states) - loss_start)

        return losses, total_loss_frames

    @staticmethod
    def _compute_reacquisition_time(states: list[str], fps: float) -> Optional[float]:
        """Compute mean time to re-enter TRACKING after a loss event."""
        reacquire_times = []
        in_loss = False
        loss_start = 0

        for i, s in enumerate(states):
            if s in ('REACQUIRING', 'SEARCHING') and not in_loss:
                # Check if previous was TRACKING
                if i > 0 and states[i - 1] == 'TRACKING':
                    in_loss = True
                    loss_start = i
            elif s == 'TRACKING' and in_loss:
                duration_frames = i - loss_start
                reacquire_times.append(duration_frames / fps)
                in_loss = False

        if not reacquire_times:
            return None
        return _mean(reacquire_times)

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(summary: dict) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  SUMMARY REPORT: {summary.get('run_id', 'unknown')}")
        print(f"{'='*60}")
        print(f"  Duration:              {summary['simulation_duration_s']:.1f}s")
        print(f"  Frames:                {summary['num_frames']}")
        print(f"  Mean FPS:              {summary['mean_fps']:.1f}  "
              f"(min={summary['min_fps']:.1f}, max={summary['max_fps']:.1f})")
        print(f"  Processing time:       mean={summary['mean_processing_time_ms']:.2f}ms  "
              f"max={summary['max_processing_time_ms']:.2f}ms")
        print(f"  Acquisition time:      {_fmt_time(summary['acquisition_time_s'])}")
        print(f"  Detection precision:   {summary['detection_precision']:.4f}")
        print(f"  Detection recall:      {summary['detection_recall']:.4f}")
        print(f"  Mean pixel error:      {summary['mean_pixel_error']:.2f} px")
        print(f"  Max pixel error:       {summary['max_pixel_error']:.2f} px")
        print(f"  Steady-state error:    {summary['steady_state_error_px']:.2f} px")
        print(f"  RMS angular error:     {summary['rms_angular_error_deg']:.6f} deg")
        print(f"  Lock retention:        {summary['lock_retention_pct']:.1f}%")
        print(f"  Loss events:           {summary['loss_events']}")
        print(f"  Mean loss duration:    {summary['mean_loss_duration_s']:.4f}s")
        print(f"  Reacquisition time:    {_fmt_time(summary['reacquisition_time_s'])}")
        print(f"  Time in center:        {summary['time_in_center_pct']:.1f}%")
        print(f"  Controller saturation: {summary['controller_saturation_pct']:.2f}%")
        print(f"{'='*60}\n")


class ReportGenerator:
    """Generates a self-contained HTML report from CSV log and summary data."""

    @staticmethod
    def generate_html(csv_path: str, summary: dict, output_dir: str) -> str:
        """Create an HTML report file.

        Args:
            csv_path: Path to the original CSV log.
            summary: Summary dict from SummaryReporter.generate().
            output_dir: Directory to write the HTML file.

        Returns:
            Path to the generated HTML file.
        """
        os.makedirs(output_dir, exist_ok=True)
        run_id = summary.get('run_id', 'unknown')
        html_path = os.path.join(output_dir, f"{run_id}_report.html")

        # Try to embed the plot as base64 PNG
        plot_b64 = ReportGenerator._load_plot_base64(csv_path)

        config_section = ReportGenerator._render_config(summary.get('config', {}))
        metrics_table = ReportGenerator._render_metrics(summary)
        limitations = ReportGenerator._render_limitations()

        plot_section = ""
        if plot_b64:
            plot_section = f"""
        <div class="section">
            <h2>Plot</h2>
            <img src="data:image/png;base64,{plot_b64}" alt="Error over time"
                 style="max-width:100%; border:1px solid #444; border-radius:4px;" />
        </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FSOC Report — {run_id}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0;
         max-width: 960px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 8px; }}
  h2 {{ color: #2E86AB; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: left; }}
  th {{ background: #16213e; color: #2E86AB; }}
  tr:nth-child(even) {{ background: #16213e; }}
  .section {{ margin-bottom: 24px; }}
  code {{ background: #16213e; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
  <h1>FSOC Coarse-Alignment Simulator — Run Report</h1>
  <p><strong>Run ID:</strong> {run_id}</p>
  <p><strong>Generated:</strong> {_timestamp()}</p>

  <div class="section">
    <h2>Configuration</h2>
    {config_section}
  </div>

  <div class="section">
    <h2>Performance Metrics</h2>
    {metrics_table}
  </div>

  {plot_section}

  <div class="section">
    <h2>Known Limitations</h2>
    {limitations}
  </div>
</body>
</html>"""

        with open(html_path, 'w') as f:
            f.write(html)

        print(f"  HTML report saved to {html_path}")
        return html_path

    # ------------------------------------------------------------------
    # HTML sub-renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_config(config: dict) -> str:
        if not config:
            return "<p><em>No configuration data available.</em></p>"
        rows = "".join(
            f"      <tr><td><code>{k}</code></td><td>{v}</td></tr>\n"
            for k, v in sorted(config.items())
        )
        return f"""    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
{rows}    </table>"""

    @staticmethod
    def _render_metrics(summary: dict) -> str:
        """Render summary metrics as an HTML table."""
        # Map of (display_label, key, format_func)
        metric_rows = [
            ("Simulation Duration", "simulation_duration_s", lambda v: f"{v:.1f} s"),
            ("Total Frames", "num_frames", lambda v: f"{v}"),
            ("Mean FPS", "mean_fps", lambda v: f"{v:.1f}"),
            ("Min FPS", "min_fps", lambda v: f"{v:.1f}"),
            ("Max FPS", "max_fps", lambda v: f"{v:.1f}"),
            ("Acquisition Time", "acquisition_time_s", lambda v: _fmt_time(v)),
            ("Detection Precision", "detection_precision", lambda v: f"{v:.4f}"),
            ("Detection Recall", "detection_recall", lambda v: f"{v:.4f}"),
            ("Mean Pixel Error", "mean_pixel_error", lambda v: f"{v:.2f} px"),
            ("Max Pixel Error", "max_pixel_error", lambda v: f"{v:.2f} px"),
            ("Steady-State Error", "steady_state_error_px", lambda v: f"{v:.2f} px"),
            ("RMS Angular Error", "rms_angular_error_deg", lambda v: f"{v:.6f} deg"),
            ("Lock Retention", "lock_retention_pct", lambda v: f"{v:.1f}%"),
            ("Loss Events", "loss_events", lambda v: f"{v}"),
            ("Mean Loss Duration", "mean_loss_duration_s", lambda v: f"{v:.4f} s"),
            ("Reacquisition Time", "reacquisition_time_s", lambda v: _fmt_time(v)),
            ("Time in Center", "time_in_center_pct", lambda v: f"{v:.1f}%"),
            ("Controller Saturation", "controller_saturation_pct", lambda v: f"{v:.2f}%"),
            ("Mean Processing Time", "mean_processing_time_ms", lambda v: f"{v:.2f} ms"),
            ("Max Processing Time", "max_processing_time_ms", lambda v: f"{v:.2f} ms"),
        ]
        lines = ['    <table>\n      <tr><th>Metric</th><th>Value</th></tr>\n']
        for label, key, fmt in metric_rows:
            val = summary.get(key)
            display = fmt(val) if val is not None else "N/A"
            lines.append(f"      <tr><td>{label}</td><td>{display}</td></tr>\n")
        lines.append("    </table>")
        return "".join(lines)

    @staticmethod
    def _render_limitations() -> str:
        return """    <ul>
      <li>Camera model assumes ideal pinhole lens with no distortion.</li>
      <li>PID gains are static — no online gain adaptation.</li>
      <li>AI detector requires a pre-trained ONNX model (not included).</li>
      <li>Multi-target tracking uses nearest-neighbor association (no JPDA/MHT).</li>
      <li>Atmospheric turbulence model is a simplified Ornstein-Uhlenbeck process.</li>
      <li>No closed-loop adaptive optics or wavefront correction.</li>
    </ul>"""

    @staticmethod
    def _load_plot_base64(csv_path: str) -> Optional[str]:
        """Try to find and base64-encode the plot PNG next to the CSV."""
        csv_dir = os.path.dirname(csv_path)
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        plot_path = os.path.join(csv_dir, f"{csv_stem}_plot.png")
        if os.path.isfile(plot_path):
            with open(plot_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('ascii')
        return None


# ======================================================================
# Helpers
# ======================================================================

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _min(values: list[float]) -> float:
    if not values:
        return 0.0
    return min(values)


def _max(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values)


def _fmt_time(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f} s"


def _json_default(obj):
    """Handle types that json.dumps cannot serialize."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec='seconds')
