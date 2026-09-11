"""
report_dialog.py — Report export dialog for the FSOC simulator GUI.

A modal QDialog that lets the user:
  1. Choose an output directory.
  2. Select which artefacts to export (CSV, JSON, HTML, PNG plot).
  3. Click Export to produce the files.

If a CSV log from a previous simulation run exists in the output directory,
the dialog uses it; otherwise it exports a configuration-only summary.

Dependencies: PySide6, standard library (csv, json, shutil)
"""

import sys
import os
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QCheckBox,
        QFileDialog, QTextEdit, QGroupBox, QMessageBox,
        QSizePolicy, QComboBox,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

from config import Config


# =========================================================================
#  ReportDialog
# =========================================================================

class ReportDialog(QDialog):
    """Modal dialog for exporting simulation reports and artefacts.

    Parameters
    ----------
    config : Config
        Current simulation configuration (used for the summary if no CSV).
    scenario_name : str
        Name of the last-run scenario (for default filenames).
    csv_path : str | None
        Path to an existing CSV log, if a simulation has been completed.
        When *None* the dialog exports a config-only summary.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(self, config: Config, scenario_name: str = "clean",
                 csv_path: str = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.scenario_name = scenario_name
        self.csv_path = csv_path
        self.output_dir = os.path.join(_SRC, "..", "results")

        self.setWindowTitle("Export Report")
        self.setMinimumWidth(520)
        self.setMinimumHeight(440)
        self.setModal(True)

        self._build_ui()

    # ===================================================================
    #  UI
    # ===================================================================

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Header ─────────────────────────────────────────────────────
        hdr = QLabel("Export Simulation Report")
        font = hdr.font()
        font.setPointSize(13)
        font.setBold(True)
        hdr.setFont(font)
        root.addWidget(hdr)

        info = QLabel(
            "Select the output directory and the artefacts to generate. "
            "If a CSV log exists it will be included automatically."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # ── Output directory ────────────────────────────────────────────
        dir_grp = QGroupBox("Output Directory")
        dir_lay = QHBoxLayout(dir_grp)

        self._dir_edit = QLineEdit(self.output_dir)
        self._dir_edit.setPlaceholderText("Choose output directory...")
        dir_lay.addWidget(self._dir_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_dir)
        dir_lay.addWidget(browse_btn)

        root.addWidget(dir_grp)

        # ── Artefacts ──────────────────────────────────────────────────
        art_grp = QGroupBox("Artefacts to Export")
        art_lay = QGridLayout(art_grp)

        self._cb_csv = QCheckBox("Per-frame CSV log")
        self._cb_csv.setChecked(True)
        self._cb_csv.setToolTip(
            "Copy the existing CSV log to the output directory.")
        art_lay.addWidget(self._cb_csv, 0, 0)

        self._cb_json = QCheckBox("JSON summary")
        self._cb_json.setChecked(True)
        self._cb_json.setToolTip(
            "Generate a structured JSON summary with all PRD metrics.")
        art_lay.addWidget(self._cb_json, 0, 1)

        self._cb_html = QCheckBox("HTML report")
        self._cb_html.setChecked(True)
        self._cb_html.setToolTip(
            "Generate a self-contained HTML report with embedded plots.")
        art_lay.addWidget(self._cb_html, 1, 0)

        self._cb_png = QCheckBox("PNG plot")
        self._cb_png.setChecked(True)
        self._cb_png.setToolTip(
            "Generate the tracking-error-over-time PNG plot.")
        art_lay.addWidget(self._cb_png, 1, 1)

        self._cb_pdf = QCheckBox("PDF report")
        self._cb_pdf.setChecked(False)
        self._cb_pdf.setToolTip(
            "Generate a multi-page PDF summary report using matplotlib.")
        art_lay.addWidget(self._cb_pdf, 2, 0)

        # If no CSV exists, disable the copy option
        if self.csv_path is None or not os.path.isfile(self.csv_path):
            self._cb_csv.setEnabled(False)
            self._cb_csv.setChecked(False)
            self._cb_csv.setToolTip(
                "No CSV log available — run a simulation first.")
            self._cb_pdf.setEnabled(False)
            self._cb_pdf.setChecked(False)
            self._cb_pdf.setToolTip(
                "No CSV log available — run a simulation first.")

        root.addWidget(art_grp)

        # ── Scenario label ─────────────────────────────────────────────
        root.addWidget(QLabel(
            f"Scenario: {self.scenario_name.upper()}   |   "
            f"FPS: {self.config.fps}   |   Duration: {self.config.duration_s}s"
        ))

        # ── Output log ─────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas, Courier New", 9))
        self._log.setMaximumHeight(120)
        self._log.setPlaceholderText("Export results will appear here...")
        root.addWidget(self._log)

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._export_btn = QPushButton("Export")
        self._export_btn.setFixedHeight(32)
        self._export_btn.setMinimumWidth(120)
        self._export_btn.clicked.connect(self._do_export)
        btn_row.addWidget(self._export_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(32)
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    # ===================================================================
    #  Slots
    # ===================================================================

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir)
        if d:
            self._dir_edit.setText(d)
            self.output_dir = d

    def _do_export(self):
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Error", "Please select an output "
                                "directory.")
            return

        os.makedirs(out_dir, exist_ok=True)
        self._log.clear()

        run_id = f"{self.scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exported = []

        # ── CSV ────────────────────────────────────────────────────────
        if self._cb_csv.isChecked() and self.csv_path and \
                os.path.isfile(self.csv_path):
            dst = os.path.join(out_dir, f"{run_id}.csv")
            try:
                shutil.copy2(self.csv_path, dst)
                exported.append(f"CSV  ->  {dst}")
            except Exception as exc:
                exported.append(f"CSV  FAILED: {exc}")

        # ── JSON ───────────────────────────────────────────────────────
        if self._cb_json.isChecked():
            try:
                from report import SummaryReporter
                reporter = SummaryReporter()
                config_dict = {
                    "fps": self.config.fps,
                    "duration_s": self.config.duration_s,
                    "random_seed": self.config.random_seed,
                    "width": self.config.width,
                    "height": self.config.height,
                    "kp": self.config.kp,
                    "ki": self.config.ki,
                    "kd": self.config.kd,
                    "threshold": self.config.threshold,
                    "scenario": self.scenario_name,
                }
                if self.csv_path and os.path.isfile(self.csv_path):
                    summary = reporter.generate(
                        self.csv_path, out_dir, config=config_dict)
                    exported.append(f"JSON ->  {os.path.join(out_dir, f'{run_id}_summary.json')}")
                else:
                    # Config-only summary
                    summary = {
                        "scenario": self.scenario_name,
                        "config": config_dict,
                        "exported_at": datetime.now().isoformat(),
                        "note": "No CSV log — configuration summary only.",
                    }
                    json_path = os.path.join(out_dir,
                                             f"{run_id}_summary.json")
                    with open(json_path, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    exported.append(f"JSON ->  {json_path}")
            except Exception as exc:
                exported.append(f"JSON  FAILED: {exc}")

        # ── HTML ───────────────────────────────────────────────────────
        if self._cb_html.isChecked():
            try:
                from report import SummaryReporter, ReportGenerator
                reporter = SummaryReporter()
                config_dict = {
                    "fps": self.config.fps,
                    "duration_s": self.config.duration_s,
                    "random_seed": self.config.random_seed,
                    "width": self.config.width,
                    "height": self.config.height,
                    "kp": self.config.kp,
                    "ki": self.config.ki,
                    "kd": self.config.kd,
                    "threshold": self.config.threshold,
                    "scenario": self.scenario_name,
                }
                if self.csv_path and os.path.isfile(self.csv_path):
                    summary = reporter.generate(
                        self.csv_path, out_dir, config=config_dict)
                    html_path = ReportGenerator.generate_html(
                        self.csv_path, summary, out_dir)
                    exported.append(f"HTML ->  {html_path}")
                else:
                    exported.append(
                        "HTML  (skipped — no CSV log to report on)")
            except Exception as exc:
                exported.append(f"HTML  FAILED: {exc}")

        # ── PNG plot ───────────────────────────────────────────────────────
        if self._cb_png.isChecked():
            try:
                from main import generate_plots
                if self.csv_path and os.path.isfile(self.csv_path):
                    plot_path = os.path.join(out_dir,
                                             f"{run_id}_plot.png")
                    generate_plots(self.csv_path, plot_path,
                                   self.scenario_name)
                    exported.append(f"PNG   ->  {plot_path}")
                else:
                    exported.append(
                        "PNG   (skipped — no CSV log to plot)")
            except Exception as exc:
                exported.append(f"PNG   FAILED: {exc}")

        # ── PDF report ─────────────────────────────────────────────────────
        if self._cb_pdf.isChecked():
            try:
                if self.csv_path and os.path.isfile(self.csv_path):
                    pdf_path = os.path.join(out_dir,
                                            f"{run_id}_report.pdf")
                    self._generate_pdf(pdf_path)
                    exported.append(f"PDF   ->  {pdf_path}")
                else:
                    exported.append(
                        "PDF   (skipped — no CSV log to report on)")
            except Exception as exc:
                exported.append(f"PDF   FAILED: {exc}")

        # ── Report results ─────────────────────────────────────────────────
        if exported:
            self._log.setPlainText(
                "Export complete:\n\n" + "\n".join(exported))
        else:
            self._log.setPlainText(
                "Nothing to export. Run a simulation first, or "
                "select different artefacts.")

    # ===================================================================
    #  PDF generation
    # ===================================================================

    def _generate_pdf(self, pdf_path: str):
        """Generate a multi-page PDF report from the CSV log using matplotlib.

        Uses matplotlib's PdfPages backend so no extra dependencies are
        needed beyond matplotlib (which is already required).
        """
        import csv
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # Read CSV data
        rows = []
        with open(self.csv_path, 'r') as f:
            lines = [l for l in f if not l.startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            rows.append(row)

        if not rows:
            raise RuntimeError("CSV log is empty.")

        timestamps = [float(r.get('timestamp', 0)) for r in rows]
        errors = [float(r.get('error_px', 0)) for r in rows]
        confs = [float(r.get('detection_confidence', 0)) for r in rows]
        states = [r.get('tracker_state', '') for r in rows]
        ang_errors = [float(r.get('angular_error_deg', 0)) for r in rows]

        ts = np.array(timestamps)
        errs = np.array(errors)
        cfs = np.array(confs)
        ang = np.array(ang_errors)

        with PdfPages(pdf_path) as pdf:
            # ---- Page 1: Title + Summary table ----
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.92, 'FSOC Simulator Report',
                    ha='center', va='top', fontsize=20, fontweight='bold')
            ax.text(0.5, 0.87,
                    f'Scenario: {self.scenario_name.upper()}  |  '
                    f'FPS: {self.config.fps}  |  '
                    f'Duration: {self.config.duration_s}s',
                    ha='center', va='top', fontsize=11, color='#555')

            # Summary stats
            tracking_pct = sum(
                1 for s in states if s == 'TRACKING') / len(states) * 100
            mean_err = float(np.mean(errs))
            max_err = float(np.max(errs))
            rms_ang = float(np.sqrt(np.mean(ang ** 2)))
            mean_conf = float(np.mean(cfs))

            table_data = [
                ['Metric', 'Value'],
                ['Total Frames', str(len(rows))],
                ['Mean Error (px)', f'{mean_err:.2f}'],
                ['Max Error (px)', f'{max_err:.2f}'],
                ['RMS Angular Error (deg)', f'{rms_ang:.4f}'],
                ['Lock Retention (%)', f'{tracking_pct:.1f}'],
                ['Mean Confidence', f'{mean_conf:.3f}'],
            ]
            table = ax.table(cellText=table_data[1:],
                             colLabels=table_data[0],
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.8)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#2E86AB')
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f9f9f9' if row % 2 else '#e8e8e8')
            pdf.savefig(fig)
            plt.close(fig)

            # ---- Page 2: Tracking Error plot ----
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11),
                                            gridspec_kw={'hspace': 0.35})
            ax1.plot(ts, errs, color='#2E86AB', linewidth=0.8, label='Error')
            ax1.axhline(50, color='#4caf50', linestyle='--',
                        linewidth=1, label='50 px')
            ax1.axhline(128, color='#ff9800', linestyle='--',
                        linewidth=1, label='128 px')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pixel Error')
            ax1.set_title('Tracking Error Over Time')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.fill_between(ts, cfs, alpha=0.3, color='#2E86AB')
            ax2.plot(ts, cfs, color='#2E86AB', linewidth=0.8)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Detection Confidence Over Time')
            ax2.set_ylim(-0.05, 1.15)
            ax2.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close(fig)

            # ---- Page 3: Configuration ----
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, 'Configuration',
                    ha='center', va='top', fontsize=16, fontweight='bold')
            cfg_lines = [
                f'FPS: {self.config.fps}',
                f'Duration: {self.config.duration_s}s',
                f'Resolution: {self.config.width}x{self.config.height}',
                f'PID: Kp={self.config.kp}, Ki={self.config.ki}, '
                f'Kd={self.config.kd}',
                f'Deadband: {self.config.deadband_pixels} px',
                f'Threshold: {self.config.threshold}',
                f'Detector: {self.config.detector_type}',
                f'Filter: {self.config.filter_type}',
            ]
            ax.text(0.1, 0.85, '\n'.join(cfg_lines),
                    va='top', fontsize=10, family='monospace',
                    linespacing=1.8)
            pdf.savefig(fig)
            plt.close(fig)
