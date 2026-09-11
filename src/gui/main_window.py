"""
main_window.py — Main application window for the FSOC coarse-alignment
simulator's PySide6 GUI.

Provides a full graphical interface with live video feed, real-time status,
configuration controls, embedded plot, and report export.

Threading model:
  - GUI (main thread): MainWindow renders video, status, plots.
  - Simulation (worker QThread): SimulationWorker steps the sim, emits
    Qt signals that are queued to the main thread.

Dependencies: PySide6, numpy, cv2
"""

import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# Ensure src/ is on sys.path so we can import sibling modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── PySide6 (graceful fallback) ──────────────────────────────────────────
try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QToolBar, QSplitter, QTabWidget,
        QMessageBox, QSizePolicy, QGroupBox, QGridLayout,
        QFileDialog, QPushButton,
    )
    from PySide6.QtCore import (
        Qt, QThread, QObject, Signal, Slot, QTimer, QSize,
    )
    from PySide6.QtGui import (
        QImage, QPixmap, QKeySequence, QAction, QIcon,
        QPalette, QColor, QFont, QPainter, QShortcut,
    )

    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

# ── Simulator modules ────────────────────────────────────────────────────
from config import Config
from detect import TrackerState
from main import make_scenario, draw_hud, CSVLogger, generate_plots
from report import SummaryReporter, ReportGenerator


# =========================================================================
#  SimulationWorker  —  runs in a QThread
# =========================================================================

class SimulationWorker(QObject):
    """Runs the simulation loop in a background thread, emitting Qt signals
    that are queued to the main (GUI) thread.

    Signals
    -------
    frame_ready(QImage, float)
        A processed frame (BGR-converted, HUD-drawn, optionally paused-overlayed)
        and the current FPS.
    status_updated(dict)
        Snapshot of key status values for the status panel.
    plot_updated(float, float, float)
        (timestamp, pixel_error, confidence) for the live plot.
    paused_state(bool)
        Emitted when the worker enters or leaves a paused state.
    simulation_done(dict)
        Summary statistics when the run completes.
    error_occurred(str)
        Error message if something goes wrong.
    """

    frame_ready = Signal(object, float)
    status_updated = Signal(object)
    plot_updated = Signal(float, float, float)
    paused_state = Signal(bool)
    simulation_done = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, config: Config, scenario_name: str, parent=None):
        super().__init__(parent)
        self.config = config
        self.scenario_name = scenario_name
        self._paused = False
        self._stop_requested = False
        self.output_dir = os.path.join(_SRC_DIR, "..", "results")
        self.save_frames = False
        self.dev_mode = False
        self.csv_path = None

    # ── Public slots (connected from main thread) ──────────────────────

    @Slot(bool)
    def set_dev_mode(self, enabled: bool):
        self.dev_mode = enabled

    @Slot()
    def request_pause(self):
        self._paused = True
        self.paused_state.emit(True)

    @Slot()
    def request_resume(self):
        self._paused = False
        self.paused_state.emit(False)

    @Slot()
    def request_stop(self):
        self._stop_requested = True

    @Slot()
    def do_reset(self):
        """Stop-and-restart: this worker dies, a fresh one is spawned."""
        self._stop_requested = True

    # ── Main work method (called via invokeMethod) ─────────────────────

    def run(self):
        """Execute the full simulation loop.  Called once per worker lifetime."""
        try:
            self._run_simulation()
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _run_simulation(self):
        config = self.config
        scenario_name = self.scenario_name

        os.makedirs(self.output_dir, exist_ok=True)

        # ── Create scenario objects ────────────────────────────────────
        sim, controller, disturbances, multi_tracker = make_scenario(config, scenario_name)
        num_frames = int(config.duration_s * config.fps)

        run_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_path = os.path.join(self.output_dir, f"{run_id}.csv")
        plot_path = os.path.join(self.output_dir, f"{run_id}_plot.png")
        self.csv_path = csv_path
        os.makedirs(os.path.join(self.output_dir, f"{run_id}_frames"),
                    exist_ok=True)

        logger = CSVLogger(csv_path, run_id, config.random_seed)

        # ── Timing state ───────────────────────────────────────────────
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0
        plot_last_update = time.time()

        # ── Simulation loop ────────────────────────────────────────────
        for frame_id in range(num_frames):
            # --- Control messages ---
            if self._stop_requested:
                break

            if self._paused:
                time.sleep(0.02)          # yield 20 ms to keep GUI responsive
                continue

            t0 = time.time()

            # --- Step simulator ---
            frame, gt = sim.step(frame_id)
            gt_primary = gt[0] if isinstance(gt, (list, tuple)) else gt
            ts = gt_primary.timestamp if hasattr(gt_primary, 'timestamp') else (frame_id / config.fps)

            # --- v1.1 frame-level disturbances (after rendering) --------
            if disturbances.get("exposure"):
                frame = disturbances["exposure"].apply(frame, ts)
            if disturbances.get("occlusion"):
                frame = disturbances["occlusion"].apply(frame, ts)
            if disturbances.get("false_beacons"):
                frame = disturbances["false_beacons"].apply(frame, ts)

            # --- Tracker + controller ---
            if multi_tracker is not None:
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

                controller.tracker = multi_tracker.trackers[first_bid]
                from detect import TrackerOutput as _TO
                tracker_output = _TO(
                    state=first_tracklet.state,
                    estimated_x=first_tracklet.estimated_x,
                    estimated_y=first_tracklet.estimated_y,
                    confidence=first_tracklet.confidence,
                    detected=first_tracklet.detected,
                    consecutive_detections=first_tracklet.consecutive_detections,
                    consecutive_misses=first_tracklet.consecutive_misses,
                    tracked_id=first_bid,
                )
            else:
                cmd = controller.update(frame, sim.cam.pan_deg, sim.cam.tilt_deg)
                tracker_output = controller.last_track_output

            sim.set_camera_pose(cmd.pan_cmd_deg, cmd.tilt_cmd_deg)

            proc_ms = (time.time() - t0) * 1000.0

            # --- FPS ---
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            # --- Log frame to CSV ---
            logger.log(frame_id, ts, current_fps, gt_primary,
                       tracker_output, cmd, proc_ms)

            # --- Tracker status ---
            if tracker_output is not None:
                est_x, est_y = tracker_output.estimated_x, tracker_output.estimated_y
                det_active = tracker_output.detected
                conf = tracker_output.confidence
            else:
                est_x, est_y = 0.0, 0.0
                det_active = False
                conf = 0.0

            # --- Draw HUD on the frame (OpenCV) -------------------------
            display_frame = frame.copy()
            hud_state = tracker_output.state if tracker_output else controller.tracker.state
            draw_hud(
                display_frame, hud_state, cmd, gt,
                tracker_output, current_fps, proc_ms,
                self.dev_mode, config,
            )

            # --- BGR -> RGB -> QImage -----------------------------------
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bpl = ch * w                            # bytes per line
            qimg = QImage(rgb.data, w, h, bpl,
                          QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qimg, current_fps)

            # --- Status snapshot ----------------------------------------
            angular_err = np.sqrt(cmd.error_pan_deg ** 2
                                  + cmd.error_tilt_deg ** 2)
            self.status_updated.emit({
                "state": hud_state.value if hasattr(hud_state, 'value') else str(hud_state),
                "est_x": est_x,
                "est_y": est_y,
                "confidence": conf,
                "detected": det_active,
                "consecutive_detections": tracker_output.consecutive_detections if tracker_output else 0,
                "consecutive_misses": tracker_output.consecutive_misses if tracker_output else 0,
                "pan_cmd_deg": cmd.pan_cmd_deg,
                "tilt_cmd_deg": cmd.tilt_cmd_deg,
                "error_pixels": cmd.error_pixels,
                "error_pan_deg": cmd.error_pan_deg,
                "error_tilt_deg": cmd.error_tilt_deg,
                "angular_error_deg": angular_err,
                "saturated": cmd.saturated,
                "fps": current_fps,
                "processing_time_ms": proc_ms,
                "gt_visible": gt_primary.target_visible if hasattr(gt_primary, 'target_visible') else False,
            })

            # --- Plot data (throttled to ~10 Hz) ------------------------
            now = time.time()
            if now - plot_last_update >= 0.1:
                self.plot_updated.emit(ts,
                                       cmd.error_pixels, conf)
                plot_last_update = now

            # --- Save frame to disk (every 30th frame) ------------------
            if self.save_frames and frame_id % 30 == 0:
                frames_dir = os.path.join(self.output_dir,
                                          f"{run_id}_frames")
                os.makedirs(frames_dir, exist_ok=True)
                fpath = os.path.join(frames_dir,
                                     f"frame_{frame_id:05d}.jpg")
                cv2.imwrite(fpath, display_frame)

        # ── Post-run: reports & statistics ──────────────────────
        logger.close()
        stats = logger.summary()
        stats["_csv_path"] = csv_path  # include for report export
        stats["_plot_path"] = plot_path
        self.simulation_done.emit(stats)

        # Generate plot (reports are deferred to the Export dialog)
        try:
            generate_plots(csv_path, plot_path, scenario_name)
        except Exception:
            pass    # non-critical


# =========================================================================
#  MainWindow
# =========================================================================

class MainWindow(QMainWindow):
    """Main application window for the FSOC coarse-alignment simulator."""

    # Worker-process signals
    _worker_frame = Signal(object, float)
    _worker_status = Signal(object)
    _worker_plot = Signal(float, float, float)
    _worker_paused = Signal(bool)
    _worker_done = Signal(object)
    _worker_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSOC Coarse-Alignment Simulator")
        self.setMinimumSize(1280, 720)
        self.resize(1600, 900)

        # ── State ──────────────────────────────────────────────────────
        self.config = Config()
        self.scenario_name = "clean"
        self.worker = None
        self.worker_thread = None
        self.running = False
        self.paused = False
        self._dark_theme = True
        self.dev_mode = False
        self.csv_path = None  # tracked for report export

        # ── Build UI ───────────────────────────────────────────────────
        self._build_menu_bar()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._connect_shortcuts()

        # ── Worker signals ─────────────────────────────────────────────
        self._worker_frame.connect(self._on_frame, Qt.ConnectionType.QueuedConnection)
        self._worker_status.connect(self._on_status, Qt.ConnectionType.QueuedConnection)
        self._worker_plot.connect(self._on_plot, Qt.ConnectionType.QueuedConnection)
        self._worker_paused.connect(self._on_paused, Qt.ConnectionType.QueuedConnection)
        self._worker_done.connect(self._on_done, Qt.ConnectionType.QueuedConnection)
        self._worker_error.connect(self._on_error, Qt.ConnectionType.QueuedConnection)

        self._apply_theme()

    # ===================================================================
    #  UI construction
    # ===================================================================

    def _build_menu_bar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        act_cfg_load = QAction("Load &Config...", self)
        act_cfg_load.setShortcut(QKeySequence("Ctrl+O"))
        act_cfg_load.triggered.connect(self._on_load_config)
        file_menu.addAction(act_cfg_load)

        act_cfg_save = QAction("&Save Config...", self)
        act_cfg_save.setShortcut(QKeySequence("Ctrl+S"))
        act_cfg_save.triggered.connect(self._on_save_config)
        file_menu.addAction(act_cfg_save)

        file_menu.addSeparator()

        act_report = QAction("Export &Report...", self)
        act_report.setShortcut(QKeySequence("Ctrl+E"))
        act_report.triggered.connect(self._on_export_report)
        file_menu.addAction(act_report)

        act_replay = QAction("&Replay CSV...", self)
        act_replay.setShortcut(QKeySequence("Ctrl+R"))
        act_replay.triggered.connect(self._on_replay_csv)
        file_menu.addAction(act_replay)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── View menu ─────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")

        self._act_theme = QAction("Toggle &Dark/Light Theme", self)
        self._act_theme.setShortcut(QKeySequence("Ctrl+T"))
        self._act_theme.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(self._act_theme)

        self._act_dev = QAction("&Dev Mode (GT Overlay)", self)
        self._act_dev.setCheckable(True)
        self._act_dev.setShortcut(QKeySequence("G"))
        self._act_dev.toggled.connect(self._on_toggle_dev)
        view_menu.addAction(self._act_dev)

        # ── Help menu ─────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")

        act_shortcuts = QAction("&Keyboard Shortcuts...", self)
        act_shortcuts.setShortcut(QKeySequence("F1"))
        act_shortcuts.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(act_shortcuts)

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setIconSize(QSize(24, 24))
        tb.setMovable(False)
        self.addToolBar(tb)

        self._act_start = QAction("▶ Start", self)
        self._act_start.setShortcut(QKeySequence("Space"))
        self._act_start.setToolTip("Start / Pause simulation (Space)")
        self._act_start.triggered.connect(self._on_start_pause)
        tb.addAction(self._act_start)

        self._act_reset = QAction("↺ Reset", self)
        self._act_reset.setShortcut(QKeySequence("R"))
        self._act_reset.setToolTip("Reset simulation (R)")
        self._act_reset.triggered.connect(self._on_reset)
        tb.addAction(self._act_reset)

        tb.addSeparator()

        self._act_dev = QAction("🔧 Dev Mode", self)
        self._act_dev.setCheckable(True)
        self._act_dev.setShortcut(QKeySequence("G"))
        self._act_dev.setToolTip("Toggle developer overlay (G)")
        self._act_dev.toggled.connect(self._on_toggle_dev)
        tb.addAction(self._act_dev)

        self._act_theme = QAction("🌓 Theme", self)
        self._act_theme.setShortcut(QKeySequence("T"))
        self._act_theme.setToolTip("Toggle dark / light theme (T)")
        self._act_theme.triggered.connect(self._on_toggle_theme)
        tb.addAction(self._act_theme)

        tb.addSeparator()

        self._act_clean = QAction("Clean", self)
        self._act_clean.setShortcut(QKeySequence("1"))
        self._act_clean.setToolTip("Scenario: Clean (1)")
        self._act_clean.triggered.connect(
            lambda: self._select_scenario("clean"))
        tb.addAction(self._act_clean)

        self._act_hard = QAction("Hard", self)
        self._act_hard.setShortcut(QKeySequence("2"))
        self._act_hard.setToolTip("Scenario: Hard (2)")
        self._act_hard.triggered.connect(
            lambda: self._select_scenario("hard"))
        tb.addAction(self._act_hard)

    def _build_central(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ── Left: video + status ───────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self._video_label = _VideoLabel()
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_lay.addWidget(self._video_label, stretch=1)

        self._status_group = self._build_status_panel()
        left_lay.addWidget(self._status_group)

        splitter.addWidget(left)

        # ── Right: tabs (config, plot, report) ─────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(360)

        from gui.config_panel import ConfigPanel
        self._config_panel = ConfigPanel()
        self._config_panel.config_applied.connect(self._on_config_applied)
        self._tabs.addTab(self._config_panel, "Configuration")

        from gui.plot_widget import PlotWidget
        self._plot_widget = PlotWidget()
        self._tabs.addTab(self._plot_widget, "Plot")

        self._report_btn_widget = self._build_report_tab()
        self._tabs.addTab(self._report_btn_widget, "Report")

        splitter.addWidget(self._tabs)
        splitter.setSizes([900, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.setSpacing(12)

        info = QLabel(
            "Export a comprehensive report after a simulation run.\n\n"
            "The report includes:\n"
            "  - Per-frame CSV log\n"
            "  - JSON summary (all metrics)\n"
            "  - Self-contained HTML report\n"
            "  - PNG tracking-error plot"
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        btn = QPushButton("Export Report...")
        btn.setFixedHeight(36)
        btn.clicked.connect(self._on_export_report)
        lay.addWidget(btn)

        btn2 = QPushButton("Replay from CSV...")
        btn2.setFixedHeight(36)
        btn2.setToolTip("Load a CSV log and replay frames in the video panel")
        btn2.clicked.connect(self._on_replay_csv)
        lay.addWidget(btn2)

        lay.addStretch()
        return w

    def _build_status_panel(self) -> QWidget:
        grp = QGroupBox("Status")
        grp.setFixedHeight(160)
        lay = QGridLayout(grp)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setVerticalSpacing(2)
        lay.setHorizontalSpacing(12)

        self._lbl_state     = self._mk_lbl(lay, 0, 0, "Tracker State:")
        self._lbl_est_pos   = self._mk_lbl(lay, 0, 2, "Estimated:")
        self._lbl_conf      = self._mk_lbl(lay, 1, 0, "Confidence:")
        self._lbl_det_streak = self._mk_lbl(lay, 1, 2, "Det Streak:")
        self._lbl_pan       = self._mk_lbl(lay, 2, 0, "Pan Cmd:")
        self._lbl_tilt      = self._mk_lbl(lay, 2, 2, "Tilt Cmd:")
        self._lbl_err_px    = self._mk_lbl(lay, 3, 0, "Error px:")
        self._lbl_err_ang   = self._mk_lbl(lay, 3, 2, "Ang Error:")
        self._lbl_fps       = self._mk_lbl(lay, 4, 0, "FPS:")
        self._lbl_proc_ms   = self._mk_lbl(lay, 4, 2, "Proc Time:")
        self._lbl_sat       = self._mk_lbl(lay, 5, 0, "Saturated:")

        # GT Visible — only visible in dev mode (item 11)
        self._gt_vis_title  = QLabel("GT Visible:")
        lay.addWidget(self._gt_vis_title, 5, 2)
        self._lbl_gt_vis    = QLabel("--")
        self._lbl_gt_vis.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        font = self._lbl_gt_vis.font()
        font.setFamily("Consolas, Courier New, monospace")
        self._lbl_gt_vis.setFont(font)
        lay.addWidget(self._lbl_gt_vis, 5, 3)
        self._gt_vis_title.setVisible(False)
        self._lbl_gt_vis.setVisible(False)

        return grp

    @staticmethod
    def _mk_lbl(grid: QGridLayout, row: int, col: int,
                title: str) -> QLabel:
        grid.addWidget(QLabel(title), row, col)
        val = QLabel("--")
        val.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        font = val.font()
        font.setFamily("Consolas, Courier New, monospace")
        val.setFont(font)
        grid.addWidget(val, row, col + 1)
        return val

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.showMessage("Ready  |  Press Start or SPACE to begin")

    def _connect_shortcuts(self):
        # Additional shortcut: Escape to quit
        QShortcut(QKeySequence("Escape"), self, self.close)

    def _apply_theme(self):
        """Apply a minimal dark palette for a professional look."""
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor(200, 200, 200))
        pal.setColor(QPalette.ColorRole.Base,             QColor(25, 25, 25))
        pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(35, 35, 35))
        pal.setColor(QPalette.ColorRole.Text,            QColor(200, 200, 200))
        pal.setColor(QPalette.ColorRole.Button,          QColor(40, 40, 40))
        pal.setColor(QPalette.ColorRole.ButtonText,      QColor(200, 200, 200))
        pal.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(pal)

        self.setStyleSheet("""
            QToolBar          { spacing: 6px; padding: 2px; }
            QGroupBox         { border: 1px solid #555; border-radius: 4px;
                                margin-top: 8px; padding-top: 14px; }
            QGroupBox::title  { subcontrol-origin: margin; left: 10px;
                                padding: 0 4px; }
            QTabWidget::pane  { border: 1px solid #555; }
            QTabBar::tab      { background: #2a2a2a; color: #aaa;
                                padding: 6px 16px; border: 1px solid #555;
                                border-bottom: none; border-top-left-radius: 4px;
                                border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #3a3a3a; color: #eee; }
            QTabBar::tab:hover     { background: #444; }
            QPushButton       { padding: 4px 12px; border: 1px solid #555;
                                border-radius: 3px; background: #3a3a3a; }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:pressed { background: #2a2a2a; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background: #2a2a2a; color: #ddd; border: 1px solid #555;
                border-radius: 3px; padding: 3px 6px;
            }
            QCheckBox         { spacing: 6px; }
            QLabel            { color: #ccc; }
            QStatusBar        { background: #1e1e1e; color: #aaa; }
        """)

    # ===================================================================
    #  Worker lifecycle
    # ===================================================================

    def _create_worker(self):
        self.worker = SimulationWorker(self.config, self.scenario_name)
        self.worker.output_dir = os.path.join(_SRC_DIR, "..", "results")
        self.worker.save_frames = self.config.save_frames
        self.worker.dev_mode = self.dev_mode

        self.worker_thread = QThread(self)
        self.worker.moveToThread(self.worker_thread)

        # Wire worker-internal signals to our one-shot forwarding signals.
        self.worker.frame_ready.connect(
            lambda qimg, fps: self._worker_frame.emit(qimg, fps))
        self.worker.status_updated.connect(
            lambda d: self._worker_status.emit(d))
        self.worker.plot_updated.connect(
            lambda t, e, c: self._worker_plot.emit(t, e, c))
        self.worker.paused_state.connect(
            lambda p: self._worker_paused.emit(p))
        self.worker.simulation_done.connect(
            lambda s: self._worker_done.emit(s))
        self.worker.error_occurred.connect(
            lambda e: self._worker_error.emit(e))

        # Thread start -> worker.run
        self.worker_thread.started.connect(self.worker.run)
        # Clean up thread when done
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(
            lambda: setattr(self, "worker_thread", None))

    def _start_worker(self):
        self._create_worker()
        self.running = True
        self.paused = False
        self._update_button_states()
        self.worker_thread.start()
        self.statusBar().showMessage(
            f"Running  |  Scenario: {self.scenario_name.upper()}")

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        self.worker = None
        self.running = False
        self.paused = False

        self._update_button_states()

    # ===================================================================
    #  Slots: toolbar / keyboard actions
    # ===================================================================

    def _on_start_pause(self):
        if not self.running:
            self._start_worker()
        elif self.paused:
            if self.worker is not None:
                self.worker.request_resume()
        else:
            if self.worker is not None:
                self.worker.request_pause()

    def _on_reset(self):
        self._stop_worker()
        self._video_label.clear()
        self._video_label.setText(
            "Press Start to begin simulation")
        self._video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._reset_status_values()
        self.statusBar().showMessage("Ready  |  Reset complete")

    def _on_toggle_dev(self, checked: bool):
        self.dev_mode = checked
        if self.worker is not None:
            self.worker.set_dev_mode(checked)
        self.statusBar().showMessage(
            f"Dev mode {'ON' if checked else 'OFF'}")

    def _on_toggle_theme(self):
        """Toggle between dark and light theme."""
        pal = self.palette()
        bg = pal.color(QPalette.ColorRole.Window)
        # If currently dark (R < 80), switch to light; else switch to dark
        if bg.red() < 80:
            self._apply_light_theme()
            self.statusBar().showMessage("Theme: Light")
        else:
            self._apply_theme()
            self.statusBar().showMessage("Theme: Dark")

    def _apply_light_theme(self):
        """Apply a light palette."""
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(235, 235, 235))
        pal.setColor(QPalette.ColorRole.Text, QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(pal)
        self.setStyleSheet("""
            QToolBar          { spacing: 6px; padding: 2px; }
            QGroupBox         { border: 1px solid #bbb; border-radius: 4px;
                                margin-top: 8px; padding-top: 14px; }
            QGroupBox::title  { subcontrol-origin: margin; left: 10px;
                                padding: 0 4px; }
            QTabWidget::pane  { border: 1px solid #bbb; }
            QTabBar::tab      { background: #e0e0e0; color: #333;
                                padding: 6px 16px; border: 1px solid #bbb;
                                border-bottom: none; }
            QTabBar::tab:selected { background: #fff; color: #111; }
            QPushButton       { padding: 4px 12px; border: 1px solid #bbb;
                                border-radius: 3px; background: #e8e8e8; }
            QPushButton:hover { background: #d0d0d0; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background: #fff; color: #222; border: 1px solid #bbb;
                border-radius: 3px; padding: 3px 6px;
            }
            QLabel            { color: #333; }
            QStatusBar        { background: #f0f0f0; color: #555; }
        """)

    def _on_show_shortcuts(self):
        """Display a dialog with all keyboard shortcuts."""
        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
<tr><th>Key</th><th>Action</th></tr>
<tr><td><b>Space</b></td><td>Start / Pause simulation</td></tr>
<tr><td><b>R</b></td><td>Reset simulation</td></tr>
<tr><td><b>1</b></td><td>Switch to Clean scenario</td></tr>
<tr><td><b>2</b></td><td>Switch to Hard scenario</td></tr>
<tr><td><b>G</b></td><td>Toggle Dev Mode (GT overlay)</td></tr>
<tr><td><b>Escape</b></td><td>Quit application</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Load config (YAML)</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save config (YAML)</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export report</td></tr>
<tr><td><b>Ctrl+R</b></td><td>Replay from CSV</td></tr>
<tr><td><b>Ctrl+T</b></td><td>Toggle Dark/Light theme</td></tr>
<tr><td><b>F1</b></td><td>Show this help</td></tr>
</table>
"""
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _select_scenario(self, name: str):
        self.scenario_name = name
        if self.running:
            self._stop_worker()
            self._start_worker()
        self.statusBar().showMessage(
            f"Scenario: {name.upper()}  |  Press Start to begin")

    def _on_config_applied(self, config: Config):
        self.config = config
        if self.running:
            self._stop_worker()
            self._start_worker()

    # ===================================================================
    #  Slots: worker signals  (all run in main thread via queued conn.)
    # ===================================================================

    @Slot(object, float)
    def _on_frame(self, qimg: QImage, fps: float):
        self._video_label.set_image(qimg)

    @Slot(object)
    def _on_status(self, s: dict):
        state = s.get("state", "?")

        # State colors for visual feedback — covers every TrackerState
        state_colors = {
            "IDLE":                     "#888888",
            "SEARCHING":                "#e6a817",
            "CANDIDATE_VERIFICATION":   "#d4a017",
            "ACQUIRING":                "#2196f3",
            "TRACKING":                 "#4caf50",
            "REACQUIRING":              "#e74c3c",
            "FAILED":                   "#b71c1c",
        }
        color = state_colors.get(state, "#aaa")

        # Color the state label itself
        self._lbl_state.setText(state)
        self._lbl_state.setStyleSheet(f"color: {color}; font-weight: bold;")

        self._lbl_est_pos.setText(
            f"({s.get('est_x', 0):.0f}, {s.get('est_y', 0):.0f})")
        self._lbl_conf.setText(f"{s.get('confidence', 0):.3f}")

        streak = s.get("consecutive_detections", 0)
        misses = s.get("consecutive_misses", 0)
        self._lbl_det_streak.setText(f"{streak} / miss:{misses}")

        self._lbl_pan.setText(f"{s.get('pan_cmd_deg', 0):+.3f} deg")
        self._lbl_tilt.setText(f"{s.get('tilt_cmd_deg', 0):+.3f} deg")

        # Color error label by magnitude
        err_px = s.get("error_pixels", 0)
        self._lbl_err_px.setText(f"{err_px:.1f} px")
        err_color = "#4caf50" if err_px < 50 else "#e6a817" if err_px < 150 else "#e74c3c"
        self._lbl_err_px.setStyleSheet(f"color: {err_color};")

        self._lbl_err_ang.setText(
            f"{s.get('angular_error_deg', 0):.4f} deg")
        self._lbl_fps.setText(f"{s.get('fps', 0):.1f}")
        self._lbl_proc_ms.setText(f"{s.get('processing_time_ms', 0):.1f} ms")

        # Color saturation indicator
        sat = s.get("saturated", False)
        self._lbl_sat.setText("Yes" if sat else "No")
        self._lbl_sat.setStyleSheet(
            f"color: {'#e74c3c' if sat else '#4caf50'};")

        # GT visible — only show when dev_mode is on
        if self.dev_mode:
            self._lbl_gt_vis.setText(
                "Yes" if s.get("gt_visible", False) else "No")
            self._gt_vis_title.setVisible(True)
            self._lbl_gt_vis.setVisible(True)
        else:
            self._gt_vis_title.setVisible(False)
            self._lbl_gt_vis.setVisible(False)

        # Video border color by state
        self._video_label.setStyleSheet(
            f"background: #111; border: 2px solid {color};")

    @Slot(float, float, float)
    def _on_plot(self, timestamp: float, error_px: float, confidence: float):
        self._plot_widget.update_data(timestamp, error_px, confidence)

    @Slot(bool)
    def _on_paused(self, is_paused: bool):
        self.paused = is_paused
        self._update_button_states()
        self._video_label.set_paused(is_paused)
        if is_paused:
            self.statusBar().showMessage(
                "PAUSED  |  Press SPACE or Start to resume")
        else:
            self.statusBar().showMessage(
                f"Running  |  Scenario: {self.scenario_name.upper()}")

    @Slot(object)
    def _on_done(self, stats: dict):
        self.running = False
        self.paused = False
        self.csv_path = stats.pop("_csv_path", None) if isinstance(stats, dict) else None

        self._update_button_states()

        if stats:
            n = stats.get("num_frames", 0)
            err = stats.get("mean_error_px", 0)
            ret = stats.get("lock_retention_pct", 0)
            self.statusBar().showMessage(
                f"Done  |  {n} frames  |  "
                f"Mean error: {err:.1f} px  |  "
                f"Lock: {ret:.1f}%")
        else:
            self.statusBar().showMessage("Done")

    @Slot(str)
    def _on_error(self, msg: str):
        self.running = False
        self.paused = False

        self._update_button_states()
        QMessageBox.critical(self, "Simulation Error", msg)
        self.statusBar().showMessage(f"Error: {msg}")

    # ===================================================================
    #  Menu actions
    # ===================================================================

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "",
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return
        try:
            from yaml_config import load_scenario, scenario_to_config
            sc = load_scenario(path)
            self.config = scenario_to_config(sc)
            self._config_panel.load_config(self.config)
            self.statusBar().showMessage(f"Config loaded: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "scenario.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return
        try:
            from yaml_config import save_scenario
            save_scenario(self.config, path, self.scenario_name)
            self.statusBar().showMessage(f"Config saved: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _on_export_report(self):
        from gui.report_dialog import ReportDialog
        dlg = ReportDialog(self.config, self.scenario_name,
                           csv_path=self.csv_path, parent=self)
        dlg.exec()

    def _on_replay_csv(self):
        """Load a CSV log and replay the error/confidence timeline in the plot."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV Log", "",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            import csv as csv_mod
            self._stop_worker()
            self._plot_widget.clear_data()
            with open(path, 'r') as f:
                lines = [l for l in f if not l.startswith('#')]
            reader = csv_mod.DictReader(lines)
            for row in reader:
                ts = float(row.get('timestamp', 0))
                err = float(row.get('error_px', 0))
                conf = float(row.get('detection_confidence', 0))
                self._plot_widget.update_data(ts, err, conf)
            # Force a final draw
            self._plot_widget._draw()
            self.csv_path = path
            self.statusBar().showMessage(f"Replayed CSV: {path}")
            self._tabs.setCurrentWidget(self._plot_widget)
        except Exception as exc:
            QMessageBox.warning(self, "Replay Error", str(exc))

    def _on_toggle_theme(self):
        """Toggle between dark and light themes."""
        self._dark_theme = not self._dark_theme
        if self._dark_theme:
            self._apply_theme()
        else:
            self._apply_light_theme()

    def _apply_light_theme(self):
        """Apply a professional light palette."""
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,          QColor(240, 240, 240))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Base,             QColor(255, 255, 255))
        pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(230, 230, 230))
        pal.setColor(QPalette.ColorRole.Text,            QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Button,          QColor(225, 225, 225))
        pal.setColor(QPalette.ColorRole.ButtonText,      QColor(30, 30, 30))
        pal.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(pal)

        self.setStyleSheet("""
            QToolBar          { spacing: 6px; padding: 2px; }
            QGroupBox         { border: 1px solid #bbb; border-radius: 4px;
                                margin-top: 8px; padding-top: 14px; }
            QGroupBox::title  { subcontrol-origin: margin; left: 10px;
                                padding: 0 4px; }
            QTabWidget::pane  { border: 1px solid #bbb; }
            QTabBar::tab      { background: #e0e0e0; color: #333;
                                padding: 6px 16px; border: 1px solid #bbb;
                                border-bottom: none; border-top-left-radius: 4px;
                                border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #fff; color: #111; }
            QTabBar::tab:hover     { background: #d0d0d0; }
            QPushButton       { padding: 4px 12px; border: 1px solid #bbb;
                                border-radius: 3px; background: #e8e8e8; }
            QPushButton:hover { background: #d0d0d0; }
            QPushButton:pressed { background: #c0c0c0; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background: #fff; color: #222; border: 1px solid #bbb;
                border-radius: 3px; padding: 3px 6px;
            }
            QCheckBox         { spacing: 6px; }
            QLabel            { color: #333; }
            QStatusBar        { background: #e0e0e0; color: #555; }
        """)

    # ===================================================================
    #  Helpers
    # ===================================================================

    def _update_button_states(self):
        if self.running and not self.paused:
            self._act_start.setText("Pause")
            self._act_start.setToolTip("Pause simulation (Space)")
        else:
            self._act_start.setText("Start")
            self._act_start.setToolTip("Start simulation (Space)")

    def _reset_status_values(self):
        for lbl in (self._lbl_state, self._lbl_est_pos, self._lbl_conf,
                    self._lbl_det_streak, self._lbl_pan, self._lbl_tilt,
                    self._lbl_err_px, self._lbl_err_ang,
                    self._lbl_fps, self._lbl_proc_ms,
                    self._lbl_sat, self._lbl_gt_vis):
            lbl.setText("--")
        self._video_label.setStyleSheet("background: #111; border: 2px solid #555;")

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()


# =========================================================================
#  Video display label with pause overlay
# =========================================================================

class _VideoLabel(QLabel):
    """QLabel that displays the camera feed and optionally a PAUSED overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111; border: 2px solid #555;")
        self.setText("Press Start to begin simulation")
        self._paused = False
        self._current_pixmap = None

    def set_image(self, qimg: QImage):
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._render()

    def set_paused(self, paused: bool):
        self._paused = paused
        self._render()

    def clear(self):
        self._current_pixmap = None
        super().clear()
        self.setText("Press Start to begin simulation")

    def _render(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._paused:
            # Dim the image
            from PySide6.QtGui import QPixmap as _QP, QColor, QPainter
            dimmed = _QP(scaled.size())
            dimmed.fill(QColor(0, 0, 0, 100))
            painter = QPainter(scaled)
            painter.drawPixmap(0, 0, dimmed)
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setPixelSize(28)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(scaled.rect(),
                             Qt.AlignmentFlag.AlignCenter, "PAUSED")
            painter.end()
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()
