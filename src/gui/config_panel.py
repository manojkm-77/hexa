"""
config_panel.py — Configuration panel widget for the FSOC simulator GUI.

Provides labeled controls for all tunable parameters:
  - Scenario selection (clean / hard / custom)
  - PID controller gains (kp, ki, kd, deadband)
  - Disturbance toggles and magnitudes
  - Detector and filter selection
  - Apply button that pushes changes into Config

The panel is a self-contained QWidget.  It reads defaults from a Config
instance and emits a new Config via the config_applied signal when the
user clicks Apply.
"""

import sys
import os
import copy

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
        QCheckBox, QPushButton, QGroupBox, QScrollArea,
        QSizePolicy, QFrame,
    )
    from PySide6.QtCore import Qt, Signal

    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import Config


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_row(widgets, spacing=6):
    """Create a horizontal layout from a list of (label_text, widget) pairs,
    or just widget instances."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(0, 0, 0, 0)
    for item in widgets:
        if isinstance(item, str):
            lbl = QLabel(item)
            lbl.setFixedWidth(110)
            lay.addWidget(lbl)
        else:
            lay.addWidget(item)
    return lay


def _make_separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ── Default disturbance magnitudes for preset scenarios ──────────────────

_CLEAN_DEFAULTS = {
    "vibration": 0.0,
    "noise": 0.0,
    "blur": 0.0,
    "turb": 0.0,
    "exposure": 0.0,
    "occl": 0.0,
    "false_beacons": 0,
}

_HARD_DEFAULTS = {
    "vibration": 0.2,
    "noise": 12.0,
    "blur": 3.0,
    "turb": 0.1,
    "exposure": 0.1,
    "occl": 0.05,
    "false_beacons": 2,
}

# v1.1 disturbance-only presets
_TURBULENCE_DEFAULTS = {
    "vibration": 0.0, "noise": 0.0, "blur": 0.0,
    "turb": 0.1, "exposure": 0.0, "occl": 0.0, "false_beacons": 0,
}
_VIBRATION_DEFAULTS = {
    "vibration": 0.2, "noise": 0.0, "blur": 0.0,
    "turb": 0.0, "exposure": 0.0, "occl": 0.0, "false_beacons": 0,
}
_SENSOR_DEGRADATION_DEFAULTS = {
    "vibration": 0.0, "noise": 12.0, "blur": 3.0,
    "turb": 0.0, "exposure": 0.1, "occl": 0.0, "false_beacons": 0,
}
_OCCLUSION_DEFAULTS = {
    "vibration": 0.0, "noise": 0.0, "blur": 0.0,
    "turb": 0.0, "exposure": 0.0, "occl": 1.0, "false_beacons": 0,
}
_CLUTTER_DEFAULTS = {
    "vibration": 0.0, "noise": 0.0, "blur": 0.0,
    "turb": 0.0, "exposure": 0.0, "occl": 0.0, "false_beacons": 3,
}

_PRESET_MAP = {
    "clean": _CLEAN_DEFAULTS,
    "hard": _HARD_DEFAULTS,
    "turbulence": _TURBULENCE_DEFAULTS,
    "vibration": _VIBRATION_DEFAULTS,
    "sensor_degradation": _SENSOR_DEGRADATION_DEFAULTS,
    "occlusion": _OCCLUSION_DEFAULTS,
    "clutter": _CLUTTER_DEFAULTS,
}


# =========================================================================
#  ConfigPanel
# =========================================================================

class ConfigPanel(QWidget):
    """QWidget that exposes every tunable parameter with a labelled control.

    Signals
    -------
    config_applied(Config)
        Emitted when the user clicks Apply. The Config object carries the
        current values of every control.
    """

    config_applied = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controls = {}     # name -> (widget, type_hint)
        self._base_config = Config()
        self._build_ui()

    # ===================================================================
    #  UI construction
    # ===================================================================

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(6)

        # ── Scenario ───────────────────────────────────────────────────
        grp = QGroupBox("Scenario")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self._scenario_combo = QComboBox()
        self._scenario_combo.addItems([
            "clean", "hard", "custom",
            "turbulence", "vibration", "sensor_degradation",
            "occlusion", "clutter",
        ])
        self._scenario_combo.currentTextChanged.connect(
            self._on_scenario_changed)
        row.addWidget(self._scenario_combo, 1)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FPS:"))
        self._fps_spin = self._add_spin("fps", min_v=1, max_v=120,
                                        val=30)
        row2.addWidget(self._fps_spin)
        row2.addWidget(QLabel("Duration (s):"))
        self._dur_spin = self._add_spin("duration_s",
                                        min_v=1, max_v=600, val=60)
        row2.addWidget(self._dur_spin)
        row2.addStretch()
        gl.addLayout(row2)

        row3 = QHBoxLayout()
        self._save_frames_cb = QCheckBox("Save frames (every 30th frame)")
        row3.addWidget(self._save_frames_cb)
        row3.addStretch()
        gl.addLayout(row3)

        lay.addWidget(grp)

        # ── PID gains ──────────────────────────────────────────────────
        grp = QGroupBox("PID Controller")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        for label, name, lo, hi, step, dec in [
            ("Kp", "kp", 0.0, 10.0, 0.05, 3),
            ("Ki", "ki", 0.0, 5.0, 0.01, 4),
            ("Kd", "kd", 0.0, 5.0, 0.01, 3),
        ]:
            row.addWidget(QLabel(f"{label}:"))
            sp = self._add_dspin(name, lo, hi, step, dec)
            row.addWidget(sp)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Deadband (px):"))
        self._deadband_spin = self._add_dspin(
            "deadband_pixels", 0.0, 50.0, 0.5, 1)
        row2.addWidget(self._deadband_spin)
        row2.addStretch()
        gl.addLayout(row2)

        lay.addWidget(grp)

        # ── Detector / Filter ──────────────────────────────────────────
        grp = QGroupBox("Detection & Filtering")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Detector:"))
        self._det_combo = QComboBox()
        self._det_combo.addItems(["classical", "ai"])
        self._det_combo.currentTextChanged.connect(
            lambda v: self._controls.__setitem__("detector_type", (v, "combo")))
        row.addWidget(self._det_combo)

        row.addSpacing(16)
        row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["kalman", "alpha_beta"])
        self._controls["filter_type"] = (self._filter_combo, "combo")
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        row.addWidget(self._filter_combo)
        row.addStretch()
        gl.addLayout(row)

        # Filter noise & params
        row2 = QHBoxLayout()
        self._kalman_label = QLabel("Kalman Noise:")
        row2.addWidget(self._kalman_label)
        self._kalman_noise = self._add_dspin(
            "measurement_noise", 0.1, 100.0, 1.0, 1)
        row2.addWidget(self._kalman_noise)

        self._alpha_label = QLabel("Alpha:")
        row2.addWidget(self._alpha_label)
        self._alpha_spin = self._add_dspin(
            "alpha", 0.01, 1.0, 0.05, 2)
        row2.addWidget(self._alpha_spin)

        self._beta_label = QLabel("Beta:")
        row2.addWidget(self._beta_label)
        self._beta_spin = self._add_dspin(
            "beta", 0.001, 1.0, 0.01, 3)
        row2.addWidget(self._beta_spin)

        row2.addWidget(QLabel("Threshold:"))
        self._threshold_spin = self._add_spin(
            "threshold", min_v=1, max_v=255, val=80)
        row2.addWidget(self._threshold_spin)
        row2.addStretch()
        gl.addLayout(row2)

        self._on_filter_changed("kalman")

        lay.addWidget(grp)

        # ── Multi-target ─────────────────────────────────────────────
        grp = QGroupBox("Multi-Target")
        gl = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._multi_cb = QCheckBox("Enable multi-target tracking")
        row.addWidget(self._multi_cb)
        row.addStretch()
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Num targets:"))
        self._num_targets_spin = self._add_spin(
            "num_targets", min_v=2, max_v=10, val=2)
        self._num_targets_spin.setEnabled(False)
        self._multi_cb.toggled.connect(self._num_targets_spin.setEnabled)
        row2.addWidget(self._num_targets_spin)

        row2.addSpacing(12)
        row2.addWidget(QLabel("Assignment:"))
        self._assign_combo = QComboBox()
        self._assign_combo.addItems(["round_robin", "nearest"])
        self._assign_combo.setEnabled(False)
        self._assign_combo.setToolTip(
            "How detected beacons are assigned to trackers")
        self._multi_cb.toggled.connect(self._assign_combo.setEnabled)
        row2.addWidget(self._assign_combo)

        row2.addStretch()
        gl.addLayout(row2)

        lay.addWidget(grp)

        # ── Disturbances ───────────────────────────────────────────────
        grp = QGroupBox("Disturbances")
        gl = QVBoxLayout(grp)

        disturbances = [
            ("Platform Vibration RMS (deg)", "vibration", "clean_vibration_rms",
             QDoubleSpinBox, 0.0, 5.0, 0.01, 3, "vib"),
            ("Sensor Noise Sigma (px)", "noise", "clean_noise_sigma",
             QDoubleSpinBox, 0.0, 50.0, 0.5, 1, "noise"),
            ("Motion Blur (px)", "blur", "clean_blur_pixels",
             QDoubleSpinBox, 0.0, 20.0, 0.1, 2, "blur"),
            ("Turbulence RMS (deg)", "turb", "turbulence_rms_deg",
             QDoubleSpinBox, 0.0, 2.0, 0.01, 3, "turb"),
            ("Exposure Variation", "exposure", "exposure_amplitude",
             QDoubleSpinBox, 0.0, 1.0, 0.01, 3, "expos"),
            ("Occlusion Freq (Hz)", "occl", "occlusion_frequency_hz",
             QDoubleSpinBox, 0.0, 1.0, 0.01, 3, "occl"),
        ]

        self._disturb_widgets = {}
        for row_label, key, cfg_attr, spin_type, lo, hi, step, dec, uid in \
                disturbances:
            row = QHBoxLayout()
            cb = QCheckBox(row_label)
            cb.setMinimumWidth(170)
            sp = spin_type()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setEnabled(False)
            cb.toggled.connect(sp.setEnabled)
            row.addWidget(cb)
            row.addWidget(sp)
            row.addStretch()
            gl.addLayout(row)
            self._disturb_widgets[key] = (cb, sp, cfg_attr)

        # False beacons (integer)
        row = QHBoxLayout()
        self._fb_cb = QCheckBox("False Beacons Count")
        self._fb_cb.setMinimumWidth(170)
        self._fb_spin = QSpinBox()
        self._fb_spin.setRange(0, 20)
        self._fb_spin.setValue(2)
        self._fb_spin.setEnabled(False)
        self._fb_cb.toggled.connect(self._fb_spin.setEnabled)
        row.addWidget(self._fb_cb)
        row.addWidget(self._fb_spin)
        row.addStretch()
        gl.addLayout(row)
        self._disturb_widgets["false_beacons"] = (
            self._fb_cb, self._fb_spin, "false_beacon_count")

        lay.addWidget(grp)

        # ── Apply ──────────────────────────────────────────────────────
        lay.addWidget(_make_separator())

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._apply_btn = QPushButton("Apply Configuration")
        self._apply_btn.setFixedHeight(32)
        self._apply_btn.setMinimumWidth(160)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)
        lay.addLayout(btn_row)

        lay.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

        # Set defaults from a fresh Config
        self.load_config(Config())

    # ===================================================================
    #  Public API
    # ===================================================================

    def _on_filter_changed(self, filter_type: str):
        is_kalman = (filter_type == "kalman")
        if hasattr(self, "_kalman_label"):
            self._kalman_label.setVisible(is_kalman)
            self._kalman_noise.setVisible(is_kalman)
            self._alpha_label.setVisible(not is_kalman)
            self._alpha_spin.setVisible(not is_kalman)
            self._beta_label.setVisible(not is_kalman)
            self._beta_spin.setVisible(not is_kalman)

    def load_config(self, config: Config):
        """Populate all controls from an existing Config instance."""
        self._base_config = copy.copy(config)
        self._fps_spin.setValue(config.fps)
        self._dur_spin.setValue(config.duration_s)
        self._save_frames_cb.setChecked(config.save_frames)
        self._det_combo.setCurrentText(config.detector_type)
        self._filter_combo.setCurrentText(config.filter_type)
        self._kalman_noise.setValue(config.measurement_noise)
        self._alpha_spin.setValue(config.alpha)
        self._beta_spin.setValue(config.beta)
        self._threshold_spin.setValue(config.threshold)
        self._on_filter_changed(config.filter_type)

        # Multi-target
        self._multi_cb.setChecked(config.multi_target)
        self._num_targets_spin.setValue(config.num_targets)
        self._num_targets_spin.setEnabled(config.multi_target)

        # PID
        self._set_dspin("kp", config.kp)
        self._set_dspin("ki", config.ki)
        self._set_dspin("kd", config.kd)
        self._set_dspin("deadband_pixels", config.deadband_pixels)

        # Disturbances
        for key, (cb, sp, cfg_attr) in self._disturb_widgets.items():
            if key == "false_beacons":
                val = getattr(config, cfg_attr, 0)
                cb.setChecked(val > 0)
                sp.setValue(val)
            else:
                val = getattr(config, cfg_attr, 0.0)
                cb.setChecked(val > 0)
                sp.setValue(val)

    def get_config(self) -> Config:
        """Build and return a Config from the current control values."""
        c = copy.copy(self._base_config)

        c.fps = self._fps_spin.value()
        c.duration_s = self._dur_spin.value()
        c.save_frames = self._save_frames_cb.isChecked()
        c.detector_type = self._det_combo.currentText()
        c.filter_type = self._filter_combo.currentText()
        c.measurement_noise = self._kalman_noise.value()
        c.alpha = self._alpha_spin.value()
        c.beta = self._beta_spin.value()
        c.threshold = self._threshold_spin.value()

        c.kp = self._get_dspin("kp")
        c.ki = self._get_dspin("ki")
        c.kd = self._get_dspin("kd")
        c.deadband_pixels = self._get_dspin("deadband_pixels")

        # Multi-target
        c.multi_target = self._multi_cb.isChecked()
        c.num_targets = self._num_targets_spin.value()

        # Disturbances
        for key, (cb, sp, cfg_attr) in self._disturb_widgets.items():
            if cb.isChecked():
                if key == "false_beacons":
                    setattr(c, cfg_attr, sp.value())
                else:
                    setattr(c, cfg_attr, sp.value())
            else:
                if key == "false_beacons":
                    setattr(c, cfg_attr, 0)
                else:
                    setattr(c, cfg_attr, 0.0)

        return c

    # ===================================================================
    #  Slots
    # ===================================================================

    def _on_scenario_changed(self, name: str):
        """When a preset is chosen, populate disturbance defaults."""
        defaults = _PRESET_MAP.get(name)
        if defaults is None:
            return     # custom: leave controls as-is

        for key, val in defaults.items():
            if key in self._disturb_widgets:
                cb, sp, _ = self._disturb_widgets[key]
                cb.setChecked(val > 0)
                sp.setValue(val)

    def _on_apply(self):
        """Collect all values and emit a Config."""
        cfg = self.get_config()
        self.config_applied.emit(cfg)

    # ===================================================================
    #  Internal helpers
    # ===================================================================

    def _add_dspin(self, name, lo, hi, step, dec) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        self._controls[name] = (sp, "dspin")
        return sp

    def _add_spin(self, name, min_v=0, max_v=1000, val=0) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(min_v, max_v)
        sp.setValue(val)
        self._controls[name] = (sp, "spin")
        return sp

    def _get_dspin(self, name) -> float:
        w, _ = self._controls.get(name, (None, None))
        return w.value() if w else 0.0

    def _set_dspin(self, name, val):
        w, _ = self._controls.get(name, (None, None))
        if w is not None:
            w.setValue(val)
