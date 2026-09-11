"""
plot_widget.py — Real-time embedded plot for the FSOC simulator GUI.

Renders two vertically-stacked sub-plots using pyqtgraph for performance:
  1. Pixel tracking error over time  (with 50 px / 128 px threshold lines)
  2. Detection confidence over time

Features:
  - Auto-scrolling window of the last 300 frames
  - High performance plotting suited for live updates

Dependencies: PySide6, pyqtgraph, numpy
"""

import sys
import os
import numpy as np

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
    from PySide6.QtCore import Qt
    import pyqtgraph as pg
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_WINDOW_SIZE = 300

class PlotWidget(QWidget):
    """Embeds a two-panel pyqtgraph layout in a Qt widget.

    Call update_data() from the GUI thread to add a new data point and refresh the
    plot. Call clear_data() on reset.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._timestamps = []
        self._errors = []
        self._confidences = []

        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if not _AVAILABLE:
            lbl = QLabel(
                "Plot unavailable — install PySide6 and pyqtgraph.\n"
                "  uv pip install PySide6 pyqtgraph")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            return

        # Use antialiasing for smoother lines
        pg.setConfigOptions(antialias=True)
        # Background color to match the theme
        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', '#ccc')

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.graphics_layout)

        # ── Error plot setup ──
        self.p1 = self.graphics_layout.addPlot(row=0, col=0, title="Tracking Error")
        self.p1.setLabel('left', 'Pixel Error')
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p1.setYRange(0, 300)
        
        # Add threshold lines
        self.thresh50 = pg.InfiniteLine(angle=0, pen=pg.mkPen(color='#4caf50', style=Qt.PenStyle.DashLine, width=1.5))
        self.thresh50.setValue(50)
        self.p1.addItem(self.thresh50)
        
        self.thresh128 = pg.InfiniteLine(angle=0, pen=pg.mkPen(color='#ff9800', style=Qt.PenStyle.DashLine, width=1.5))
        self.thresh128.setValue(128)
        self.p1.addItem(self.thresh128)

        self.curve_error = self.p1.plot(pen=pg.mkPen(color='#2E86AB', width=2))

        # ── Confidence plot setup ──
        self.p2 = self.graphics_layout.addPlot(row=1, col=0, title="Detection Confidence")
        self.p2.setLabel('left', 'Confidence')
        self.p2.setLabel('bottom', 'Time (s)')
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p2.setYRange(-0.05, 1.15)
        self.p2.setXLink(self.p1) # Link x-axis with the first plot

        # We'll use a curve with a brush for a filled look
        self.curve_conf = self.p2.plot(pen=pg.mkPen(color='#2E86AB', width=2), brush=pg.mkBrush(color=(46, 134, 171, 100)), fillLevel=0)

    def update_data(self, timestamp: float, error_px: float, confidence: float):
        if not _AVAILABLE:
            return

        self._timestamps.append(timestamp)
        self._errors.append(error_px)
        self._confidences.append(confidence)

        # Trim to window
        if len(self._timestamps) > _WINDOW_SIZE:
            self._timestamps = self._timestamps[-_WINDOW_SIZE:]
            self._errors = self._errors[-_WINDOW_SIZE:]
            self._confidences = self._confidences[-_WINDOW_SIZE:]

        self._draw()

    def clear_data(self):
        if not _AVAILABLE:
            return
        self._timestamps.clear()
        self._errors.clear()
        self._confidences.clear()
        self._draw()

    def _draw(self):
        if not self._timestamps:
            self.curve_error.setData([], [])
            self.curve_conf.setData([], [])
            return

        ts = np.array(self._timestamps).ravel()
        errs = np.array(self._errors).ravel()
        confs = np.array(self._confidences).ravel()

        self.curve_error.setData(ts, errs)
        self.curve_conf.setData(ts, confs)
        
        # Adjust Y bounds for error
        ymax = max(np.max(errs) * 1.15, 100)
        self.p1.setYRange(0, ymax)
