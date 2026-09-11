"""Tests for the PySide6 GUI components."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

HAS_PYSIDE6 = False
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    HAS_PYSIDE6 = True
except ImportError:
    pass

# Need a QApplication instance for all widget tests
_app = None


def get_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestMainWindow:
    """Tests for MainWindow."""

    def test_creation(self):
        """MainWindow can be instantiated."""
        from gui.main_window import MainWindow
        get_app()
        win = MainWindow()
        assert win.windowTitle() == "FSOC Coarse-Alignment Simulator"
        win.close()

    def test_minimum_size(self):
        """MainWindow has correct minimum size."""
        from gui.main_window import MainWindow
        get_app()
        win = MainWindow()
        assert win.minimumWidth() >= 1280
        assert win.minimumHeight() >= 720
        win.close()


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestConfigPanel:
    """Tests for ConfigPanel."""

    def test_creation(self):
        """ConfigPanel can be instantiated."""
        from gui.config_panel import ConfigPanel
        get_app()
        panel = ConfigPanel()
        assert panel is not None

    def test_get_config(self):
        """ConfigPanel returns a valid config."""
        from gui.config_panel import ConfigPanel
        get_app()
        panel = ConfigPanel()
        cfg = panel.get_config()
        assert hasattr(cfg, 'kp')
        assert hasattr(cfg, 'filter_type')


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestPlotWidget:
    """Tests for PlotWidget."""

    def test_creation(self):
        """PlotWidget can be instantiated."""
        from gui.plot_widget import PlotWidget
        get_app()
        widget = PlotWidget()
        assert widget is not None

    def test_update_data(self):
        """PlotWidget accepts data updates."""
        from gui.plot_widget import PlotWidget
        get_app()
        widget = PlotWidget()
        # Should not raise
        widget.update_data(
            list(range(100)),
            [float(i % 50) for i in range(100)],
            [0.9 if i % 10 != 0 else 0.1 for i in range(100)],
        )


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestReportDialog:
    """Tests for ReportDialog."""

    def test_creation(self):
        """ReportDialog can be instantiated."""
        from gui.report_dialog import ReportDialog
        from config import Config
        get_app()
        dialog = ReportDialog(Config())
        assert dialog is not None
