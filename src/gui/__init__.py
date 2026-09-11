"""
gui package — PySide6 GUI for the FSOC coarse-alignment simulator.

Provides a full graphical interface with live video, real-time plots,
configuration controls, and report export.

Usage:
    from gui import MainWindow
    # or
    from gui.main_window import MainWindow
"""

try:
    from gui.main_window import MainWindow
except ImportError:
    # PySide6 not installed — provide a helpful fallback
    MainWindow = None
    import sys
    print(
        "\n  [GUI unavailable] PySide6 is not installed.\n"
        "  Install it with:  uv pip install PySide6>=6.5\n"
        "  Or:  uv sync --extra gui\n",
        file=sys.stderr,
    )
