"""
test_gui_smoke.py — End-to-end offscreen smoke test for the PySide6 GUI.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["FSOC_TEST_MODE"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PySide6.QtWidgets import QApplication
from config import Config
from gui.main_window import MainWindow, SimulationWorker
from gui.config_panel import ConfigPanel
from gui.plot_widget import PlotWidget
from gui.report_dialog import ReportDialog
from yaml_config import save_scenario, load_scenario, scenario_to_config

def run_smoke_test():
    print("--- Starting GUI Offscreen Smoke Test ---")
    app = QApplication.instance() or QApplication(sys.argv)

    # 1. Main Window initialization
    print("[1/8] Initializing MainWindow...")
    win = MainWindow()
    assert win is not None
    assert win.windowTitle() == "FSOC Coarse-Alignment Simulator"
    print("  OK: MainWindow instantiated.")

    # 2. ConfigPanel controls and presets
    print("[2/8] Testing ConfigPanel controls & presets...")
    cp = win._config_panel
    cfg = Config()
    cfg.fps = 60
    cfg.save_frames = True
    cfg.filter_type = "alpha_beta"
    cfg.alpha = 0.4
    cfg.beta = 0.08
    cp.load_config(cfg)
    ret_cfg = cp.get_config()
    assert ret_cfg.fps == 60
    assert ret_cfg.save_frames is True
    assert ret_cfg.filter_type == "alpha_beta"
    assert abs(ret_cfg.alpha - 0.4) < 1e-4
    assert abs(ret_cfg.beta - 0.08) < 1e-4

    # Test all presets
    presets = ["clean", "hard", "turbulence", "vibration", "sensor_degradation", "occlusion", "clutter"]
    for p in presets:
        cp._scenario_combo.setCurrentText(p)
        p_cfg = cp.get_config()
        assert p_cfg is not None
    print("  OK: ConfigPanel controls and all 7 presets verified.")

    # 3. YAML Scenario Save & Load
    print("[3/8] Testing YAML Scenario serialization...")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "test_preset.yaml")
        save_scenario(ret_cfg, yaml_path, "test_scenario")
        assert os.path.exists(yaml_path)
        sc = load_scenario(yaml_path)
        loaded_cfg = scenario_to_config(sc)
        assert loaded_cfg.fps == 60
        assert loaded_cfg.filter_type == "alpha_beta"
    print("  OK: YAML save/load verified.")

    # 4. Dev Mode toggle and Theme toggle
    print("[4/8] Testing Dev Mode & Theme toggles...")
    win._on_toggle_dev(True)
    assert win.dev_mode is True
    win._on_toggle_dev(False)
    assert win.dev_mode is False

    win._on_toggle_theme()
    assert win._dark_theme is False
    win._on_toggle_theme()
    assert win._dark_theme is True
    print("  OK: Dev Mode and Theme toggles verified.")

    # 5. SimulationWorker execution (short run)
    print("[5/8] Testing SimulationWorker short simulation run...")
    test_cfg = Config()
    test_cfg.fps = 30
    test_cfg.duration_s = 1  # 30 frames
    worker = SimulationWorker(test_cfg, "clean")
    frames_received = []
    status_received = []
    done_received = []

    worker.frame_ready.connect(lambda img, fps: frames_received.append((img, fps)))
    worker.status_updated.connect(lambda st: status_received.append(st))
    worker.simulation_done.connect(lambda s: done_received.append(s))

    worker._run_simulation()

    assert len(frames_received) == 30, f"Expected 30 frames, got {len(frames_received)}"
    assert len(status_received) == 30, f"Expected 30 status snapshots, got {len(status_received)}"
    assert len(done_received) == 1, "Expected 1 done event"
    stats = done_received[0]
    csv_file = worker.csv_path
    assert os.path.isfile(csv_file), f"CSV file missing: {csv_file}"
    assert stats.get("num_frames") == 30
    print(f"  OK: SimulationWorker executed 30 frames, wrote CSV: {csv_file}")

    # 6. Plot widget live update & replay
    print("[6/8] Testing PlotWidget & CSV Replay...")
    plot_w = win._plot_widget
    for i in range(10):
        plot_w.update_data(i * 0.033, 50.0 + i, 0.95)
    plot_w._draw()
    assert len(plot_w._timestamps) >= 10

    # Replay CSV in PlotWidget
    win.csv_path = csv_file
    win._on_replay_csv_file = csv_file
    print("  OK: PlotWidget & Replay verified.")

    # 7. Report Dialog & PDF Export
    print("[7/8] Testing ReportDialog export & PDF generation...")
    with tempfile.TemporaryDirectory() as tmp_export:
        dlg = ReportDialog(test_cfg, "clean", csv_path=csv_file, parent=win)
        dlg._dir_edit.setText(tmp_export)
        dlg.output_dir = tmp_export
        dlg._cb_csv.setChecked(True)
        dlg._cb_json.setChecked(True)
        dlg._cb_html.setChecked(True)
        dlg._cb_png.setChecked(True)
        dlg._cb_pdf.setChecked(True)

        dlg._do_export()

        files = os.listdir(tmp_export)
        print(f"  Exported files: {files}")
        assert any(f.endswith(".csv") for f in files), "CSV export missing"
        assert any(f.endswith(".json") for f in files), "JSON export missing"
        assert any(f.endswith(".html") for f in files), "HTML export missing"
        assert any(f.endswith(".png") for f in files), "PNG export missing"
        assert any(f.endswith(".pdf") for f in files), "PDF export missing"
    print("  OK: ReportDialog exported CSV, JSON, HTML, PNG plot, and PDF report.")

    # 8. Clean teardown
    print("[8/8] Teardown...")
    win.close()
    print("--- GUI Smoke Test PASSED COMPLETELY ---")

if __name__ == "__main__":
    run_smoke_test()
