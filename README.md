# FSOC Coarse-Alignment Simulator

A production-quality software-in-the-loop simulator for Free-Space Optical Communication (FSOC) terminal coarse alignment, built for ISRO's FSOC research program. Generates a virtual environment with a moving optical beacon, renders it through a simulated pan-tilt camera, detects and tracks the beacon, and commands the camera to keep it centered — all without physical hardware.

## Features

### Core Pipeline

- **Virtual environment** — Configurable camera model (FOV, pan/tilt limits, rate limits) with Gaussian beacon rendering on a gradient sky background
- **5 motion models** — Circular, sinusoidal, constant-velocity, random-maneuvering, and scripted trajectory replay
- **7 disturbance types** — Platform vibration, sensor noise (Gaussian + Poisson), motion blur, atmospheric turbulence (Ornstein-Uhlenbeck), exposure variation, occlusion events, and false beacon generation
- **Beacon detection** — Grayscale thresholding (fixed or Otsu), optional HSV color filtering, morphological filtering, connected-component analysis, shape filtering, persistence tracking, and centroid extraction
- **Kalman tracking** — 4-state constant-velocity filter (x, y, vx, vy) with measurement smoothing and gap prediction
- **Alpha-beta filter** — Simpler alternative to Kalman for less demanding scenarios
- **7-state tracking FSM** — IDLE → SEARCHING → CANDIDATE_VERIFICATION → ACQUIRING → TRACKING → REACQUIRING → FAILED
- **PID control** — Per-axis proportional-integral-derivative controller with rate limiting, deadband, and anti-windup
- **Search patterns** — Spiral search pattern for beacon acquisition during SEARCHING state

### Multi-Target & AI

- **Multi-target tracking** — Nearest-neighbor data association with identity-switch detection
- **AI detector** — ONNX Runtime inference with classical detector fallback via `DetectorBase` ABC
- **Training pipeline** — Synthetic data generation, lightweight CNN training, ONNX export

### Evaluation & Reporting

- **8-category evaluation matrix** — Weighted scoring across nominal, motion, vibration, turbulence, sensor degradation, clutter, occlusion, and multi-target scenarios
- **Per-frame CSV logging** — Ground truth, estimates, errors, tracker state, processing time
- **JSON summary reports** — All PRD metrics computed automatically
- **Self-contained HTML reports** — Embedded plots (base64), configuration, metrics tables
- **PNG plots** — Error-over-time charts with state coloring and detection traces

### PySide6 Desktop GUI

- **Live video feed** — Real-time rendering with HUD overlay and dynamic state-colored border
- **Status dashboard** — Tracker state badge (7 FSM states), pan/tilt readouts, pixel/angular errors, confidence, streaks, FPS, latency
- **Configuration tab** — Presets (7 scenarios), PID tuning, detector/filter choice (Kalman + Alpha-Beta), disturbance sliders, multi-target controls, frame saving
- **Live plotting tab** — Real-time `pyqtgraph` charts with tracking error, control effort, and FPS timeline
- **Report export tab** — Export CSV, JSON, HTML, PNG, and multi-page PDF reports + CSV replay
- **Theme toggle** — Dark and light presentation themes (`T` / `Ctrl+T`)

### Infrastructure

- **argparse CLI** — 12 command-line flags for all modes
- **YAML scenario configuration** — Schema-validated, fully mapped to Config
- **Headless mode** — Batch execution without display
- **Evaluation mode** — Automated multi-seed, multi-category evaluation
- **PyInstaller packaging** — Standalone executable via `build.spec`

## Quick Start

### Prerequisites

- Python 3.12 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```sh
uv sync
```

### Run (Interactive)

```sh
uv run python src/main.py
```

An OpenCV window opens with the virtual camera view and heads-up display.

### Controls

| Key   | Action                         |
|-------|--------------------------------|
| SPACE | Start / pause                  |
| R     | Reset simulation               |
| 1     | Switch to clean scenario       |
| 2     | Switch to hard scenario        |
| G     | Toggle ground-truth overlay    |
| Q/ESC | Quit                           |

## CLI Reference

```sh
# Interactive clean scenario
uv run python src/main.py --scenario clean

# Headless run (no display, produces CSV + JSON + HTML)
uv run python src/main.py --headless --scenario clean

# Evaluation framework (8-category matrix, multi-seed)
uv run python src/main.py --eval --duration 10

# Load YAML scenario configuration
uv run python src/main.py --yaml scenarios/clean.yaml

# Multi-target mode
uv run python src/main.py --multi-target

# PySide6 GUI
uv run python src/main.py --gui

# AI detector with ONNX model
uv run python src/main.py --detector ai --model models/beacon.onnx

# Alpha-beta filter instead of Kalman
uv run python src/main.py --filter alpha_beta

# Custom output directory and duration
uv run python src/main.py --output results/my_run --duration 30
```

### All CLI Flags

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--scenario` | `clean`, `hard` | `clean` | Scenario to run |
| `--yaml` | path | — | YAML scenario config file |
| `--eval` | flag | — | Run evaluation framework |
| `--headless` | flag | — | Run without display |
| `--gui` | flag | — | Launch PySide6 GUI |
| `--multi-target` | flag | — | Multi-target simulation |
| `--detector` | `classical`, `ai` | `classical` | Detector type |
| `--filter` | `kalman`, `alpha_beta` | `kalman` | State estimation filter |
| `--model` | path | — | ONNX model for AI detector |
| `--output` | dir | `results` | Output directory |
| `--duration` | int | 60 | Duration override (seconds) |
| `--seeds` | CSV | 42,123,456,789,1024 | Seeds for eval mode |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   sim.py    │────▶│  detect.py   │────▶│  control.py    │
│             │     │              │     │                │
│ CameraState │     │ BeaconDetect │     │ PIDController  │
│ MotionModel │     │ KalmanFilter │     │ PanTiltControl │
│ Disturbances│     │ Tracker FSM  │     │ SearchPattern  │
│ Simulator   │     │ MultiTracker │     │ IntegratedCtrl │
│ FrameRender │     │ AlphaBeta    │     └────────────────┘
└─────────────┘     └──────────────┘            │
       │                                         │
       └─── renders frames + ground truth ───────┘
                        │
              ┌─────────▼──────────┐
              │     main.py        │
              │  CLI, scenarios,   │
              │  HUD, logging,     │
              │  plots, reports    │
              └────────────────────┘
```

### Module Reference

| Module | Role | Key Classes |
|--------|------|-------------|
| `src/sim.py` | Virtual environment, camera, 5 motion models, 7 disturbances, rendering, multi-target | `CameraState`, `TargetState`, `Simulator`, `MultiTargetSimulator`, `BeaconConfig` |
| `src/detect.py` | Beacon detection (classical + HSV + shape filtering), Kalman/alpha-beta filters, 7-state FSM, multi-target tracking | `BeaconDetector`, `KalmanFilter2D`, `AlphaBetaFilter`, `Tracker`, `MultiTracker` |
| `src/control.py` | PID controller, pan-tilt control, spiral search pattern | `PIDController`, `PanTiltController`, `SearchPattern`, `IntegratedController` |
| `src/config.py` | Central configuration dataclass (50+ parameters) | `Config` |
| `src/main.py` | Entry point, scenario assembly, HUD overlay, CSV logging, plot generation, argparse CLI | `make_scenario()`, `run_scenario()`, `CSVLogger` |
| `src/evaluation.py` | 8-category evaluation matrix, metrics calculator, scenario runner | `MetricsCalculator`, `ScenarioRunner`, `EvaluationMatrix` |
| `src/report.py` | JSON summary reports, self-contained HTML reports | `SummaryReporter`, `ReportGenerator` |
| `src/yaml_config.py` | YAML scenario loader with schema validation | `load_scenario()`, `scenario_to_config()` |
| `src/ai_detector.py` | ONNX Runtime AI detector with classical fallback | `AIDetector` |
| `src/train_detector.py` | CNN training pipeline, ONNX export | `BeaconCNN`, `BeaconDataset`, `train_model()` |
| `src/gui/` | PySide6 desktop application | `MainWindow`, `SimulationWorker`, `ConfigPanel`, `PlotWidget`, `ReportDialog` |

**Ground-truth isolation** is a core design rule. `GroundTruth` is produced by the simulator but never passed to the detector or controller. This enforces realistic tracking behavior — the system can only work with what it "sees" through the virtual camera.

## Scenarios

### Clean

- Circular beacon trajectory, no disturbances
- Demonstrates ideal tracking performance
- Baseline for controller tuning

### Hard

- Sinusoidal beacon trajectory (azimuth + elevation)
- Platform vibration (RMS 0.2 deg), sensor noise (sigma 12), motion blur (3 px)
- Demonstrates robustness under realistic conditions

### YAML Configuration

Scenarios can also be defined via YAML files under `scenarios/`:

```yaml
simulation:
  fps: 30
  duration_s: 60
camera:
  width: 1280
  height: 720
  h_fov_deg: 60.0
motion:
  type: circular
  center_az_deg: 5.0
  radius_deg: 8.0
disturbances:
  vibration_rms_deg: 0.2
  turbulence_rms_deg: 0.1
detector:
  threshold: 80
controller:
  kp: 1.0
  ki: 0.0
  kd: 0.15
```

## Configuration

All tunable parameters live in the `Config` dataclass in `src/config.py`. No other module hardcodes simulation values.

| Group | Key Parameters | Description |
|-------|---------------|-------------|
| Simulation | `fps`, `duration_s`, `random_seed` | Frame rate, duration, reproducibility |
| Camera | `width`, `height`, `h_fov_deg`, `v_fov_deg` | Resolution and field of view |
| Beacon | `beacon_sigma_px`, `beacon_peak` | Beacon size and brightness |
| Detector | `detector_type`, `threshold`, `use_hsv` | Classical vs AI, threshold, color filtering |
| Filter | `filter_type`, `measurement_noise`, `alpha`, `beta` | Kalman vs alpha-beta |
| PID | `kp`, `ki`, `kd`, `deadband_pixels` | Controller gains |
| FSM | `acquire_threshold`, `lose_threshold`, `reacquire_timeout` | State machine timing |
| Disturbances | `turbulence_rms_deg`, `exposure_rate_hz`, `occlusion_duration_s`, `false_beacon_count` | v1.1 disturbance magnitudes |
| Multi-target | `multi_target`, `num_targets`, `target_colors` | Multi-beacon tracking |

## Testing

### Module Self-Tests

Each module has self-test functions in its `if __name__ == "__main__"` block:

```sh
uv run python src/sim.py
uv run python src/detect.py
uv run python src/control.py
uv run python src/evaluation.py
```

### pytest

```sh
uv run pytest
```

15 test files with 200+ test cases covering:

| Test File | Coverage |
|-----------|----------|
| `test_sim.py` | Projection math, motion models, disturbances, simulator step |
| `test_detect.py` | Beacon detector, Kalman filter, tracker FSM |
| `test_control.py` | PID convergence, rate limiting, deadband, closed-loop |
| `test_yaml_config.py` | YAML loading, validation, both scenarios |
| `test_report.py` | Summary reporter, HTML report generation |
| `test_evaluation.py` | Metrics calculator, evaluation matrix, weights |
| `test_ai_detector.py` | AI detector fallback, ONNX mock, preprocessing |
| `test_detector_base.py` | DetectorBase ABC, AlphaBetaFilter |
| `test_multi_target.py` | Multi-target simulator, multi-tracker, identity switches |
| `test_main_v11.py` | v1.1 disturbance integration in make_scenario() |
| `test_scripted_trajectory.py` | ScriptedTrajectory motion model |
| `test_train_detector.py` | BeaconCNN, dataset (requires PyTorch) |
| `test_config_v11.py` | Config v1.1 field presence and defaults |
| `test_gui.py` | GUI component instantiation (requires PySide6) |

## Project Structure

```
hexaverse/
├── src/
│   ├── main.py                # Entry point, CLI, scenarios, HUD, logging
│   ├── config.py              # All tunable parameters (Config dataclass)
│   ├── sim.py                 # Virtual environment, 5 motion models, 7 disturbances
│   ├── detect.py              # Detector, Kalman/alpha-beta, 7-state FSM, multi-tracker
│   ├── control.py             # PID controller, pan-tilt control, search pattern
│   ├── evaluation.py          # 8-category evaluation matrix
│   ├── report.py              # JSON + HTML report generation
│   ├── yaml_config.py         # YAML scenario loader
│   ├── ai_detector.py         # ONNX AI detector with fallback
│   ├── train_detector.py      # CNN training pipeline
│   ├── generate_training_data.py  # Synthetic training data
│   ├── gui/                   # PySide6 desktop application
│   │   ├── __init__.py
│   │   ├── main_window.py     # MainWindow + SimulationWorker
│   │   ├── config_panel.py    # Configuration controls
│   │   ├── plot_widget.py     # Live matplotlib plots
│   │   └── report_dialog.py   # Report export dialog
│   ├── PRD_full.md            # Full product requirements document
│   └── sample_clean_log.csv   # Sample CSV for testing
├── tests/                     # 15 test files, 200+ tests
├── scenarios/                 # YAML scenario configs
│   ├── clean.yaml
│   └── hard.yaml
├── docs/
│   ├── technical_report.md    # Technical report (architecture, evaluation, results)
│   └── user_manual.md         # Complete user manual
├── build.spec                 # PyInstaller packaging spec
├── pyproject.toml             # Project metadata and dependencies
├── CLAUDE.md                  # Claude Code agent guidance
└── .python-version            # Python 3.12
```

## Tech Stack

- **Python 3.12** — managed by uv (see `.python-version`)
- **OpenCV** (`cv2`) — rendering, image processing, display
- **NumPy** — math, array operations
- **Matplotlib** (Agg backend) — post-run performance plots
- **PyYAML** — scenario configuration loading
- **PySide6** (optional) — desktop GUI
- **ONNX Runtime** (optional) — AI detector inference
- **PyTorch** (optional) — CNN training pipeline

### Optional Dependencies

```sh
# GUI support
uv sync --extra gui

# AI detector + training
uv sync --extra ai

# Everything
uv sync --extra full
```

## Packaging

Build a standalone executable with PyInstaller:

```sh
uv run pyinstaller build.spec
```

The executable will be in `dist/hexaverse/`.

## License

MIT License. See [LICENSE](LICENSE) for details.
