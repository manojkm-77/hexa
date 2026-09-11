# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FSOC (Free-Space Optical Communication) coarse-alignment simulator for ISRO. Generates a virtual environment with a moving optical beacon, renders it through a simulated pan-tilt camera, detects and tracks the beacon, and commands the camera to keep it centered — all without physical hardware.

The full product specification is in `src/PRD_full.md`.

## Commands

- **Install dependencies:** `uv sync`
- **Run the simulator:** `uv run python src/main.py` — launches an OpenCV window with keyboard controls. Supports interactive, headless, GUI, evaluation, and multi-target modes.
- **Run module self-tests:** `uv run python src/sim.py` / `uv run python src/detect.py` / `uv run python src/control.py` / `uv run python src/evaluation.py`
- **Run tests:** `uv run pytest` — 222 tests across 15 files (15 skip when PySide6/PyTorch absent)
- **Run GUI:** `uv run python src/main.py --gui` (requires PySide6)
- **Run evaluation:** `uv run python src/main.py --eval --duration 10`

**Note:** The root-level `main.py` is a placeholder ("Hello from hexaverse!"). The actual application entry point is `src/main.py`.

## CLI Flags

```bash
uv run python src/main.py --scenario clean          # Interactive clean scenario
uv run python src/main.py --scenario hard           # Interactive hard scenario
uv run python src/main.py --headless                # No display (auto both scenarios)
uv run python src/main.py --eval --duration 10      # 8-category evaluation matrix
uv run python src/main.py --yaml scenarios/clean.yaml  # YAML config
uv run python src/main.py --multi-target            # Multi-target mode
uv run python src/main.py --gui                     # PySide6 GUI
uv run python src/main.py --detector ai --model path.onnx  # AI detector
uv run python src/main.py --filter alpha_beta       # Alpha-beta filter
```

## Runtime Controls (OpenCV window)

SPACE = start/pause, R = reset, 1 = clean scenario, 2 = hard scenario, G = toggle ground-truth overlay, Q/ESC = quit

## Architecture

```
sim.py  →  detect.py  →  control.py
  │            │              │
  └── renders frames ─────────┘
       and ground truth
```

**`src/sim.py`** — Virtual environment. 5 motion models (circular, sinusoidal, constant-velocity, random-maneuvering, scripted trajectory). 7 disturbance models (platform vibration, sensor noise with optional Poisson, motion blur, atmospheric turbulence, exposure variation, occlusion, false beacons). `FrameRenderer` with optional background stars. `Simulator` and `MultiTargetSimulator`. Ground truth must NEVER be passed to the detector or controller.

**`src/detect.py`** — `BeaconDetector` (grayscale threshold or Otsu, optional HSV color filtering, morphological opening, connected components, shape/persistence filtering). `KalmanFilter2D` and `AlphaBetaFilter`. `Tracker` with 7-state FSM: IDLE → SEARCHING → CANDIDATE_VERIFICATION → ACQUIRING → TRACKING → REACQUIRING → FAILED. `MultiTracker` with nearest-neighbor data association and identity-switch detection.

**`src/control.py`** — `PIDController` (per-axis with rate limiting, deadband, anti-windup). `PanTiltController` (pixel-space error to angular commands). `SearchPattern` (spiral search for SEARCHING state). `IntegratedController` ties tracker + controller together.

**`src/config.py`** — Single `Config` dataclass holding 50+ tunable parameters. All other modules import from here: `from config import Config`.

**`src/main.py`** — Entry point with argparse CLI (12 flags). `make_scenario()` assembles sim+controller+disturbances from config. `run_scenario()` runs the loop with HUD overlay. `CSVLogger` produces per-frame logs. `generate_plots()` produces error-over-time charts.

**`src/evaluation.py`** — 8-category weighted evaluation matrix (nominal, motion, vibration, turbulence, sensor_degradation, clutter, occlusion, multitarget). `MetricsCalculator`, `ScenarioRunner`, `EvaluationMatrix`.

**`src/report.py`** — `SummaryReporter` (JSON summary with all PRD metrics). `ReportGenerator` (self-contained HTML with embedded base64 plots).

**`src/yaml_config.py`** — YAML scenario loader with schema validation. Maps nested YAML sections to flat Config fields. Supports all 4 motion types and v1.1 disturbances.

**`src/ai_detector.py`** — `AIDetector(DetectorBase)` with ONNX Runtime inference and classical detector fallback.

**`src/train_detector.py`** — `BeaconCNN` (3-layer CNN), `BeaconDataset`, `train_model()` with ONNX export.

**`src/gui/`** — PySide6 desktop application: `MainWindow` with `SimulationWorker` (QThread), `ConfigPanel`, `PlotWidget` (matplotlib), `ReportDialog`.

## Key Design Rules

- **Ground-truth isolation:** `GroundTruth` is returned by the simulator but never exposed to the detector or controller. This enforces realistic tracking behavior.
- **All configuration in one place:** Edit `src/config.py` only. Other modules read from `Config` — never hardcode simulation parameters elsewhere.
- **Import style:** Modules import from siblings directly (e.g., `from sim import CameraState`). Source root is `src/` — run from project root or use `uv run`.
- **Scenarios:** "clean" = circular trajectory, no disturbances. "hard" = sinusoidal with vibration, noise, and blur. New scenarios go in `make_scenario()` in `src/main.py` or via YAML files in `scenarios/`.
- **Disturbance application:** v1.1 disturbances (turbulence, exposure, occlusion, false beacons) are applied in the run loop BETWEEN `sim.step()` and `controller.update()`, never monkey-patched onto the simulator.

## Agent Team

Six specialized agents in `.claude/agents/`, each owning a domain from the PRD:

| Agent | Owns | Key File(s) |
|-------|------|-------------|
| `sim-engineer` | Virtual environment, camera, motion models, disturbances, rendering | `src/sim.py` |
| `vision-engineer` | Beacon detection, Kalman filter, tracking FSM, multi-tracker | `src/detect.py` |
| `control-engineer` | PID controller, pan-tilt control, search patterns | `src/control.py` |
| `systems-engineer` | Main loop, scenarios, HUD, logging, configuration | `src/main.py`, `src/config.py` |
| `ai-engineer` | AI detector, training data, ONNX export, benchmarking | `src/ai_detector.py`, `src/train_detector.py` |
| `qa-engineer` | Testing, evaluation framework, metrics, acceptance criteria | `tests/`, `src/evaluation.py` |
| `architect` | System architecture, module interfaces, integration, code review | Cross-cutting |
| `gui-engineer` | PySide6 desktop GUI, real-time display, config controls, report export | `src/gui/` |

Use agents via the Agent tool with `subagent_type` set to the agent name (e.g., `sim-engineer`).

## Tech Stack

- Python 3.12 (managed by `uv`, see `.python-version`)
- OpenCV (`cv2`) for rendering and image processing
- NumPy for math
- Matplotlib (Agg backend) for post-run plots
- PyYAML for scenario configuration
- PySide6 (optional) for desktop GUI
- ONNX Runtime (optional) for AI detector inference
- PyTorch (optional) for CNN training pipeline
