# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FSOC (Free-Space Optical Communication) coarse-alignment simulator. Generates a virtual environment with a moving optical beacon, renders it through a simulated pan-tilt camera, detects and tracks the beacon, and commands the camera to keep it centered — all without physical hardware.

The full product specification is in `src/PRD_full.md`.

## Commands

- **Install dependencies:** `uv sync`
- **Run the simulator:** `uv run python src/main.py` — launches an OpenCV window with keyboard controls. Supports both interactive (GUI) and headless (batch) modes.
- **Run module self-tests:** `uv run python src/sim.py` / `uv run python src/detect.py` / `uv run python src/control.py` — each module has `_test_*()` functions in its `if __name__ == "__main__"` block
- **Run tests:** `uv run pytest` (tests are in `tests/`, currently minimal — `tests/test_sim.py` exists but is empty)

**Note:** The root-level `main.py` is a placeholder ("Hello from hexaverse!"). The actual application entry point is `src/main.py`.

## Runtime Controls (src/main.py)

SPACE = start/pause, R = reset, 1 = clean scenario, 2 = hard scenario, G = toggle ground-truth overlay, Q/ESC = quit

## Architecture

Four modules under `src/` with strict information-flow rules:

```
sim.py  →  detect.py  →  control.py
  │            │              │
  └── renders frames ─────────┘
       and ground truth
```

**`src/sim.py`** — Virtual environment. Defines `CameraState`, `TargetState`, `GroundTruth`, and motion models (`CircularMotion`, `SinusoidalMotion`, `ConstantVelocity`, `RandomManeuvering`). Also disturbance models (`PlatformVibration`, `SensorNoise`, `MotionBlur`). The `Simulator` class renders synthetic frames and returns ground truth alongside each frame. Ground truth must NEVER be passed to the detector or controller.

**`src/detect.py`** — Beacon detection pipeline: grayscale → threshold → morphological opening → connected components → centroid selection. `BeaconDetector` operates on raw frames only. `KalmanFilter2D` smooths detections and predicts through gaps (state: [x, y, vx, vy]). `Tracker` wraps both with a finite state machine: SEARCHING → ACQUIRING → TRACKING → REACQUIRING.

**`src/control.py`** — `PIDController` (per-axis with rate limiting, deadband, anti-windup) and `PanTiltController` (converts pixel-space error to angular pan/tilt commands). `IntegratedController` ties the tracker and pan-tilt controller together.

**`src/config.py`** — Single `Config` dataclass holding all tunable parameters (simulation, camera, beacon, detector, Kalman, PID, FSM thresholds, scenario settings). All other modules import from here: `from config import Config`.

**`src/main.py`** — Entry point. Assembles scenarios, runs the main loop, handles display/logging/plots. Produces CSV logs under `logs/` with per-frame metrics.

**`tests/test_sim.py`** — Empty placeholder for tests.

## Key Design Rules

- **Ground-truth isolation:** `GroundTruth` is returned by the simulator but never exposed to the detector or controller. This enforces realistic tracking behavior.
- **All configuration in one place:** Edit `src/config.py` only. Other modules read from `Config` — never hardcode simulation parameters elsewhere.
- **Import style:** Modules import from siblings directly (e.g., `from sim import CameraState`). Source root is `src/` — run from project root or use `uv run`.
- **Scenarios:** "clean" = circular trajectory, no disturbances. "hard" = sinusoidal with vibration, noise, and blur. New scenarios go in `make_scenario()` in `src/main.py`.

## Tech Stack

- Python 3.12 (managed by `uv`, see `.python-version`)
- OpenCV (`cv2`) for rendering and image processing
- NumPy for math
- Matplotlib (Agg backend) for post-run plots
- FastAPI + Pydantic in dependencies (not yet used in the main simulator)
