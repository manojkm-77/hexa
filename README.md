# FSOC Virtual Camera Tracking System

A software-in-the-loop coarse-alignment simulator for Free-Space Optical Communication (FSOC) terminals. Generates a moving optical beacon in a virtual environment, renders the scene through a simulated pan-tilt camera, detects and tracks the beacon using classical computer vision, and continuously commands the camera to keep it centered -- all without physical hardware.

## Features

- **Virtual environment** -- Configurable camera model (FOV, pan/tilt limits, rate limits) with Gaussian beacon rendering
- **Motion models** -- Circular, sinusoidal, constant-velocity, and random-maneuvering trajectories
- **Beacon detection** -- Grayscale thresholding, morphological filtering, connected-component analysis, centroid extraction
- **Kalman tracking** -- 4-state filter (x, y, vx, vy) with measurement smoothing and gap prediction
- **Finite state machine** -- SEARCHING -> ACQUIRING -> TRACKING -> REACQUIRING with configurable thresholds
- **PID control** -- Per-axis proportional-integral-derivative controller with rate limiting, deadband, and anti-windup
- **Disturbances** -- Platform vibration, sensor noise, and motion blur, independently configurable
- **Scenarios** -- "clean" (ideal) and "hard" (realistic disturbances) out of the box
- **Logging** -- Per-frame CSV logs with ground truth, estimates, errors, and tracker state
- **Post-run plots** -- Error-over-time charts with state coloring and detection traces
- **Dev mode** -- Ground-truth overlay for debugging (press G)

## Quick Start

### Prerequisites

- Python 3.12 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```sh
uv sync
```

### Run

```sh
uv run python src/main.py
```

An OpenCV window opens with the virtual camera view and a heads-up display.

### Controls

| Key   | Action                         |
|-------|--------------------------------|
| SPACE | Start / pause                  |
| R     | Reset simulation               |
| 1     | Switch to clean scenario       |
| 2     | Switch to hard scenario        |
| G     | Toggle ground-truth overlay    |
| Q/ESC | Quit                           |

### Headless Mode

When no display is available (e.g., SSH), the simulator runs both scenarios automatically and prints a comparison table. Set `FSOC_TEST_MODE=1` to shorten the duration to 10 seconds.

## Architecture

Four modules under `src/` with strict information-flow rules:

```
sim.py  -->  detect.py  -->  control.py
  |              |               |
  +-- renders frames + ground truth
```

| Module | Role | Key Classes |
|--------|------|-------------|
| `src/sim.py` | Virtual environment, camera, target motion, disturbances, rendering | `CameraState`, `TargetState`, `Simulator`, `GroundTruth` |
| `src/detect.py` | Beacon detection, Kalman filter, tracking FSM | `BeaconDetector`, `KalmanFilter2D`, `Tracker` |
| `src/control.py` | PID controller, pan-tilt command generation | `PIDController`, `PanTiltController`, `IntegratedController` |
| `src/config.py` | Central configuration dataclass | `Config` |
| `src/main.py` | Entry point, scenario assembly, display, logging, plots | `make_scenario()`, `run_scenario()`, `CSVLogger` |

**Ground-truth isolation** is a core design rule. `GroundTruth` is produced by the simulator but never passed to the detector or controller. This enforces realistic tracking behavior -- the system can only work with what it "sees" through the virtual camera, not with knowledge of where the beacon actually is.

## Scenarios

### Clean

- Circular beacon trajectory, no disturbances
- Demonstrates ideal tracking performance
- Useful as a baseline for controller tuning

### Hard

- Sinusoidal beacon trajectory (azimuth + elevation)
- Platform vibration (RMS 0.2 deg)
- Sensor noise (sigma 12)
- Motion blur (3 px)
- Demonstrates robustness under realistic conditions

New scenarios are added by extending `make_scenario()` in `src/main.py` and adding corresponding parameters in `src/config.py`.

## Configuration

All tunable parameters live in the `Config` dataclass in `src/config.py`. No other module hardcodes simulation values.

Key parameter groups:

| Group | Parameters | Typical Adjustments |
|-------|-----------|-------------------|
| Simulation | `fps`, `duration_s`, `random_seed` | Lower `fps` for faster test runs |
| Camera | `width`, `height`, `h_fov_deg`, `v_fov_deg`, `rate_limit_deg_s` | Narrower FOV makes tracking harder |
| Beacon | `beacon_sigma_px`, `beacon_peak` | Smaller sigma = harder to detect |
| Detector | `threshold`, `min_area`, `max_area` | Lower threshold catches dimmer beacons |
| Kalman | `measurement_noise` | Higher = trusts detections less, smoother predictions |
| PID | `kp`, `ki`, `kd`, `deadband_pixels` | Start with `kp=1, ki=0, kd=0.15`; add `ki` for steady-state offset |
| FSM | `acquire_threshold`, `lose_threshold`, `reacquire_timeout` | Lower acquire = faster lock; higher lose = more resilient |
| Scenarios | `clean_*`, `hard_*` | Increase disturbance amplitudes for stress testing |

## Testing

### Module Self-Tests

Each module has self-test functions in its `if __name__ == "__main__"` block:

```sh
uv run python src/sim.py
uv run python src/detect.py
uv run python src/control.py
```

### pytest

```sh
uv run pytest
```

Tests are in `tests/`. Currently minimal (`tests/test_sim.py` is a placeholder).

## Project Structure

```
src/
  config.py              # All tunable parameters (Config dataclass)
  sim.py                 # Virtual environment, camera, motion models, disturbances
  detect.py              # Beacon detector, Kalman filter, tracking FSM
  control.py             # PID controller, pan-tilt control
  main.py                # Entry point, scenarios, display, logging, plots
  PRD_full.md            # Full product requirements document
  sample_clean_log.csv

tests/
  test_sim.py            # Test placeholder
```

## Tech Stack

- **Python 3.12** -- managed by uv (see `.python-version`)
- **OpenCV** (`cv2`) -- rendering, image processing, display
- **NumPy** -- math, array operations
- **Matplotlib** (Agg backend) -- post-run performance plots

## License

MIT License. See [LICENSE](LICENSE) for details.
