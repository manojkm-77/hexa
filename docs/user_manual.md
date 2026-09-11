# FSOC Coarse-Alignment Simulator — User Manual

**Version 1.1**
**Last updated: 2026-09-09**

---

## Table of Contents

1. [Installation](#1-installation)
2. [System Requirements](#2-system-requirements)
3. [Application Startup](#3-application-startup)
4. [GUI Controls](#4-gui-controls)
5. [Configuration Parameters](#5-configuration-parameters)
6. [Scenario Creation](#6-scenario-creation)
7. [Detector Selection](#7-detector-selection)
8. [Disturbance Settings](#8-disturbance-settings)
9. [Experiment Execution](#9-experiment-execution)
10. [Log Export](#10-log-export)
11. [Report Interpretation](#11-report-interpretation)
12. [Troubleshooting](#12-troubleshooting)
13. [Known Limitations](#13-known-limitations)
14. [Complete Example Experiment](#14-complete-example-experiment)

---

## 1. Installation

### 1.1 Prerequisites

The simulator requires the following before installation:

- **Python 3.12** — managed via `uv`; the exact version is pinned in `.python-version`
- **uv** package manager — install from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Git** — for cloning the repository

### 1.2 Clone the Repository

```sh
git clone https://github.com/your-org/hexaverse.git
cd hexaverse
```

### 1.3 Install Dependencies

```sh
uv sync
```

This reads `pyproject.toml`, creates a virtual environment under `.venv/`, and installs all required packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, vectorized operations |
| `opencv-python` (`cv2`) | Frame rendering, image processing, display |
| `matplotlib` | Post-run performance plots (Agg backend) |
| `pyyaml` | YAML scenario config parsing |
| `pyside6` | Optional GUI mode (`--gui`) |

To verify the installation:

```sh
uv run python src/config.py
```

If this prints the `Config` dataclass without errors, dependencies are installed correctly.

### 1.4 Optional: AI Detector Dependencies

The AI detector (`--detector ai`) requires an ONNX Runtime model. If unavailable, the system falls back to the classical detector automatically. See [Section 7: Detector Selection](#7-detector-selection).

---

## 2. System Requirements

### 2.1 Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 10+, macOS 12+, or Linux (X11/Wayland) |
| Python | 3.12 (pinned in `.python-version`) |
| RAM | 4 GB |
| Disk | 200 MB for installation; ~500 MB per 60-second run (frames + logs) |
| Display | 1280x720 or larger (for GUI mode) |
| GPU | Not required |

### 2.2 Recommended

| Component | Recommendation |
|-----------|----------------|
| RAM | 8 GB (for multi-target and long-duration runs) |
| CPU | Multi-core (frame processing is single-threaded but logging/plots benefit) |
| SSD | Significantly reduces frame-save I/O |

### 2.3 Headless / Remote Environments

When running over SSH or in a container without a display, use `--headless` mode. The simulator detects missing `$DISPLAY` automatically and falls back to headless operation. In headless mode, no OpenCV windows are created; frames are saved to disk and results are printed to stdout.

---

## 3. Application Startup

### 3.1 Standard Launch

```sh
uv run python src/main.py
```

Opens an OpenCV window titled "FSOC Virtual Camera Tracking" with the clean scenario (circular beacon trajectory, no disturbances). Press SPACE to start.

### 3.2 Command-Line Options

| Flag | Description |
|------|-------------|
| `--scenario {clean,hard}` | Select the built-in scenario (default: `clean`) |
| `--yaml PATH` | Load configuration from a YAML scenario file (see [Section 6](#6-scenario-creation)) |
| `--headless` | Run without display; auto-detected when `$DISPLAY` is unset |
| `--eval` | Run the evaluation framework in headless mode (8-category matrix) |
| `--gui` | Launch the PySide6 desktop GUI |
| `--multi-target` | Enable multi-target tracking mode |
| `--detector {classical,ai}` | Select detector backend (default: `classical`) |
| `--filter {kalman,alpha_beta}` | Override the state estimation filter |
| `--model PATH` | Path to ONNX model file for the AI detector |
| `--output DIR` | Output directory for logs, plots, and reports (default: `results/`) |
| `--duration SECONDS` | Override the simulation duration |
| `--seeds SEEDS` | Comma-separated random seeds for evaluation mode |

### 3.3 Common Startup Examples

Run the clean scenario interactively (default):

```sh
uv run python src/main.py
```

Run the hard scenario headless, saving to a custom directory:

```sh
uv run python src/main.py --scenario hard --headless --output runs/hard_test
```

Load a custom YAML configuration:

```sh
uv run python src/main.py --yaml scenarios/clean.yaml
```

Run a 30-second evaluation with two specific seeds:

```sh
uv run python src/main.py --eval --duration 30 --seeds 42,123
```

Launch the graphical interface:

```sh
uv run python src/main.py --gui
```

### 3.4 Environment Variables

| Variable | Effect |
|----------|--------|
| `FSOC_TEST_MODE=1` | Shortens duration to 10 seconds regardless of config (useful for CI) |
| `DISPLAY` | If unset or empty, headless mode is activated automatically |

---

## 4. GUI Controls

### 4.1 Keyboard Controls (OpenCV Window)

When running in interactive mode (not `--headless`, not `--gui`), the OpenCV display window responds to the following keys:

| Key | Action | Details |
|-----|--------|---------|
| `SPACE` | Start / Pause | Toggles simulation execution. Pauses mid-frame; display dims with "PAUSED" overlay. |
| `R` | Reset | Resets camera pose, tracker state, and frame counter to initial values. Reopens the CSV log. |
| `1` | Clean scenario | Switches to the clean scenario (circular trajectory, no disturbances). |
| `2` | Hard scenario | Switches to the hard scenario (sinusoidal trajectory, vibration, noise, blur). |
| `G` | Dev mode toggle | Shows/hides the ground-truth overlay (red cross at true beacon position). **For debugging only — never use during evaluation.** |
| `Q` or `ESC` | Quit | Exits the application. |

### 4.2 HUD (Heads-Up Display)

The top bar of the display shows real-time information:

| HUD Element | Description |
|-------------|-------------|
| **STATE** | Current tracker FSM state: SEARCHING (yellow), ACQUIRING (yellow), TRACKING (green), REACQUIRING (red) |
| **PAN / TILT** | Current pan and tilt command angles in degrees |
| **ERROR** | Pixel error between estimated and frame center. Color-coded: green (<50 px), yellow (50-150 px), red (>150 px) |
| **ANG** | Angular error decomposed into pan and tilt components |
| **FPS** | Current rendering frames per second |
| **PROC** | Processing time per frame in milliseconds |
| **DET** | Whether the beacon is currently detected (YES/NO) |
| **CONF** | Detection confidence (0.0 to 1.0) |
| **DET_STREAK / MISS** | Consecutive detection count and consecutive miss count |

Visual indicators:
- **Green border** around the frame: system is in TRACKING state
- **Red border** around the frame: system is in REACQUIRING state
- **Green circle**: estimated beacon position
- **Red cross** (dev mode only): ground-truth beacon position
- **Grey crosshair**: frame center (tracking target)
- **Grey rectangle**: central 20% zone (acceptable tracking region)

### 4.3 PySide6 Desktop GUI (`--gui`)

When launched with `--gui`, the application opens a multi-threaded desktop GUI designed for real-time tracking, live diagnostics, interactive tuning, and reporting.

```sh
uv run python src/main.py --gui
# or on Windows PowerShell:
.\.venv\Scripts\python.exe src/main.py --gui
```

#### 4.3.1 GUI Layout & Components

1. **Left Panel — Live Video & Status Dashboard**:
   - **Video Feed (`_VideoLabel`)**: High-framerate rendering of the camera view with HUD overlays, colored state-boundary borders (Green = Tracking, Amber = Searching/Acquiring, Red = Reacquiring), and center alignment crosshairs.
   - **Status Panel**: Displays real-time numerical readouts:
     - **Tracker State**: High-contrast badge displaying any of the 7 FSM states.
     - **Estimated Coordinates**: Sub-pixel `(X, Y)` beacon location.
     - **Confidence & Streaks**: Continuous detection confidence, hit streak, and consecutive misses.
     - **Pan & Tilt Commands**: Commanded pan/tilt angles in degrees.
     - **Tracking Errors**: Dynamic color-coded pixel error (Green `<50px`, Amber `<150px`, Red `>150px`) and angular error.
     - **Performance Metrics**: Live FPS, loop latency in milliseconds, and actuator rate saturation warning.

2. **Right Panel — Tabbed Workspace**:
   - **Configuration Tab (`ConfigPanel`)**:
     - **Preset Selection**: Instant switching between `clean`, `hard`, `turbulence`, `vibration`, `sensor_degradation`, `occlusion`, and `clutter`.
     - **Simulation Settings**: Tunable FPS, run duration (seconds), and frame-saving toggle (`Save frames every 30th`).
     - **PID Controller**: Real-time tuning for proportional gain ($K_p$), integral gain ($K_i$), derivative gain ($K_d$), and deadband pixels.
     - **Detection & Filtering**: Choice between classical blob detection and AI detection; choice between Kalman filtering (with adjustable measurement noise) and Alpha-Beta filtering (with dynamic $\alpha$ and $\beta$ sliders).
     - **Multi-Target Controls**: Toggle multi-beacon tracking with target count and assignment strategy (`round_robin` or `nearest`).
     - **Disturbance Injections**: Independent sliders and toggles for platform vibration, sensor noise, motion blur, atmospheric turbulence, exposure fluctuations, occlusions, and false beacons.
     - **Apply Button**: Emits real-time updates directly into the simulation worker.
   - **Live Plot Tab (`PlotWidget`)**:
     - High-performance `pyqtgraph` canvas streaming real-time error curves, detection confidence timelines, and framerates without blocking the UI thread.
     - Automatically decimates high-frequency data for zero rendering lag.
   - **Report Tab (`ReportDialog`)**:
     - Summary metrics dashboard with one-click export to CSV, JSON, interactive HTML reports, PNG error curves, and multi-page PDF summary documents.
     - **CSV Replay**: Load historical `.csv` logs and replay tracking error timelines directly on the plot canvas.

#### 4.3.2 Keyboard Shortcuts & Menu Actions

| Shortcut | Action | Scope / Context |
|----------|--------|-----------------|
| **`Space`** | Start / Pause simulation | Global toolbar & canvas |
| **`R`** | Reset simulation to initial pose | Global toolbar & canvas |
| **`1`** | Switch to Clean scenario preset | Global |
| **`2`** | Switch to Hard scenario preset | Global |
| **`G`** | Toggle Dev Mode (Ground-Truth overlay) | Global |
| **`T`** / **`Ctrl+T`** | Toggle Dark / Light Theme | Global |
| **`Ctrl+O`** | Load scenario config from YAML | File Menu |
| **`Ctrl+S`** | Save current configuration to YAML | File Menu |
| **`Ctrl+E`** | Open Report Export Dialog | File Menu |
| **`Ctrl+R`** | Replay CSV Log in Plot Widget | File Menu |
| **`F1`** | Show Keyboard Shortcuts Dialog | Help Menu |
| **`Esc`** / **`Ctrl+Q`** | Quit Application | Global |

---

## 5. Configuration Parameters

All parameters are defined in the `Config` dataclass at `src/config.py`. Edit this file directly — no other module hardcodes values. Parameters are organized into the following sections.

### 5.1 Simulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fps` | int | 30 | Frames per second for the simulation loop |
| `duration_s` | int | 60 | Total simulation duration in seconds |
| `random_seed` | int | 42 | Seed for reproducible random motion and disturbances |

### 5.2 Camera

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 1280 | Frame width in pixels |
| `height` | int | 720 | Frame height in pixels |
| `h_fov_deg` | float | 60.0 | Horizontal field of view in degrees |
| `v_fov_deg` | float | 40.0 | Vertical field of view in degrees |
| `pan_range` | tuple | (-180, 180) | Pan angle limits in degrees (min, max) |
| `tilt_range` | tuple | (-30, 90) | Tilt angle limits in degrees (min, max) |
| `rate_limit_deg_s` | float | 60.0 | Maximum angular rate in degrees per second |

### 5.3 Beacon

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beacon_sigma_px` | float | 4.0 | Gaussian sigma for beacon rendering (px). Smaller = harder to detect. |
| `beacon_peak` | float | 255.0 | Peak brightness of the Gaussian beacon (0-255) |

### 5.4 Detector Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detector_type` | str | `"classical"` | `"classical"` or `"ai"` |
| `ai_model_path` | str | `""` | Path to ONNX model file for AI detector |

### 5.5 Classical Detector

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | int | 80 | Fixed brightness threshold for binarization (0-255) |
| `min_area` | int | 2 | Minimum blob area in pixels |
| `max_area` | int | 500 | Maximum blob area in pixels |

### 5.6 Filter Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_type` | str | `"kalman"` | `"kalman"` or `"alpha_beta"` |

### 5.7 Alpha-Beta Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Position smoothing coefficient |
| `beta` | float | 0.1 | Velocity smoothing coefficient |

### 5.8 Kalman Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `measurement_noise` | float | 8.0 | Measurement noise covariance. Higher = trusts detections less, smoother predictions. |

### 5.9 PID Controller

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kp` | float | 1.0 | Proportional gain (start here) |
| `ki` | float | 0.0 | Integral gain (add only if steady-state offset persists) |
| `kd` | float | 0.15 | Derivative gain (damping) |
| `deadband_pixels` | float | 3.0 | Ignore errors below this threshold (pixels) |

### 5.10 Tracker FSM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verify_threshold` | int | 2 | Consecutive detections needed to confirm a candidate |
| `acquire_error_threshold` | float | 100.0 | Pixel error must be below this to enter TRACKING |
| `acquire_threshold` | int | 3 | Consecutive detections to enter TRACKING from ACQUIRING |
| `lose_threshold` | int | 5 | Consecutive misses to trigger REACQUIRING |
| `reacquire_timeout` | int | 150 | Frames before giving up and returning to SEARCHING (~5 seconds) |

### 5.11 Clean Scenario

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clean_motion` | str | `"circular"` | Motion model: `"circular"`, `"sinusoidal"`, `"constant"`, `"random"` |
| `clean_center_az` | float | 5.0 | Center azimuth for circular motion (deg) |
| `clean_center_el` | float | 3.0 | Center elevation for circular motion (deg) |
| `clean_radius` | float | 8.0 | Radius for circular motion (deg) |
| `clean_speed` | float | 4.0 | Angular speed for circular motion (deg/s) |
| `clean_max_acc` | float | 10.0 | Max acceleration for random motion (deg/s^2) |
| `clean_change_interval` | float | 2.0 | Direction change interval for random motion (s) |
| `clean_max_vel` | float | 20.0 | Max velocity for random motion (deg/s) |
| `clean_az_rate` | float | 5.0 | Azimuth rate for constant-velocity motion (deg/s) |
| `clean_el_rate` | float | 2.0 | Elevation rate for constant-velocity motion (deg/s) |
| `clean_vibration_rms` | float | 0.0 | Platform vibration RMS (deg). 0 = disabled. |
| `clean_noise_sigma` | float | 0.0 | Sensor noise sigma. 0 = disabled. |
| `clean_blur_pixels` | float | 0.0 | Motion blur kernel size (px). 0 = disabled. |

### 5.12 Hard Scenario

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hard_motion` | str | `"sinusoidal"` | Motion model for the hard scenario |
| `hard_az_amp` | float | 10.0 | Azimuth sinusoidal amplitude (deg) |
| `hard_az_freq` | float | 0.2 | Azimuth sinusoidal frequency (Hz) |
| `hard_el_amp` | float | 6.0 | Elevation sinusoidal amplitude (deg) |
| `hard_el_freq` | float | 0.3 | Elevation sinusoidal frequency (Hz) |
| `hard_vibration_rms` | float | 0.2 | Platform vibration RMS (deg) |
| `hard_noise_sigma` | float | 12.0 | Sensor noise sigma |
| `hard_blur_pixels` | float | 3.0 | Motion blur kernel size (px) |

### 5.13 v1.1 Disturbances

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `turbulence_rms_deg` | float | 0.1 | Atmospheric turbulence RMS (deg). Set to 0 to disable. |
| `turbulence_correlation_s` | float | 1.0 | Turbulence temporal correlation time (s) |
| `exposure_rate_hz` | float | 0.1 | Exposure variation frequency (Hz). Set to 0 to disable. |
| `exposure_amplitude` | float | 0.3 | Exposure variation amplitude (fractional) |
| `occlusion_duration_s` | float | 1.0 | Occlusion event duration (s). Set to 0 to disable. |
| `occlusion_frequency_hz` | float | 0.05 | Occlusion event frequency (Hz) |
| `false_beacon_count` | int | 2 | Number of false beacons injected. Set to 0 to disable. |
| `false_beacon_brightness_range` | tuple | (80, 180) | Brightness range for false beacons |

### 5.14 Multi-Target

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi_target` | bool | False | Enable multi-target tracking mode |
| `num_targets` | int | 2 | Number of simultaneous beacons |
| `target_colors` | list | [(255,255,255), (0,255,255)] | BGR color for each beacon |

---

## 6. Scenario Creation

### 6.1 YAML Configuration

Custom scenarios are defined in YAML files under the `scenarios/` directory. The simulator loads them via `--yaml scenarios/your_file.yaml`.

### 6.2 YAML Structure

A scenario YAML file maps directly to `Config` fields. The top-level keys correspond to parameter sections:

```yaml
# scenarios/clean.yaml — Clean scenario configuration
simulation:
  fps: 30
  duration_s: 60
  random_seed: 42

camera:
  width: 1280
  height: 720
  h_fov_deg: 60.0
  v_fov_deg: 40.0
  pan_range: [-180.0, 180.0]
  tilt_range: [-30.0, 90.0]
  rate_limit_deg_s: 60.0

beacon:
  sigma_px: 4.0
  peak: 255.0

detector:
  type: classical
  threshold: 80
  min_area: 2
  max_area: 500

filter:
  type: kalman
  measurement_noise: 8.0

pid:
  kp: 1.0
  ki: 0.0
  kd: 0.15
  deadband_pixels: 3.0

fsm:
  acquire_threshold: 3
  lose_threshold: 5
  reacquire_timeout: 150

scenario:
  name: clean
  motion: circular
  center_az: 5.0
  center_el: 3.0
  radius: 8.0
  speed: 4.0
  vibration_rms: 0.0
  noise_sigma: 0.0
  blur_pixels: 0.0
  turbulence_rms_deg: 0.0
  turbulence_correlation_s: 1.0
  exposure_rate_hz: 0.0
  exposure_amplitude: 0.0
  occlusion_duration_s: 0.0
  occlusion_frequency_hz: 0.0
  false_beacon_count: 0
```

### 6.3 YAML Loading Flow

1. `--yaml` flag provides the file path
2. `yaml_config.load_scenario()` parses the YAML into a dictionary
3. `yaml_config.scenario_to_config()` maps dictionary keys to `Config` dataclass fields
4. CLI flags (e.g., `--filter`, `--detector`) override YAML values

### 6.4 Hard Scenario Example

```yaml
# scenarios/hard.yaml — Hard scenario with disturbances
simulation:
  fps: 30
  duration_s: 60
  random_seed: 42

camera:
  width: 1280
  height: 720
  h_fov_deg: 60.0
  v_fov_deg: 40.0

detector:
  type: classical
  threshold: 80

filter:
  type: kalman

pid:
  kp: 1.0
  ki: 0.0
  kd: 0.15
  deadband_pixels: 3.0

scenario:
  name: hard
  motion: sinusoidal
  az_amp: 10.0
  az_freq: 0.2
  el_amp: 6.0
  el_freq: 0.3
  vibration_rms: 0.2
  noise_sigma: 12.0
  blur_pixels: 3.0
  turbulence_rms_deg: 0.1
  turbulence_correlation_s: 1.0
  exposure_rate_hz: 0.1
  exposure_amplitude: 0.3
  occlusion_duration_s: 1.0
  occlusion_frequency_hz: 0.05
  false_beacon_count: 2
```

---

## 7. Detector Selection

### 7.1 Classical Detector

The default detector (`detector_type: "classical"`) uses a classical computer-vision pipeline:

1. Convert frame to grayscale
2. Apply fixed brightness threshold (configurable via `threshold`)
3. Morphological opening to remove noise
4. Connected-component analysis to find blobs
5. Filter blobs by area (`min_area` to `max_area`)
6. Select the brightest/most central blob
7. Compute centroid as the detected position

This detector is deterministic, fast, and requires no trained model.

### 7.2 AI Detector

When `detector_type: "ai"`, the system attempts to load an ONNX neural-network model:

```sh
uv run python src/main.py --detector ai --model path/to/model.onnx
```

The AI detector wraps the classical detector as a fallback. If the ONNX model fails to load (missing file, missing ONNX Runtime, incompatible format), the system prints a warning and falls back to the classical detector:

```
[Warning] AIDetector not available, falling back to classical
```

### 7.3 State Estimation Filters

Regardless of detector choice, the raw detection passes through a state estimation filter:

| Filter | Command | Key Parameters | Behavior |
|--------|---------|----------------|----------|
| **Kalman Filter** (default) | `--filter kalman` | `measurement_noise` | 4-state filter [x, y, vx, vy]. Smooths noisy detections and predicts position through gaps. Higher `measurement_noise` = smoother but slower response. |
| **Alpha-Beta Filter** | `--filter alpha_beta` | `alpha`, `beta` | Simpler fixed-coefficient filter. `alpha` controls position smoothing, `beta` controls velocity tracking. |

### 7.4 Tracker FSM

The `Tracker` class wraps the detector and filter in a finite state machine:

```
SEARCHING → ACQUIRING → TRACKING → REACQUIRING → SEARCHING
```

| State | Behavior |
|-------|----------|
| SEARCHING | Scanning for the beacon. Executes a search pattern. |
| ACQUIRING | Beacon detected. Verifying over `acquire_threshold` consecutive frames. |
| TRACKING | Beacon locked. PID controller actively centers the beacon. Green border on HUD. |
| REACQUIRING | Beacon lost. Attempting to re-detect within `reacquire_timeout` frames before returning to SEARCHING. |

---

## 8. Disturbance Settings

The simulator supports eight independently configurable disturbance types. Each is applied during simulation and degrades tracking performance to varying degrees.

### 8.1 Platform Vibration

| Property | Value |
|----------|-------|
| Config parameter | `clean_vibration_rms` / `hard_vibration_rms` |
| Default (clean) | 0.0 (disabled) |
| Default (hard) | 0.2 deg |
| Effect | Adds random angular offset to camera pose each frame, simulating mechanical vibration |

### 8.2 Sensor Noise

| Property | Value |
|----------|-------|
| Config parameter | `clean_noise_sigma` / `hard_noise_sigma` |
| Default (clean) | 0.0 (disabled) |
| Default (hard) | 12.0 |
| Effect | Adds Gaussian noise to pixel values, degrading detection accuracy |

### 8.3 Motion Blur

| Property | Value |
|----------|-------|
| Config parameter | `clean_blur_pixels` / `hard_blur_pixels` |
| Default (clean) | 0.0 (disabled) |
| Default (hard) | 3.0 px |
| Effect | Applies directional blur kernel to the rendered frame, simulating camera motion during exposure |

### 8.4 Atmospheric Turbulence (v1.1)

| Property | Value |
|----------|-------|
| Config parameter | `turbulence_rms_deg`, `turbulence_correlation_s` |
| Default | 0.1 deg RMS, 1.0 s correlation |
| Effect | Slow, temporally correlated angular jitter on the camera, simulating atmospheric beam wander. Set `turbulence_rms_deg` to 0 to disable. |

### 8.5 Exposure Variation (v1.1)

| Property | Value |
|----------|-------|
| Config parameter | `exposure_rate_hz`, `exposure_amplitude` |
| Default | 0.1 Hz, 0.3 amplitude |
| Effect | Sinusoidal variation in frame brightness, simulating changing illumination conditions. Set `exposure_rate_hz` to 0 to disable. |

### 8.6 Occlusion (v1.1)

| Property | Value |
|----------|-------|
| Config parameter | `occlusion_duration_s`, `occlusion_frequency_hz` |
| Default | 1.0 s duration, 0.05 Hz frequency |
| Effect | Periodically blacks out the beacon region of the frame, simulating partial obstruction. Set `occlusion_duration_s` to 0 to disable. |

### 8.7 False Beacons (v1.1)

| Property | Value |
|----------|-------|
| Config parameter | `false_beacon_count`, `false_beacon_brightness_range` |
| Default | 2 beacons, brightness (80, 180) |
| Effect | Injects additional Gaussian spots into the frame to confuse the detector. Tests the tracker's ability to reject distractors. Set `false_beacon_count` to 0 to disable. |

### 8.8 Multi-Target Disturbances

| Property | Value |
|----------|-------|
| Config parameter | `multi_target`, `num_targets`, `target_colors` |
| Default | Disabled (single target) |
| Effect | Spawns multiple beacons with distinct colors and independent trajectories. Tests identity maintenance and cross-target confusion. Enabled via `--multi-target` CLI flag. |

### 8.9 Enabling and Disabling

Disturbances are enabled by setting their parameters to non-zero values and disabled by setting them to zero. The simulator skips disabled disturbances entirely (no computation overhead).

Example: enable only turbulence and false beacons in the clean scenario:

```yaml
scenario:
  vibration_rms: 0.0       # disabled
  noise_sigma: 0.0          # disabled
  blur_pixels: 0.0          # disabled
  turbulence_rms_deg: 0.15  # enabled
  false_beacon_count: 3     # enabled
```

---

## 9. Experiment Execution

### 9.1 Interactive Mode

The default mode. An OpenCV window opens and the user controls execution via keyboard:

```sh
uv run python src/main.py --scenario clean
```

Steps:
1. Window opens showing the virtual camera view (paused)
2. Press SPACE to start the simulation
3. Press 1/2 to switch scenarios mid-run
4. Press G to toggle ground-truth overlay for debugging
5. Press SPACE to pause at any time
6. Press R to reset and start over
7. Press Q or ESC to quit

Output is saved to the `results/` directory (or `--output` path).

### 9.2 Headless Mode

For automated or remote execution:

```sh
uv run python src/main.py --headless
```

Without a specific `--scenario`, both clean and hard scenarios run sequentially and a comparison table is printed:

```
Metric                     Clean         Hard
--------------------------------------------------
acquisition_time_s            0.10         0.53
mean_error_px                 3.42        12.87
max_error_px                 28.50        89.33
lock_retention_pct           98.50        87.20
num_losses                    0.00         2.00
detection_rate                0.99         0.91
rms_angular_error_deg         0.0234       0.0891
mean_fps                     29.80        28.50
```

Frames are saved as JPEG images in a per-run subdirectory. CSV logs and PNG plots are generated automatically.

### 9.3 Evaluation Mode

Run the full 8-category evaluation matrix:

```sh
uv run python src/main.py --eval
```

Options:
- `--duration 30` — override the default 10-second evaluation duration
- `--seeds 42,123,456` — specify multiple seeds for statistical significance

The evaluation framework runs multiple scenarios with varying disturbance combinations and produces a score (0-100) for each of the 8 evaluation categories. See [Section 11: Report Interpretation](#11-report-interpretation) for details.

### 9.4 GUI Mode

```sh
uv run python src/main.py --gui
```

Launches a PySide6 desktop application with:
- Live video feed
- Scenario selector
- Parameter adjustment sliders
- Start/Pause/Reset buttons
- Real-time metrics panel
- Export controls for logs and reports

---

## 10. Log Export

### 10.1 CSV Log

Every run produces a CSV file in the output directory with the naming pattern `{scenario}_{timestamp}.csv`.

**Header metadata** (comment lines):
```
# run_id=clean_20260909_143022
# seed=42
# timestamp=2026-09-09T14:30:22.123456
```

**19 fields per row:**

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `run_id` | string | Unique identifier for the run |
| 2 | `timestamp` | float | Simulation time in seconds |
| 3 | `frame_id` | int | Frame number (0-based) |
| 4 | `fps` | float | Measured frames per second |
| 5 | `target_id` | string | Beacon identifier (e.g., `beacon_01`) |
| 6 | `gt_x` | float | Ground-truth pixel x-coordinate |
| 7 | `gt_y` | float | Ground-truth pixel y-coordinate |
| 8 | `est_x` | float | Estimated pixel x-coordinate (from Kalman filter) |
| 9 | `est_y` | float | Estimated pixel y-coordinate (from Kalman filter) |
| 10 | `error_px` | float | Euclidean pixel error between estimate and frame center |
| 11 | `angular_error_deg` | float | Combined angular error in degrees |
| 12 | `detection_confidence` | float | Detection confidence (0.0-1.0) |
| 13 | `detected` | int | 1 if beacon detected this frame, 0 otherwise |
| 14 | `tracker_state` | string | FSM state: SEARCHING, ACQUIRING, TRACKING, or REACQUIRING |
| 15 | `pan_cmd` | float | Pan command sent to camera (deg) |
| 16 | `tilt_cmd` | float | Tilt command sent to camera (deg) |
| 17 | `pan_actual` | float | Actual camera pan angle (deg) |
| 18 | `tilt_actual` | float | Actual camera tilt angle (deg) |
| 19 | `processing_time_ms` | float | Per-frame processing time (ms) |

### 10.2 PNG Plot

Each run produces a plot file (`{run_id}_plot.png`) with two subplots:

- **Top**: Pixel error over time with background color-coding by tracker state (green = TRACKING, yellow = SEARCHING/ACQUIRING, red = REACQUIRING). Reference lines at 50 px and 128 px (central 20% boundary).
- **Bottom**: Detection binary trace (1 = detected, 0 = not detected).

### 10.3 JSON Summary

The `SummaryReporter` generates a structured JSON file (`evaluation_results.json` in the output directory) containing:

- Per-run summary statistics
- Configuration snapshot
- Evaluation category scores (when using `--eval`)

### 10.4 HTML Report

The `ReportGenerator` produces an HTML report file with:
- Embedded performance plots
- Summary statistics table
- Configuration parameters used
- Timestamp and run metadata

### 10.5 Frame Archive

Every frame with the HUD overlay is saved as a JPEG image in a per-run subdirectory:

```
results/
  clean_20260909_143022_frames/
    frame_00000.jpg
    frame_00001.jpg
    ...
```

This allows frame-by-frame review and video reconstruction.

---

## 11. Report Interpretation

### 11.1 Summary Statistics

After each run, the following metrics are printed and logged:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Acquisition time (s)** | Time from start until first TRACKING state | < 2.0 s |
| **Detection rate** | Fraction of frames where beacon was detected | > 0.95 |
| **Mean error (px)** | Average pixel error between estimated position and frame center | < 15 px |
| **Max error (px)** | Worst-case pixel error during the run | < 150 px |
| **RMS angular error (deg)** | Root-mean-square of combined pan/tilt angular error | < 0.05 deg |
| **Lock retention (%)** | Percentage of frames in TRACKING state | > 90% |
| **Loss events** | Number of transitions from TRACKING to REACQUIRING | 0 for clean, < 5 for hard |
| **Mean confidence** | Average detection confidence score | > 0.8 |
| **Mean FPS** | Average frames per second | > 25 (at 30 fps config) |

### 11.2 Evaluation Matrix

The `--eval` mode produces scores for 8 evaluation categories, each scored 0-100:

| Category | What It Measures |
|----------|-----------------|
| **Acquisition Speed** | How quickly the system locks onto the beacon |
| **Tracking Accuracy** | Steady-state pixel and angular error |
| **Tracking Stability** | Lock retention and loss event frequency |
| **Disturbance Rejection** | Performance under vibration, noise, and blur |
| **Reacquisition** | Ability to recover from beacon loss |
| **Detection Robustness** | Detection rate under varying conditions |
| **Temporal Consistency** | Smoothness of tracking commands (jitter) |
| **Overall** | Weighted average of all categories |

**Scoring scale:**
- 90-100: Excellent — meets all acceptance criteria
- 70-89: Good — acceptable for most scenarios
- 50-69: Marginal — needs tuning
- Below 50: Poor — significant issues

### 11.3 Plot Interpretation

**Error-over-time plot:**
- A well-tuned system shows error dropping quickly to below 50 px and staying there
- Spikes indicate disturbance events or reacquisition after loss
- The error should stay below the 128 px line (central 20% of the frame) for effective tracking
- Periodic spikes in the hard scenario are expected due to sinusoidal motion

**Detection trace:**
- Solid bar at 1.0 indicates continuous detection
- Gaps indicate occlusion events or detection failures
- The detection rate metric summarizes this trace as a single number

---

## 12. Troubleshooting

### 12.1 Display Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| "No display available" message | No `$DISPLAY` environment variable (SSH, container) | Use `--headless` flag |
| OpenCV window is black | GPU driver issue or display server conflict | Set `OPENCV_VIDEOIO_PRIORITY_MSMF=0` or try `--headless` |
| Window appears but is unresponsive | `cv2.waitKey()` blocking on wrong thread | Ensure main thread runs the display loop |

### 12.2 Performance Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| FPS drops below 20 | High processing time per frame | Reduce `width`/`height`, reduce `fps`, disable heavy disturbances |
| Large frame archive consumes disk | Long runs at high resolution | Reduce `duration_s` or run with `--headless` without frame saving |
| Evaluation takes too long | Running many seed combinations | Use fewer seeds with `--seeds 42` or shorter `--duration 10` |

### 12.3 Detection Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Beacon never detected | `threshold` too high or `beacon_sigma_px` too small | Lower `threshold` (try 40-60), increase `beacon_sigma_px` |
| False detections from noise | `threshold` too low or `noise_sigma` too high | Raise `threshold`, reduce `noise_sigma` |
| Tracker oscillates between states | `acquire_threshold` too low or `lose_threshold` too low | Increase `acquire_threshold` to 4-5, increase `lose_threshold` to 8-10 |
| Beacon detected but not tracked | `acquire_error_threshold` too restrictive | Increase `acquire_error_threshold` (try 200) |

### 12.4 Configuration Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| YAML loading fails | Malformed YAML syntax | Validate YAML structure; check indentation |
| CLI override not applied | Flag parsed after YAML load | CLI flags override YAML values; check flag names |
| "AIDetector not available" warning | ONNX Runtime not installed or model path wrong | Install onnxruntime, verify `--model` path |

### 12.5 Log and Report Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| CSV is empty | Simulation ran with zero frames | Check `duration_s` and `fps` values |
| Plot shows flat error at max | Beacon never acquired | Check detector parameters and disturbance settings |
| HTML report not generated | Report module import failed | Check that `src/report.py` exists and has no syntax errors |

---

## 13. Known Limitations

1. **2D angular simulation only.** The simulator models azimuth and elevation angles, not full 3D ray-tracing. Depth, perspective distortion, and range-dependent effects are not simulated.

2. **No real camera input.** Input is entirely synthetic. Processing real camera feeds is a v2.0 roadmap item.

3. **Single-threaded frame processing.** Detection, filtering, and control run sequentially per frame. Parallelization is a future optimization.

4. **AI detector requires pre-trained model.** No pre-trained model ships with the repository. The AI detector falls back to classical when no model is provided.

5. **OpenCV display required for interactive mode.** The GUI (`--gui`) requires PySide6, which may not be available on all platforms.

6. **Deterministic with fixed seed.** Results are fully reproducible for a given seed, but the same seed across different OS/hardware may produce slightly different floating-point results.

7. **No multi-target identity-switch metrics.** Multi-target mode tracks multiple beacons but does not yet compute MOTA/MOTP or identity-switch counts.

8. **Fixed threshold detector.** The classical detector uses a fixed brightness threshold, not adaptive. Performance degrades under significant exposure variation.

9. **Frame archive can be large.** A 60-second run at 30 fps saves 1,800 JPEG frames, consuming approximately 100-200 MB of disk space.

10. **No network or multi-user support.** The simulator runs as a single-user desktop application.

---

## 14. Complete Example Experiment

This section walks through a full experiment: configure a custom scenario, run it, review the results, and export logs.

### 14.1 Objective

Evaluate tracker performance under atmospheric turbulence with a random beacon trajectory.

### 14.2 Step 1: Create a Custom Scenario

Create the file `scenarios/turbulence_test.yaml`:

```yaml
simulation:
  fps: 30
  duration_s: 30
  random_seed: 42

camera:
  width: 1280
  height: 720
  h_fov_deg: 60.0
  v_fov_deg: 40.0
  rate_limit_deg_s: 60.0

beacon:
  sigma_px: 4.0
  peak: 255.0

detector:
  type: classical
  threshold: 80
  min_area: 2
  max_area: 500

filter:
  type: kalman
  measurement_noise: 8.0

pid:
  kp: 1.0
  ki: 0.0
  kd: 0.15
  deadband_pixels: 3.0

scenario:
  name: clean
  motion: random
  max_acc: 10.0
  change_interval: 2.0
  max_vel: 20.0
  vibration_rms: 0.0
  noise_sigma: 0.0
  blur_pixels: 0.0
  turbulence_rms_deg: 0.15
  turbulence_correlation_s: 1.0
  exposure_rate_hz: 0.0
  exposure_amplitude: 0.0
  occlusion_duration_s: 0.0
  occlusion_frequency_hz: 0.0
  false_beacon_count: 0
```

### 14.3 Step 2: Run the Experiment

```sh
uv run python src/main.py --yaml scenarios/turbulence_test.yaml --output results/turbulence_test
```

### 14.4 Step 3: Observe the HUD

Watch the display window:
- The beacon follows a random (non-repeating) trajectory
- Turbulence causes slow angular jitter on the camera
- The STATE indicator should reach TRACKING within 1-2 seconds
- The ERROR value should stabilize below 50 px

### 14.5 Step 4: Review the Console Output

After the run completes, the console prints:

```
  Run complete: 900 frames

  Summary:
    Duration:            30.0s
    Mean FPS:            29.8
    Mean processing:     8.45 ms
    Acquisition time:    0.13s
    Detection rate:      98.7%
    Mean error:          4.21 px
    Max error:           45.80 px
    RMS angular error:   0.0298 deg
    Lock retention:      96.3%
    Loss events:         1
    Mean confidence:     0.923
```

### 14.6 Step 5: Examine the Plot

Open `results/turbulence_test/turbulence_test_{timestamp}_plot.png`:
- The top subplot shows pixel error over time with colored background
- Turbulence-induced spikes should be visible but bounded
- The bottom subplot shows continuous detection (solid bar at 1.0)

### 14.7 Step 6: Inspect the CSV Log

Open the CSV file in a spreadsheet or analyze with Python:

```python
import pandas as pd
df = pd.read_csv("results/turbulence_test/turbulence_test_YYYYMMDD_HHMMSS.csv", comment='#')
print(df.describe())
print(f"Time in TRACKING: {(df['tracker_state'] == 'TRACKING').mean()*100:.1f}%")
```

### 14.8 Step 7: Review the HTML Report

Open the generated HTML report in a browser for a formatted summary with embedded plots and configuration details.

### 14.9 Step 8: Compare Against Baseline

Run the same scenario without turbulence to establish a baseline:

```sh
uv run python src/main.py --yaml scenarios/clean.yaml --output results/baseline
```

Compare the two sets of metrics to quantify the impact of turbulence on tracking performance.

---

## Appendix A: Motion Models

| Model | Config Value | Parameters | Behavior |
|-------|-------------|------------|----------|
| Circular | `motion: circular` | `center_az`, `center_el`, `radius`, `speed` | Beacon traces a circle in angular space |
| Sinusoidal | `motion: sinusoidal` | `az_amp`, `az_freq`, `el_amp`, `el_freq` | Independent sinusoidal oscillation in azimuth and elevation |
| Constant Velocity | `motion: constant` | `az_rate`, `el_rate` | Beacon moves at fixed angular velocity |
| Random Maneuvering | `motion: random` | `max_acc`, `change_interval`, `max_vel` | Random acceleration changes at fixed intervals |

## Appendix B: Tracker State Machine Reference

```
                    detect confirmed
    ┌──────────┐  ───────────────────>  ┌────────────┐
    │          │                         │            │
    │SEARCHING │                         │ ACQUIRING  │
    │          │  <───────────────────   │            │
    └──────────┘   lose_threshold        └────────────┘
         ^         consecutive misses          │
         │                                     │ acquire_threshold
         │                                     │ consecutive detections
         │                                     │ AND error < acquire_error_threshold
         │                                     v
    ┌──────────┐   reacquire_timeout     ┌────────────┐
    │          │  ────────────────────>   │            │
    │SEARCHING │   frames elapsed         │  TRACKING  │
    │          │                         │            │
    └──────────┘                         └────────────┘
         ^                                     │
         │          lose_threshold             │
         │          consecutive misses         │
         │                                     v
         │  ┌───────────────────────────────────┐
         └──│          REACQUIRING              │
            └───────────────────────────────────┘
```

## Appendix C: File Output Structure

```
results/
  {scenario}_{timestamp}.csv              # Per-frame CSV log
  {scenario}_{timestamp}_plot.png         # Error-over-time plot
  {scenario}_{timestamp}_summary.json     # Structured JSON summary
  {scenario}_{timestamp}_report.html      # HTML report with embedded plots
  {scenario}_{timestamp}_frames/          # Frame archive directory
    frame_00000.jpg
    frame_00001.jpg
    ...
  evaluation_results.json                 # Evaluation matrix (when using --eval)
```
