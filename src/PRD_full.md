# Product Requirements Document

# FSOC Virtual Camera Tracking System

**AI-Based Coarse-Alignment Simulator for Mobile Free-Space Optical Communication Terminals**

| Field | Value |
|-------|-------|
| Document version | 2.0 |
| Status | Full Product Specification |
| Last updated | 2026-09-09 |
| Owner | Project Team |
| Classification | Internal / Submission |
| Baseline codebase | `fsoc-virtual-tracker/` (sim.py, detect.py, control.py, main.py, config.py) |

---

## Table of Contents

1. [Product Vision](#1-product-vision)
2. [Problem Context](#2-problem-context)
3. [Target Users and Use Cases](#3-target-users-and-use-cases)
4. [Scope](#4-scope)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [System Architecture](#7-system-architecture)
8. [Module Specifications](#8-module-specifications)
9. [Data Models and Interfaces](#9-data-models-and-interfaces)
10. [Configuration System](#10-configuration-system)
11. [Virtual Environment and Camera Model](#11-virtual-environment-and-camera-model)
12. [Target Motion Models](#12-target-motion-models)
13. [Disturbance and Sensor Models](#13-disturbance-and-sensor-models)
14. [Beacon Detection and Identification](#14-beacon-detection-and-identification)
15. [Tracking Filter and State Machine](#15-tracking-filter-and-state-machine)
16. [Pan-Tilt Controller](#16-pan-tilt-controller)
17. [AI Detection Method](#17-ai-detection-method)
18. [GUI and Operator Workflow](#18-gui-and-operator-workflow)
19. [Logging and Performance Reports](#19-logging-and-performance-reports)
20. [Evaluation Framework](#20-evaluation-framework)
21. [Validation and Testing](#21-validation-and-testing)
22. [Packaging and Distribution](#22-packaging-and-distribution)
23. [Technical Report Structure](#23-technical-report-structure)
24. [User Manual Structure](#24-user-manual-structure)
25. [Scenarios and Evaluation](#25-scenarios-and-evaluation)
26. [Performance Requirements](#26-performance-requirements)
27. [Acceptance Criteria](#27-acceptance-criteria)
28. [Technical Constraints](#28-technical-constraints)
29. [Risks and Mitigations](#29-risks-and-mitigations)
30. [Milestones and Timeline](#30-milestones-and-timeline)
31. [Future Roadmap](#31-future-roadmap)
32. [Glossary](#32-glossary)
33. [Revision History](#33-revision-history)

---

## 1. Product Vision

### 1.1 Vision Statement

A standalone software application that generates a moving optical beacon in a configurable virtual environment, renders the scene through a virtual pan-tilt camera, detects and identifies the beacon in the camera stream, estimates its image position, and continuously commands the camera to keep the beacon inside the field of view. The system injects realistic disturbances and automatically produces live and post-run performance statistics. The system supports both a classical computer-vision detector as a baseline and an AI-based detector as a comparative method, evaluated on identical test sequences.

The recommended implementation is a **software-in-the-loop coarse-alignment simulator**. The first version uses a deterministic virtual scene and a classical detector as the baseline. An AI detector is then added as an optional or comparative method rather than making the entire system dependent on a trained neural network.

> **Coarse alignment** is the process of bringing the remote optical terminal or beacon into a suitable camera region before fine pointing takes over. It does not replace fine laser pointing or communication-link optimization.

### 1.2 Product Goals

| # | Goal | Metric |
|---|------|--------|
| G1 | Demonstrate closed-loop beacon tracking in real time | Target acquired and centered in under 2 seconds; steady-state error below 15 pixels |
| G2 | Operate without any optical or pan-tilt hardware | Runs on a standard laptop with Python and OpenCV |
| G3 | Produce reproducible, measurable results | Every run logged with seed, configuration, per-frame metrics, and summary report |
| G4 | Support comparison between detection methods | Classical and AI detectors share a common interface and are evaluated on identical scenarios |
| G5 | Handle a full suite of realistic disturbances | Platform vibration, atmospheric turbulence, sensor noise, motion blur, exposure variation, occlusion, and false beacons — each independently configurable |
| G6 | Support multi-target scenarios | Multiple beacons with unique identities, crossing trajectories, and identity-switch metrics |
| G7 | Provide an operator GUI | Desktop application with live video, controls, configuration, scenario management, and report export |
| G8 | Deliver a complete package | Executable, source code, technical report, and user manual |

### 1.3 Non-Goals

| Capability | Status | Rationale |
|-----------|--------|-----------|
| Fine laser pointing | Out of scope | This system performs coarse alignment only |
| Communication-link optimization | Out of scope | Link budget, modulation, and coding are separate concerns |
| Real-time processing of external camera feeds (v1) | Out of scope | Input is synthetic frames; real camera support is a v2.0 roadmap item |
| Field deployment | Out of scope | This is a simulation and evaluation tool, not a fielded system |
| 3-D physics engine (v1) | Deferred | 2-D angular simulator is sufficient for v1; 3-D renderer is a v2.0 item |

### 1.4 Success Definition

The product is successful when all of the following hold:

1. The application runs without specialized optical or pan-tilt hardware.
2. A scenario can be configured, started, paused, reset, and replayed.
3. At least one moving beacon is detected and tracked through a synthetic video stream.
4. The virtual camera changes pan and tilt in response to image-space target error.
5. The system supports target loss and reacquisition.
6. Disturbances can be enabled independently and in combination.
7. The application displays real-time tracking state and statistics.
8. Every run produces raw logs and a summary performance report.
9. The baseline detector and optional AI detector can be compared on the same scenarios.
10. The executable, source code, technical report, and user manual are delivered together.
11. The evaluation results include configuration, random seed, metrics, plots, and known limitations.

---

## 2. Problem Context

### 2.1 FSOC Coarse Alignment

Free-space optical communication (FSOC) uses narrow laser beams to transmit data between platforms — drones, satellites, ground vehicles, ships. These links can achieve multi-Gbps throughput with no spectrum licensing, but the narrow beam divergence (typically under 1 degree) creates a pointing challenge.

Before fine pointing and communication can begin, the receiving terminal's camera must locate the transmitting terminal's beacon (or vice versa) and bring it into a suitable region of the field of view. This step is called **coarse alignment**.

### 2.2 Current Limitations

| Limitation | Impact |
|-----------|--------|
| Manual operation | Human operators adjust pointing until the beacon is visible; slow and impractical for autonomous platforms |
| Hardware dependency | Dedicated pointing assemblies with gimbals and encoders are expensive, heavy, and power-hungry |
| Slow acquisition | Scanning the full sky for a beacon can take minutes; link unavailable during search |
| Fragility under disturbance | Vibration, atmospheric turbulence, and platform motion cause the beacon to leave the FOV, requiring reacquisition |
| No reproducible benchmarking | Algorithms are tested on different hardware with different conditions; results are not comparable |

### 2.3 How This Product Addresses It

The FSOC Virtual Camera Tracking System simulates the coarse-alignment problem in software, providing a reproducible testbed for developing and comparing detection and control strategies:

- A **virtual environment** models the angular relationship between the camera and one or more beacons.
- A **virtual pan-tilt camera** with configurable FOV and actuator limits renders synthetic frames.
- A **disturbance suite** injects vibration, turbulence, noise, blur, exposure variation, occlusion, and false beacons — each independently controllable.
- A **classical computer-vision detector** finds the beacon in each frame.
- An **optional AI detector** trained on synthetic frames provides a comparative method behind the same interface.
- A **Kalman-filtered tracker** with a **finite-state machine** manages the search-acquire-track-reacquire lifecycle.
- A **PID controller** converts image-space error to pan/tilt commands that keep the beacon centered.
- An **evaluation framework** with eight scenario categories and weighted scoring produces fair, reproducible comparisons.

---

## 3. Target Users and Use Cases

### 3.1 Primary Users

| User | Description | Primary Need |
|------|-------------|--------------|
| FSOC researchers | Engineers developing pointing, acquisition, and tracking (PAT) systems | Reproducible testbed for algorithm comparison with controlled disturbances |
| Hackathon judges | Technical reviewers evaluating project impact, architecture, and demo quality | A live, working demo with measurable results |
| Students learning AI/CV | Engineering students studying closed-loop vision systems | A working example with clean, documented code |
| Project evaluators | Reviewers assessing technical depth and engineering discipline | Architecture diagrams, logs, plots, technical report, user manual |
| AI/ML practitioners | Developers training and benchmarking object detectors on synthetic data | Labeled synthetic frame generation, ONNX export, comparison metrics |

### 3.2 Primary Use Cases

#### UC-1: Live Demo
An operator starts the application, selects the clean scenario, watches the beacon be acquired and tracked, switches to the hard scenario to show robustness, and reviews the error plot and summary report.

**Trigger:** Operator runs the executable or `python main.py`
**Preconditions:** Application installed (executable or source with dependencies)
**Postconditions:** CSV log, error plot PNG, and summary report saved to `results/`
**Success:** Beacon visibly tracked; state transitions visible on HUD; report generated

#### UC-2: Headless Benchmark
A researcher runs all eight evaluation categories automatically in headless mode, collects CSVs, summary statistics, and the evaluation matrix, and compares two detector configurations by changing the configuration and re-running.

**Trigger:** `python main.py --headless --eval`
**Preconditions:** All scenarios configured; no display required
**Postconditions:** One CSV per scenario, summary JSON, evaluation matrix CSV, comparison plots
**Success:** Metrics are reproducible across runs with the same seed; matrix shows clear performance differentiation

#### UC-3: AI Detector Development
A developer generates thousands of labeled synthetic frames using the simulator, trains a small object detector, exports it to ONNX, drops it behind the `DetectorBase` interface, and benchmarks it against the classical baseline on the same test sequences.

**Trigger:** Developer runs the frame-generation script and the training pipeline
**Preconditions:** PyTorch or ONNX Runtime installed; labeled data generated
**Postconditions:** AI detector integrated and benchmarked
**Success:** AI detector achieves precision/recall comparable to or better than the classical detector; fallback to classical on low confidence

#### UC-4: Parameter Tuning
An engineer adjusts PID gains, Kalman noise, or disturbance parameters in the YAML configuration file and re-runs a scenario to see the effect on tracking error and lock retention.

**Trigger:** Engineer edits the YAML config
**Preconditions:** System runs without errors
**Postconditions:** New parameters produce a CSV, plot, and updated summary
**Success:** Tuned parameters improve target metrics (lower error, higher lock retention)

#### UC-5: Scenario Creation
A developer creates a new scenario by writing a YAML file specifying motion model, disturbance levels, beacon parameters, and duration. The system loads, validates, and runs it.

**Trigger:** Developer creates `scenarios/new_scenario.yaml`
**Preconditions:** YAML schema validation passes
**Postconditions:** New scenario is selectable via GUI or CLI
**Success:** Scenario runs end-to-end with logging, plotting, and evaluation

#### UC-6: Multi-Target Evaluation
A researcher configures a scenario with two crossing beacons of different colors and measures the identity-switch rate — how often the tracker follows the wrong beacon after a crossing.

**Trigger:** Researcher loads multi-target scenario
**Preconditions:** Multi-target mode enabled; beacons assigned unique IDs and colors
**Postconditions:** Identity-switch metric in summary report
**Success:** System tracks the designated beacon through the crossing; switch rate reported

#### UC-7: Report Export
An operator completes a run and clicks "Export Report" in the GUI. The system generates a structured HTML or PDF report containing configuration, seed, all metrics, plots, and known limitations.

**Trigger:** Operator clicks Export Report
**Preconditions:** At least one completed run
**Postconditions:** Report file saved to disk
**Success:** Report contains all required sections (see §23)

---

## 4. Scope

### 4.1 Full Product Scope

| Capability | Phase | Status |
|-----------|-------|--------|
| 2-D angular environment simulator | v1.0 | Delivered |
| Virtual pan-tilt camera with pinhole projection | v1.0 | Delivered |
| 3-D renderer (optional, for realistic depth/occlusion) | v2.0 | Planned |
| Four+ target motion models | v1.0 | Delivered (4 models) |
| Scripted trajectory replay mode | v1.1 | Planned |
| Classical beacon detector (brightness + connected components) | v1.0 | Delivered |
| AI-based detector (trained on synthetic frames, ONNX export) | v1.1 | Planned |
| 2-D Kalman filter (constant velocity) | v1.0 | Delivered |
| Alpha-beta filter (alternative) | v1.1 | Planned |
| Four-state tracking FSM | v1.0 | Delivered |
| Seven-state tracking FSM (with candidate verification and failed states) | v1.1 | Planned |
| PID pan-tilt controller with rate limiting, deadband, anti-windup | v1.0 | Delivered |
| Platform vibration disturbance | v1.0 | Delivered |
| Atmospheric turbulence disturbance | v1.1 | Planned |
| Sensor noise (Gaussian + Poisson) | v1.0 | Delivered (Gaussian) |
| Motion blur disturbance | v1.0 | Delivered |
| Exposure variation disturbance | v1.1 | Planned |
| Occlusion model (scripted masked regions) | v1.1 | Planned |
| False-beacon distractors | v1.1 | Planned |
| Multi-target scenarios with identity discrimination | v1.1 | Planned |
| Live OpenCV display with HUD overlay | v1.0 | Delivered |
| PySide6 GUI with config controls, scenario save/load, report export | v1.1 | Planned |
| Per-frame CSV logging | v1.0 | Delivered |
| Structured JSON summary | v1.1 | Planned |
| HTML/PDF performance report | v1.1 | Planned |
| Matplotlib error plots (PNG) | v1.0 | Delivered |
| Two demo scenarios (clean + hard) | v1.0 | Delivered |
| Eight-category evaluation matrix with weighted scoring | v1.1 | Planned |
| YAML/JSON scenario configuration | v1.1 | Planned |
| Standalone config.py (dataclass) | v1.0 | Delivered |
| Module-level self-tests | v1.0 | Delivered |
| Unit test framework (pytest) | v1.1 | Planned |
| PyInstaller standalone executable | v1.1 | Planned |
| Technical report (10–15 pages) | v1.1 | Planned |
| User manual | v1.1 | Planned |
| Hardware-in-the-loop (real pan-tilt gimbal) | v2.0 | Planned |

### 4.2 Scope Decisions and Rationale

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Coordinate system | 2-D angular (azimuth, elevation) | 3-D Cartesian with depth | Coarse alignment depends primarily on angular separation; 2-D is easier to validate and sufficient for v1 |
| Beacon rendering | Gaussian blob | Hard pixel, textured sprite | Gaussian is realistic (beacons bloom) and easier to detect |
| Baseline detection | Brightness threshold + connected components | Template matching, SIFT/SURF | Point-source beacons are the bright case for classical CV; fast and explainable |
| AI detection | Optional, behind common interface | Replace classical entirely | AI may not outperform classical on point sources; keep classical as fallback |
| Tracker filter | Kalman filter (constant velocity) | Particle filter, IMM | Kalman is well-understood, ~40 lines, provides velocity for prediction |
| Controller | PID with rate limits | LQR, MPC | PID is standard for pointing control; easy to tune and explain |
| GUI | PySide6 (v1.1) + cv2.imshow (v1.0) | Web UI, Qt only | PySide6 gives desktop controls; cv2.imshow is the v1.0 fallback |
| Config format | YAML (v1.1) + config.py (v1.0) | JSON, TOML, INI | YAML is human-readable and supports comments; config.py is the v1.0 fast-start |
| Packaging | PyInstaller (v1.1) | Docker, conda | PyInstaller produces a single executable for Windows/Linux/macOS |
| Evaluation | Weighted 8-category matrix | Single score | Matrix shows strengths and weaknesses across disturbance types |

---

## 5. Functional Requirements

### 5.1 Scenario Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-SM1 | The system shall load, validate, save, and reset experiment configurations. | P0 |
| FR-SM2 | Configurations shall be stored in versioned YAML files. | P1 |
| FR-SM3 | The system shall validate configuration schema before running and report errors with field names. | P0 |
| FR-SM4 | The system shall support a library of named scenarios selectable from the GUI or CLI. | P1 |
| FR-SM5 | Each scenario shall record its configuration and random seed with every result. | P0 |
| FR-SM6 | The system shall support scenario replay: identical seed produces identical metrics. | P0 |

### 5.2 Simulation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-S1 | The system shall simulate a 2-D angular environment with azimuth and elevation coordinates in degrees. | P0 |
| FR-S2 | The system shall render a configurable background with brightness gradients and optional stars or terrain. | P1 |
| FR-S3 | The system shall render the beacon as a 2-D Gaussian blob with configurable sigma (pixels) and peak intensity (0–255). | P0 |
| FR-S4 | The system shall support at least four target motion models: constant angular velocity, sinusoidal, circular, and random maneuvering with bounded acceleration. | P0 |
| FR-S5 | Each motion model shall expose position, velocity, and acceleration at each simulation time step. | P1 |
| FR-S6 | The system shall support a scripted trajectory mode for exact replay of test cases. | P1 |
| FR-S7 | The system shall use a fixed simulation time step (1/FPS seconds) regardless of actual rendering speed. | P0 |
| FR-S8 | The system shall support a configurable random seed for reproducible experiments. | P0 |
| FR-S9 | The simulation shall be resettable to its initial state. | P0 |
| FR-S10 | For multiple targets, each beacon shall have a unique identifier, color, intensity, and trajectory. | P1 |
| FR-S11 | The system shall generate multi-target test cases where targets cross, approach, temporarily disappear, or differ in intensity. | P1 |

### 5.3 Virtual Camera

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-C1 | The system shall model a pinhole-projection camera with configurable resolution, horizontal FOV, and vertical FOV. | P0 |
| FR-C2 | The system shall project angular target positions to image pixel coordinates using the camera's pan, tilt, and FOV. | P0 |
| FR-C3 | The system shall perform the inverse projection (pixel to angular) for use by the controller. | P0 |
| FR-C4 | The system shall determine whether a target is visible (inside the image frame) at each frame. | P0 |
| FR-C5 | The camera shall have configurable pan and tilt range limits. | P0 |
| FR-C6 | The camera shall have a configurable maximum angular velocity (rate limit). | P0 |
| FR-C7 | The camera pose shall be settable by the controller each frame. | P0 |
| FR-C8 | A 3-D renderer shall be available as an optional upgrade path for realistic depth, occlusion, and platform geometry. | P2 |

### 5.4 Disturbances

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-D1 | The system shall support a platform vibration model: sum of sinusoidal signals plus random jitter, applied to the camera boresight. | P0 |
| FR-D2 | The system shall support an atmospheric turbulence model: low-frequency random angular displacement producing slowly varying pointing error. | P1 |
| FR-D3 | The system shall support a sensor noise model: Gaussian and Poisson noise on pixel values. | P0 |
| FR-D4 | The system shall support a motion blur model: directional kernel based on relative angular velocity. | P0 |
| FR-D5 | The system shall support an exposure variation model: brightness and gain modulation. | P1 |
| FR-D6 | The system shall support an occlusion model: random or scripted masked regions forcing temporary track loss. | P1 |
| FR-D7 | The system shall support a false-beacon model: distractor points with similar appearance to test identification robustness. | P1 |
| FR-D8 | Each disturbance shall be independently enable-able and disable-able through configuration. | P0 |
| FR-D9 | Each disturbance shall be reproducible via the random seed. | P0 |
| FR-D10 | Disturbances shall be applied to a copy of the camera state, not the original. | P0 |
| FR-D11 | Each disturbance shall implement an `apply()` method on a common interface so it can be toggled individually during debugging. | P0 |
| FR-D12 | Disturbance parameters shall be independently configurable to create controlled evaluation levels (low, medium, high). | P1 |

### 5.5 Beacon Detection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-DE1 | The detector shall receive only the rendered frame — no ground truth. | P0 |
| FR-DE2 | The baseline detector shall convert the frame to grayscale and HSV color spaces, apply brightness and color thresholding, perform morphological opening and closing, extract connected components or contours, and reject candidates outside permitted size, shape, intensity, or persistence limits. | P0 |
| FR-DE3 | The detector shall compute each candidate's centroid, area, bounding box, and confidence score. | P0 |
| FR-DE4 | The detector shall select the candidate matching the designated beacon identity or the best available track. | P0 |
| FR-DE5 | The detector shall support white, red, green, and infrared-like simulated beacons. | P1 |
| FR-DE6 | The detector shall return `detected=False` when no candidate passes the filters. | P0 |
| FR-DE7 | Detection thresholds shall be exposed in the GUI and configuration file. | P1 |

### 5.6 AI Detection (Optional / Comparative)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-AI1 | The system shall support an AI detector as a second detector behind a common interface (`DetectorBase`). | P1 |
| FR-AI2 | The AI detector shall output a bounding box, class or beacon identity, and confidence score. | P1 |
| FR-AI3 | The system shall generate thousands of labeled synthetic frames across target positions, sizes, brightness values, disturbances, and backgrounds for training. | P1 |
| FR-AI4 | The training data shall be split into training, validation, and test sets by scenario seed rather than by adjacent frames. | P1 |
| FR-AI5 | The AI detector shall be exported to ONNX for cross-platform deployment if required. | P2 |
| FR-AI6 | The system shall keep a fallback path to the classical detector if the AI model is unavailable or below its confidence threshold. | P1 |
| FR-AI7 | AI detection improvement shall not be claimed unless demonstrated through repeatable metrics (precision, recall, acquisition time, lock-retention rate). | P0 |

### 5.7 Tracking and State Machine

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-T1 | The tracker shall use a 2-D constant-velocity Kalman filter with state vector [x, y, vx, vy] and measurement [x, y]. | P0 |
| FR-T2 | The Kalman filter shall predict every frame and update only when a detection is available. | P0 |
| FR-T3 | An alpha-beta filter shall be available as a simpler alternative. | P2 |
| FR-T4 | The tracker shall implement a finite-state machine. v1.0: four states (Searching, Acquiring, Tracking, Reacquiring). v1.1: seven states (adding Candidate Verification, Idle, Failed). | P0 |
| FR-T5 | SEARCHING state shall sweep or spiral the virtual camera. | P0 |
| FR-T6 | ACQUIRING (or Candidate Verification) state shall require N consecutive detections before transitioning to TRACKING. | P0 |
| FR-T7 | TRACKING state shall apply continuous feedback control. | P0 |
| FR-T8 | REACQUIRING state shall predict briefly using Kalman velocity, then search around the last known position. | P0 |
| FR-T9 | The tracker shall distinguish between a true loss and a single bad frame. | P0 |
| FR-T10 | A FAILED state shall record failure and stop or restart. | P1 |
| FR-T11 | The tracker shall be resettable to initial state. | P0 |

### 5.8 Pan-Tilt Controller

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-CT1 | The controller shall convert image-space pixel error to angular error using the camera FOV and resolution. | P0 |
| FR-CT2 | The error conversion shall be: `error_pan = (target_x - cx) / width * h_fov`; `error_tilt = (cy - target_y) / height * v_fov` (y inverted). | P0 |
| FR-CT3 | The controller shall use a PID controller per axis with configurable gains. | P0 |
| FR-CT4 | The controller shall apply rate limiting (maximum angular velocity per frame). | P0 |
| FR-CT5 | The controller shall apply a deadband to prevent micro-jitter. | P0 |
| FR-CT6 | The controller shall apply anti-windup to the integral term. | P1 |
| FR-CT7 | The controller shall clamp output to the pan/tilt range limits. | P0 |
| FR-CT8 | The controller shall use a fixed time step for integration. | P0 |
| FR-CT9 | The controller shall be tunable first in a disturbance-free scenario, then tested with progressively stronger disturbances. | P0 |
| FR-CT10 | The controller shall output: pan command, tilt command, pan rate, tilt rate, angular error, pixel error, and saturation flag. | P0 |

### 5.9 Ground Truth

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-GT1 | Every frame shall produce a ground-truth record containing: target angular position, target pixel position, target visibility flag, camera pan/tilt, timestamp, and frame ID. | P0 |
| FR-GT2 | Ground truth shall never be passed to the detector or controller. | P0 |
| FR-GT3 | Ground truth shall be computed using the disturbed camera state. | P0 |
| FR-GT4 | Ground-truth overlays shall be disabled in evaluation mode to prevent accidental leakage. | P0 |

### 5.10 GUI and Operator Workflow

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-GUI1 | The main window shall contain live virtual-camera video. | P0 |
| FR-GUI2 | The GUI shall show: current pan/tilt angles, target estimate, confidence, current state, instantaneous error, FPS, processing time, and lock-retention indicator. | P0 |
| FR-GUI3 | The GUI shall provide configuration controls for target motion and disturbances. | P1 |
| FR-GUI4 | The GUI shall provide start, pause, reset, and stop controls. | P0 |
| FR-GUI5 | The GUI shall provide scenario save/load controls. | P1 |
| FR-GUI6 | The GUI shall provide an export-performance-report control. | P1 |
| FR-GUI7 | The GUI shall use separate visual styles for measured detections and hidden ground truth. | P0 |
| FR-GUI8 | Ground-truth overlays shall be disabled in evaluation mode. | P0 |
| FR-GUI9 | The simulation engine shall be independent from the GUI to allow headless testing and batch experiments. | P0 |

### 5.11 Logging and Reporting

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-L1 | The system shall log one record per frame. | P0 |
| FR-L2 | Each log record shall contain at least: run_id, timestamp, frame_id, fps, target_id, ground-truth x/y, estimated x/y, position_error_px, angular_error_deg, detection_confidence, detected, tracker_state, pan_cmd, tilt_cmd, pan_actual, tilt_actual, processing_time_ms. | P0 |
| FR-L3 | The CSV file header shall include the run ID, random seed, and timestamp. | P1 |
| FR-L4 | The system shall generate a summary report containing all metrics listed in §19.2. | P0 |
| FR-L5 | The system shall export raw CSV, structured JSON, and a human-readable HTML or PDF report. | P1 |
| FR-L6 | The system shall generate error-over-time plots as PNG. | P0 |
| FR-L7 | Results shall be organized by disturbance level and target trajectory. | P1 |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NF-P1 | The system shall run at ≥ 30 FPS in the clean scenario on a standard laptop. | ≥ 30 FPS |
| NF-P2 | The system shall run at ≥ 15 FPS in any single-disturbance scenario. | ≥ 15 FPS |
| NF-P3 | Per-frame processing time (sim + detect + control) shall be under 35ms in clean conditions. | < 35ms |
| NF-P4 | AI detector inference shall be under 30ms per frame. | < 30ms |
| NF-P5 | Acquisition time shall be under 2 seconds in nominal conditions. | < 2s |
| NF-P6 | Steady-state tracking error in the clean scenario shall be under 15 pixels. | < 15px |
| NF-P7 | Lock retention shall be above 95% in the clean scenario. | > 95% |

### 6.2 Reliability

| ID | Requirement |
|----|-------------|
| NF-R1 | The system shall run for at least 60 seconds without crashing in any scenario. |
| NF-R2 | The system shall handle edge cases: beacon at FOV edge, all-zero frame, beacon temporarily outside frame. |
| NF-R3 | Identical random seeds shall produce identical per-frame metrics. |
| NF-R4 | Ground-truth data shall never reach the detector or controller. |
| NF-R5 | The system shall not corrupt state on user input errors (invalid config, missing file). |

### 6.3 Usability

| ID | Requirement |
|----|-------------|
| NF-U1 | The system shall be startable with a single command or executable. |
| NF-U2 | The README and user manual shall cover installation, startup, controls, configuration, troubleshooting, and at least one complete example experiment. |
| NF-U3 | The GUI shall be usable on a 1280×720 display. |
| NF-U4 | Error messages shall be meaningful and actionable. |

### 6.4 Portability

| ID | Requirement |
|----|-------------|
| NF-PO1 | The system shall run on Linux, macOS, and Windows with Python 3.11+. |
| NF-PO2 | The system shall use only pure-Python dependencies (NumPy, OpenCV, SciPy, Matplotlib, PySide6, PyTorch/ONNX optional). |
| NF-PO3 | The system shall detect headless environments and operate without a display. |
| NF-PO4 | No GPU required for the baseline system. GPU recommended for AI detector training only. |

### 6.5 Maintainability

| ID | Requirement |
|----|-------------|
| NF-M1 | Each module shall have a single, clear responsibility. |
| NF-M2 | Module interfaces shall be stable. |
| NF-M3 | Each module shall include self-tests. |
| NF-M4 | All configurable parameters shall be in configuration files, not hardcoded. |
| NF-M5 | The simulation engine shall be independent from the GUI. |
| NF-M6 | The codebase shall follow PEP 8. |

### 6.6 Reproducibility

| ID | Requirement |
|----|-------------|
| NF-RE1 | Every run shall record the random seed. |
| NF-RE2 | The same seed and configuration shall produce identical per-frame metrics. |
| NF-RE3 | All RNGs shall be seeded. |
| NF-RE4 | The simulation time step shall be fixed. |

---

## 7. System Architecture

### 7.1 Architecture Overview

The system follows a modular pipeline with clear interfaces between simulation, sensing, detection, control, visualization, and logging. The guiding principle is **ground-truth isolation**: ground truth flows only to the logger and dev-mode HUD, never to the detector or controller.

### 7.2 Module Map

```
Scenario Configuration (YAML / config.py)
        │
        ▼
Target and Environment Simulator ──────► Disturbance Model
        │                                        │
        +──────────────────► Virtual Camera ◄───+
                                  │
                          Synthetic Video Frame
                                  │
                    Preprocessing and Image Quality
                                  │
                   Beacon Detection and Identification
                     (Classical baseline + optional AI)
                                  │
                 Position, Confidence, and Track Filter
                   (Kalman / alpha-beta + FSM)
                                  │
                   Pan-Tilt Controller and State Machine
                   (PID + search pattern + rate limits)
                                  │
                       Camera Pose Update / Rendering
                                  │
                    GUI, Metrics, Logs, Replay, Reports
```

### 7.3 File Structure

```
fsoc-virtual-tracker/
├── app/
│   ├── main.py              # Entry point, main loop, display, logging
│   ├── gui/                 # PySide6 GUI (v1.1)
│   ├── simulation/          # Environment, camera, motion, disturbances
│   ├── vision/              # Detector, Kalman filter, tracker FSM
│   ├── control/             # PID controller, search pattern
│   ├── evaluation/          # Evaluation matrix, scenario runner
│   └── logging/             # CSV logger, report generator
├── configs/                 # Default configuration files
├── scenarios/               # YAML scenario definitions
├── tests/                   # pytest unit and integration tests
├── models/                  # Trained AI detector models (ONNX)
├── docs/                    # Technical report, user manual
├── requirements.txt
├── README.md
└── build/                   # PyInstaller output
```

**v1.0 flat structure** (current baseline): `sim.py`, `detect.py`, `control.py`, `main.py`, `config.py` at the project root. The package structure above is the v1.1 target.

### 7.4 Data Flow (Per Frame)

The following sequence executes once per frame, in strict order:

1. **Update target state** — the motion model advances the beacon's angular position, velocity, and acceleration by dt.
2. **Copy camera state** — a copy of the camera's current pose is created.
3. **Apply disturbances** — vibration, turbulence, and camera motion offsets are applied to the camera copy.
4. **Project visible targets** — angular positions are projected to pixel coordinates.
5. **Render frame** — background, beacon shapes, clutter, and optional occlusions are drawn.
6. **Apply sensor effects** — blur, noise, exposure variation, and saturation.
7. **Compute ground truth** — true target pixel position using the disturbed camera state.
8. **Detect beacon** — the detector processes the frame (grayscale → threshold → morphology → CC → centroid). The detector receives only the frame.
9. **Update Kalman filter** — update on detection; predict on miss.
10. **Run state machine** — FSM transitions based on detection/miss counts.
11. **Compute controller command** — pixel error → angular error → PID → rate-limited pan/tilt command.
12. **Update camera pose** — `sim.set_camera_pose()` for the next frame.
13. **Log to CSV** — one row written.
14. **Display HUD** — frame annotated and shown.

### 7.5 Ground-Truth Isolation

```
sim.step() ──── GroundTruth ────► Logger (CSV / JSON)
                 │
                 └──────────────► HUD overlay (dev mode only, disabled in eval mode)

sim.step() ──── frame ──────────► Detector ──► Tracker ──► Controller
                                                   │
                                                   └──► Logger
```

---

## 8. Module Specifications

### 8.1 config.py (v1.0) / YAML Configuration (v1.1)

**v1.0:** Python dataclass with all parameters. Fast to change, type-safe.

**v1.1:** Versioned YAML files in `scenarios/`. Schema-validated. Example structure:

```yaml
simulation:
  duration_s: 60
  fps: 30
  random_seed: 42

camera:
  width: 1280
  height: 720
  horizontal_fov_deg: 60
  vertical_fov_deg: 40
  pan_rate_deg_s: 60
  tilt_rate_deg_s: 60

beacons:
  - id: beacon_01
    color: [255, 255, 255]
    intensity: 1.0
    trajectory: circular
    initial_position_deg: [10, 5]

disturbances:
  platform_vibration_rms_deg: 0.15
  atmospheric_jitter_rms_deg: 0.10
  gaussian_noise_sigma: 8
  motion_blur_pixels: 2
  false_beacon_rate_hz: 0.2

controller:
  mode: pid
  kp: 0.8
  ki: 0.05
  kd: 0.1
  deadband_pixels: 3
```

### 8.2 sim.py — Environment, Camera, Motion, Disturbances, Renderer

| Class / Function | Description |
|-----------------|-------------|
| `CameraState` | Camera pose + intrinsics with derived properties |
| `TargetState` | Beacon state: azimuth, elevation, velocity, acceleration |
| `GroundTruth` | Per-frame truth record |
| `angular_to_pixel()` | Project angular position to pixel coordinates |
| `pixel_to_angular()` | Inverse projection |
| `is_visible()` | Visibility check |
| `ConstantVelocity` | Motion model: fixed angular rate |
| `SinusoidalMotion` | Motion model: amplitude/frequency/phase per axis |
| `CircularMotion` | Motion model: constant speed around a center |
| `RandomManeuvering` | Motion model: bounded acceleration, random intervals |
| `ScriptedTrajectory` (v1.1) | Motion model: replay from file |
| `PlatformVibration` | Disturbance: sinusoidal + jitter on camera boresight |
| `AtmosphericTurbulence` (v1.1) | Disturbance: low-frequency random angular displacement |
| `SensorNoise` | Disturbance: Gaussian + Poisson noise on pixels |
| `MotionBlur` | Disturbance: directional kernel from velocity |
| `ExposureVariation` (v1.1) | Disturbance: brightness/gain modulation |
| `Occlusion` (v1.1) | Disturbance: scripted masked regions |
| `FalseBeacons` (v1.1) | Disturbance: distractor points |
| `FrameRenderer` | Renders background + Gaussian beacon |
| `Simulator` | Main orchestrator; `step()` returns `(frame, GroundTruth)` |

### 8.3 detect.py — Detector, Kalman Filter, Tracker

| Class | Description |
|-------|-------------|
| `DetectorBase` (v1.1) | Common interface for classical and AI detectors |
| `BeaconDetector` | Classical: grayscale → threshold → morphology → CC → centroid |
| `AIDetector` (v1.1) | ONNX-based detector behind `DetectorBase` |
| `KalmanFilter2D` | 4-state constant-velocity Kalman filter |
| `AlphaBetaFilter` (v1.1) | Simpler alternative filter |
| `Tracker` | Wraps detector + filter + FSM; `track()` entry point |

**Detector pipeline (classical):**
1. Convert to grayscale and HSV
2. Brightness and color thresholding
3. Morphological opening and closing
4. Connected components or contours
5. Reject candidates outside size/shape/intensity/persistence limits
6. Compute centroid, area, bounding box, confidence
7. Select candidate matching designated identity or best track

**Common detector interface (v1.1):**
```python
class DetectorBase:
    def detect(self, frame: np.ndarray) -> Detection:
        ...
```

### 8.4 control.py — PID Controller, Search Pattern, Integrated Controller

| Class | Description |
|-------|-------------|
| `PIDController` | Single-axis PID with rate limit, deadband, anti-windup |
| `PanTiltController` | Two-axis PID; pixel error → angular error → command |
| `SearchPattern` | Horizontal sweep with tilt stepping for SEARCHING |
| `IntegratedController` | Top-level: runs tracker + controller + search per frame |

**Error conversion:**
```
error_pan  = (target_x - cx) / width  * h_fov_deg
error_tilt = (cy - target_y) / height * v_fov_deg   # y inverted
```

### 8.5 main.py — Main Loop, Display, Logging, Plots

| Component | Description |
|-----------|-------------|
| `make_scenario()` | Creates simulator + controller from config |
| `draw_hud()` | Draws HUD overlay on frame |
| `CSVLogger` | Per-frame CSV writer with summary computation |
| `generate_plots()` | Two-panel Matplotlib error + detection plot |
| `run_scenario()` | Runs one scenario end-to-end |
| `main()` | Entry point; headless detection, scenario selection |

### 8.6 evaluation/ (v1.1) — Evaluation Framework

| Component | Description |
|-----------|-------------|
| `ScenarioRunner` | Runs all 8 evaluation categories with multiple seeds |
| `MetricsCalculator` | Computes precision, recall, acquisition time, lock retention, etc. |
| `EvaluationMatrix` | Weighted scoring across categories |
| `ReportGenerator` | HTML/PDF report from results |

### 8.7 gui/ (v1.1) — PySide6 GUI

| Component | Description |
|-----------|-------------|
| `MainWindow` | Video display, HUD, controls, config panel |
| `ConfigPanel` | Scenario parameters, disturbance toggles, detector selection |
| `PlotWidget` | Embedded real-time error plot (PyQtGraph) |
| `ReportDialog` | Export report with options |

---

## 9. Data Models and Interfaces

### 9.1 Core Data Structures

#### CameraState
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| pan_deg | float | degrees | Camera azimuth (boresight) |
| tilt_deg | float | degrees | Camera elevation (boresight) |
| width | int | pixels | Image width |
| height | int | pixels | Image height |
| h_fov_deg | float | degrees | Horizontal field of view |
| v_fov_deg | float | degrees | Vertical field of view |

**Derived:** `cx`, `cy`, `deg_per_pixel_x`, `deg_per_pixel_y`

#### TargetState
| Field | Type | Unit |
|-------|------|------|
| azimuth_deg | float | degrees |
| elevation_deg | float | degrees |
| vel_az_deg_s | float | deg/s |
| vel_el_deg_s | float | deg/s |
| acc_az_deg_s2 | float | deg/s² |
| acc_el_deg_s2 | float | deg/s² |

#### GroundTruth
| Field | Type | Description |
|-------|------|-------------|
| target_az_deg | float | True target azimuth |
| target_el_deg | float | True target elevation |
| target_pixel_x | float | True target x (pixels) |
| target_pixel_y | float | True target y (pixels) |
| target_visible | bool | Inside FOV flag |
| camera_pan_deg | float | Disturbed camera pan |
| camera_tilt_deg | float | Disturbed camera tilt |
| timestamp | float | Simulation time (s) |
| frame_id | int | Frame index |

#### Detection
| Field | Type | Description |
|-------|------|-------------|
| detected | bool | Whether a beacon was found |
| cx | float | Centroid x (pixels) |
| cy | float | Centroid y (pixels) |
| area | float | Blob area (pixels²) |
| bbox | tuple | (x, y, w, h) |
| confidence | float | 0–1, brightness-based |
| beacon_id | int (v1.1) | Identity for multi-target |

#### TrackerOutput
| Field | Type | Description |
|-------|------|-------------|
| state | TrackerState | Current FSM state |
| estimated_x | float | Kalman-estimated x |
| estimated_y | float | Kalman-estimated y |
| confidence | float | Detection confidence |
| detected | bool | Detection this frame |
| consecutive_detections | int | Detection streak |
| consecutive_misses | int | Miss streak |
| tracked_id | int (v1.1) | Which beacon is tracked |

#### ControllerOutput
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| pan_cmd_deg | float | degrees | Commanded pan |
| tilt_cmd_deg | float | degrees | Commanded tilt |
| pan_rate_cmd_deg_s | float | deg/s | Commanded pan rate |
| tilt_rate_cmd_deg_s | float | deg/s | Commanded tilt rate |
| error_pan_deg | float | degrees | Angular pan error |
| error_tilt_deg | float | degrees | Angular tilt error |
| error_pixels | float | pixels | Euclidean pixel error |
| saturated | bool | — | Rate limit applied flag |

### 9.2 CSV Log Schema

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Unique run identifier |
| timestamp | float | Simulation time (s) |
| frame_id | int | Frame index |
| fps | float | Measured FPS |
| target_id | string | Beacon identifier |
| gt_x | float | Ground-truth x (pixels) |
| gt_y | float | Ground-truth y (pixels) |
| est_x | float | Estimated x (pixels) |
| est_y | float | Estimated y (pixels) |
| error_px | float | Pixel error from center |
| angular_error_deg | float | Angular error |
| detection_confidence | float | 0–1 |
| detected | int | 1/0 |
| tracker_state | string | FSM state |
| pan_cmd | float | Commanded pan |
| tilt_cmd | float | Commanded tilt |
| pan_actual | float | Actual pan |
| tilt_actual | float | Actual tilt |
| processing_time_ms | float | Per-frame processing time |

### 9.3 Module Interface Contracts

| Caller | Callee | Method | Input → Output |
|--------|--------|--------|----------------|
| main.py | sim.py | `Simulator.step(frame_id)` | int → (ndarray, GroundTruth) |
| main.py | control.py | `IntegratedController.update(frame, pan, tilt)` | ndarray, float, float → ControllerOutput |
| control.py | detect.py | `Tracker.track(frame)` | ndarray → TrackerOutput |
| detect.py | (internal) | `DetectorBase.detect(frame)` | ndarray → Detection |
| detect.py | (internal) | `KalmanFilter2D.predict()` | → (float, float) |
| detect.py | (internal) | `KalmanFilter2D.update(x, y)` | float, float → |
| main.py | sim.py | `Simulator.set_camera_pose(pan, tilt)` | float, float → |
| main.py | (internal) | `CSVLogger.log(...)` | various → |
| main.py | (internal) | `generate_plots(csv, out, name)` | str, str, str → PNG |
| evaluation/ | (internal) | `ScenarioRunner.run_all()` | → results dict |

---

## 10. Configuration System

### 10.1 v1.0: config.py

Python dataclass with all parameters. See §8.1.

### 10.2 v1.1: YAML Configuration

Versioned YAML files in `scenarios/`. Each file defines a complete experiment: simulation parameters, camera, beacons, disturbances, and controller. Schema-validated on load.

### 10.3 Parameter Reference

#### Simulation
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| fps | 30 | Hz | Loop frequency |
| duration_s | 60 | seconds | Run duration |
| random_seed | 42 | — | Seed for all RNGs |

#### Camera
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| width | 1280 | pixels | Image width |
| height | 720 | pixels | Image height |
| h_fov_deg | 60.0 | degrees | Horizontal FOV |
| v_fov_deg | 40.0 | degrees | Vertical FOV |
| pan_range | (-180, 180) | degrees | Pan limits |
| tilt_range | (-30, 90) | degrees | Tilt limits |
| rate_limit_deg_s | 60.0 | deg/s | Max angular velocity |

#### Beacon
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| beacon_sigma_px | 4.0 | pixels | Gaussian blob radius |
| beacon_peak | 255.0 | 0–255 | Peak brightness |
| beacon_size_range | 3–20 | pixels | Near/far size range |

#### Detector
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| threshold | 80 | 0–255 | Brightness threshold |
| min_area | 2 | pixels² | Min blob size |
| max_area | 500 | pixels² | Max blob size |

#### Kalman Filter
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| measurement_noise | 8.0 | pixels² | R covariance |
| process_noise_pos | 1.0 | pixels² | Q position |
| process_noise_vel | 0.5 | pixels² | Q velocity |

#### Controller
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| kp | 1.0 | — | Proportional gain |
| ki | 0.0 | — | Integral gain |
| kd | 0.15 | — | Derivative gain |
| deadband_pixels | 3.0 | pixels | Error deadband |

#### Tracker FSM
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| acquire_threshold | 3 | frames | Detections for TRACKING |
| lose_threshold | 5 | frames | Misses for REACQUIRING |
| reacquire_timeout | 150 | frames | Timeout (~5s) |

#### Disturbances (v1.1 full suite)
| Parameter | Default | Unit | Effect |
|-----------|---------|------|--------|
| platform_vibration_rms_deg | 0.15 | degrees | Vibration RMS |
| atmospheric_jitter_rms_deg | 0.10 | degrees | Turbulence RMS |
| gaussian_noise_sigma | 8.0 | 0–255 | Sensor noise σ |
| motion_blur_pixels | 2.0 | pixels | Max blur length |
| exposure_variation_rate | 0.1 | Hz | Brightness modulation |
| occlusion_duration_s | 1.0 | seconds | Occlusion length |
| false_beacon_rate_hz | 0.2 | Hz | Distractor spawn rate |

### 10.4 Tuning Guide

| Symptom | Parameter | Direction |
|---------|-----------|-----------|
| Beacon not detected | `threshold` | Decrease |
| Too many false detections | `threshold` | Increase |
| Controller oscillates | `kp` | Decrease |
| Controller oscillates | `kd` | Increase |
| Steady-state offset | `ki` | Increase from 0 |
| Tracking jittery | `measurement_noise` | Increase |
| Loses lock too easily | `lose_threshold` | Increase |
| Slow reacquisition | `reacquire_timeout` | Decrease |
| Micro-jitter at lock | `deadband_pixels` | Increase |
| Slow sweep in SEARCHING | `rate_limit_deg_s` | Increase |

---

## 11. Virtual Environment and Camera Model

### 11.1 Coordinate System

The environment uses a 2-D angular coordinate system:
- **Azimuth** (pan axis): increases to the right, in degrees.
- **Elevation** (tilt axis): increases upward, in degrees.
- **Image coordinates**: x to the right, y downward (standard image convention).

### 11.2 Background

A vertical brightness gradient (sky-like): configurable top and bottom brightness values. Optional stars or terrain for v1.1.

### 11.3 Beacon Rendering

The beacon is rendered as a 2-D Gaussian blob:
- Configurable sigma (pixels) — controls blob size.
- Configurable peak intensity (0–255).
- Additive blending with clipping — saturates at center, falls off naturally.
- More realistic than a hard pixel and easier to detect with brightness thresholding.

### 11.4 Pinhole Projection

```
pixel_x = cx + (az - pan) / deg_per_pixel_x
pixel_y = cy - (el - tilt) / deg_per_pixel_y    # y inverted: up in angle = down in pixels
```

Inverse:
```
az = pan + (pixel_x - cx) * deg_per_pixel_x
el = tilt + (cy - pixel_y) * deg_per_pixel_y
```

### 11.5 Visibility Check

A target is visible if `0 <= pixel_x < width` and `0 <= pixel_y < height`.

---

## 12. Target Motion Models

### 12.1 Constant Angular Velocity

| Parameter | Description |
|-----------|-------------|
| az_rate_deg_s | Azimuth rate |
| el_rate_deg_s | Elevation rate |

Exposes position, velocity (constant), acceleration (zero).

### 12.2 Sinusoidal Motion

| Parameter | Description |
|-----------|-------------|
| az_amp_deg | Azimuth amplitude |
| az_freq_hz | Azimuth frequency |
| az_phase_deg | Azimuth phase |
| el_amp_deg | Elevation amplitude |
| el_freq_hz | Elevation frequency |
| el_phase_deg | Elevation phase |

### 12.3 Circular Motion

| Parameter | Description |
|-----------|-------------|
| center_az_deg | Circle center azimuth |
| center_el_deg | Circle center elevation |
| radius_deg | Circle radius |
| angular_speed_deg_s | Speed along circle |
| start_angle_deg | Initial angle |

### 12.4 Random Maneuvering

| Parameter | Description |
|-----------|-------------|
| max_acc_deg_s2 | Maximum acceleration |
| change_interval_s | Interval between acceleration changes |
| max_vel_deg_s | Maximum velocity |
| seed | RNG seed |

Bounded acceleration changed at random intervals. Velocity clamped. Reproducible via seed.

### 12.5 Scripted Trajectory (v1.1)

Replays a trajectory from a file. Used for exact test-case replay. The same file produces the same positions every time.

---

## 13. Disturbance and Sensor Models

Each disturbance is a separate class with an `apply()` method. This makes it possible to disable one effect at a time during debugging and to create controlled evaluation levels.

### 13.1 Platform Vibration

| Property | Value |
|----------|-------|
| Model | Sum of sinusoidal signals at mechanical frequencies (5, 12, 25, 47 Hz) plus Gaussian jitter |
| Effect | Moves the image rapidly around the true position |
| Parameters | RMS (degrees), frequency list, seed |
| Applied to | Camera boresight (copy) |

### 13.2 Atmospheric Turbulence (v1.1)

| Property | Value |
|----------|-------|
| Model | Low-frequency random angular displacement (Ornstein-Uhlenbeck or filtered noise) |
| Effect | Slowly varying pointing error |
| Parameters | RMS (degrees), correlation time, seed |

### 13.3 Camera Motion (v1.1)

| Property | Value |
|----------|-------|
| Model | Applied pan/tilt offset and velocity |
| Effect | Changes target location even when target is stationary |
| Parameters | Offset (degrees), velocity (deg/s) |

### 13.4 Sensor Noise

| Property | Value |
|----------|-------|
| Model | Gaussian (default) + Poisson (v1.1) |
| Effect | Reduces detection confidence |
| Parameters | Sigma (0–255), seed |
| Implementation | `cv2.randn()` on single channel, broadcast to 3 channels |

### 13.5 Motion Blur

| Property | Value |
|----------|-------|
| Model | Directional kernel based on relative angular velocity |
| Effect | Weakens and elongates the beacon |
| Parameters | Max blur length (pixels), skip threshold |
| Implementation | `cv2.filter2D()` with directional kernel |

### 13.6 Exposure Variation (v1.1)

| Property | Value |
|----------|-------|
| Model | Brightness and gain modulation (sinusoidal or random) |
| Effect | Changes beacon contrast |
| Parameters | Rate (Hz), amplitude |

### 13.7 Occlusion (v1.1)

| Property | Value |
|----------|-------|
| Model | Random or scripted masked regions |
| Effect | Forces temporary track loss and reacquisition |
| Parameters | Duration (seconds), frequency, shape |

### 13.8 False Beacons (v1.1)

| Property | Value |
|----------|-------|
| Model | Distractor points with similar appearance |
| Effect | Tests identification robustness |
| Parameters | Spawn rate (Hz), brightness, size, persistence |

---

## 14. Beacon Detection and Identification

### 14.1 Classical Detector (Baseline)

**Pipeline:**
1. Convert to grayscale (and HSV for color beacons)
2. Brightness threshold (fixed default 80, or Otsu with floor of 60)
3. Morphological opening (3×3 elliptical kernel, 1 iteration)
4. Connected components with stats
5. Filter by area (min 2, max 500 pixels)
6. Reject by shape, intensity, or persistence (v1.1)
7. Compute centroid, area, bounding box, confidence
8. Select brightest surviving candidate

**Confidence:** `min(1.0, mean_brightness / threshold)`

**Supported beacon colors:** white (default), red, green, infrared-like (v1.1)

### 14.2 AI Detector (Optional / Comparative)

**Architecture:** Lightweight object detector (e.g., YOLO-nano, SSD) trained on synthetic frames.

**Training workflow:**
1. Generate thousands of labeled frames across positions, sizes, brightness, disturbances, backgrounds
2. Split by scenario seed (not adjacent frames) into train/val/test
3. Train a small detector suitable for real-time inference
4. Export to ONNX
5. Compare against classical baseline on identical test sequences

**Fallback:** If AI confidence < threshold, use classical detector result.

**Common interface:**
```python
class DetectorBase:
    def detect(self, frame: np.ndarray) -> Detection: ...
```

### 14.3 Detector Selection

The detector is selectable via configuration (`detector: classical` or `detector: ai`). The system loads the appropriate implementation at startup. Both produce the same `Detection` output.

---

## 15. Tracking Filter and State Machine

### 15.1 Kalman Filter

| Property | Value |
|----------|-------|
| State vector | [x, y, vx, vy] (position + velocity in pixels) |
| Measurement | [x, y] (centroid from detector) |
| State transition | Constant velocity (dt = 1 frame) |
| Predict | Every frame |
| Update | Only when `detected=True` |
| During gaps | Extrapolates from last velocity |

### 15.2 Alpha-Beta Filter (v1.1 alternative)

Simpler alternative: 15 lines of code. Sufficient when full Kalman is unnecessary.

### 15.3 State Machine

**v1.0 (4 states):**

| State | Behavior | Exit Condition |
|-------|----------|----------------|
| SEARCHING | Sweep or spiral the camera | Valid detection or timeout |
| ACQUIRING | Move toward target; require N detections | Stable candidate (N=3) or rejection |
| TRACKING | Continuous feedback control | Lost target (N misses) or scenario end |
| REACQUIRING | Predict briefly, search locally | Target reacquired or timeout |

**v1.1 (7 states):**

| State | Behavior | Exit Condition |
|-------|----------|----------------|
| IDLE | Wait for scenario start | Start command |
| SEARCHING | Sweep or spiral | Valid detection or timeout |
| CANDIDATE VERIFICATION | Require several consistent detections | Stable candidate or rejection |
| ACQUIRING | Move target toward central region | Error below threshold |
| TRACKING | Continuous feedback | Lost target or scenario end |
| REACQUIRING | Predict, then search around last known | Target reacquired or timeout |
| FAILED | Record failure, stop or restart | Manual reset |

### 15.4 State Transition Diagram

```
IDLE ──start──► SEARCHING
SEARCHING ──detection──► CANDIDATE_VERIFICATION (v1.1) / ACQUIRING (v1.0)
CANDIDATE_VERIFICATION ──N detections──► ACQUIRING
CANDIDATE_VERIFICATION ──rejection──► SEARCHING
ACQUIRING ──error < threshold──► TRACKING
ACQUIRING ──timeout──► SEARCHING
TRACKING ──N misses──► REACQUIRING
REACQUIRING ──detection──► TRACKING
REACQUIRING ──timeout──► SEARCHING
Any ──fatal error──► FAILED (v1.1)
FAILED ──reset──► IDLE
```

---

## 16. Pan-Tilt Controller

### 16.1 Error Conversion

```
error_pan  = (target_x - cx) / width  * h_fov_deg
error_tilt = (cy - target_y) / height * v_fov_deg
```

Note: y is inverted because image y increases downward but angular elevation increases upward.

### 16.2 PID Controller

| Feature | Implementation |
|---------|---------------|
| Proportional | `P = kp * error` |
| Integral | `I += error * dt` (with anti-windup clamp) |
| Derivative | `D = kd * (error - prev_error) / dt` |
| Output | `new_output = current + (P + I + D) * dt` |
| Rate limit | `delta = clip(delta, -rate_limit * dt, rate_limit * dt)` |
| Deadband | If `abs(error) < deadband`, set error to 0 |
| Saturation | Clip to pan/tilt range |

### 16.3 Controller Sequence (Per Frame)

1. Compute filtered target error
2. Convert pixel error to angular error
3. Apply PID control
4. Limit commanded angular velocity
5. Update camera pan and tilt
6. Record command, actual pose, and residual error

### 16.4 Tuning Protocol

1. Tune in a disturbance-free scenario (clean)
2. Test progressively stronger vibration, turbulence, noise, and maneuvers
3. Start with P-only; add I only if steady-state offset; add D for damping
4. Verify no oscillation at any disturbance level

---

## 17. AI Detection Method

### 17.1 Architecture

The AI detector is a lightweight object detector trained on synthetic frames generated by the simulator. It outputs a bounding box, class (beacon identity), and confidence score.

### 17.2 Training Data Generation

| Parameter | Value |
|-----------|-------|
| Frame count | 5,000–10,000 |
| Variation | Positions (full FOV), sizes (3–20px), brightness, disturbance levels, backgrounds |
| Labels | Auto-generated from ground truth (bounding box + class) |
| Split | By scenario seed (not adjacent frames) — 70% train, 15% val, 15% test |

### 17.3 Model

| Property | Value |
|----------|-------|
| Architecture | YOLO-nano or SSD-lite (suitable for real-time) |
| Input | 1280×720 (or resized to 640×360 for speed) |
| Output | Bounding box + class + confidence |
| Export | ONNX Runtime for cross-platform inference |
| Inference target | < 30ms per frame |

### 17.4 Fallback

If the AI detector's confidence is below a configurable threshold, the system falls back to the classical detector for that frame. This prevents AI failures from breaking the closed loop.

### 17.5 Benchmarking

AI detection improvement is claimed **only** if demonstrated through repeatable metrics:
- Precision and recall
- Acquisition time
- Lock-retention rate
- False-lock rate
- Identity-switch rate (multi-target)

The AI and classical detectors are evaluated on identical test sequences with identical seeds.

---

## 18. GUI and Operator Workflow

### 18.1 v1.0: OpenCV HUD

A single OpenCV window with overlaid text and graphics:
- State indicator (color-coded: yellow=searching/acquiring, green=tracking, red=reacquiring)
- Pan/tilt angles
- Pixel error (color-coded by severity)
- Angular error
- FPS / processing time
- Detection status / confidence / streak / misses
- Lock border (green when tracking, red when reacquiring)
- Center crosshair and target zone (central 20%)
- Estimated position (green circle)
- Optional ground-truth cross (red, dev mode only)
- Bottom bar with keyboard controls

**Controls:** SPACE (pause), R (reset), 1 (clean), 2 (hard), G (dev mode), Q/ESC (quit)

### 18.2 v1.1: PySide6 GUI

| Component | Description |
|-----------|-------------|
| Video panel | Live camera feed with HUD overlay |
| Control panel | Start, pause, reset, stop, scenario selector |
| Configuration panel | Target motion, disturbances (toggles + sliders), detector selection |
| Status panel | State, pan/tilt, error, confidence, FPS, lock indicator |
| Plot panel | Embedded real-time error plot (PyQtGraph) |
| Report panel | Export report button, log viewer |

**Threading:** Simulation runs in a worker thread; GUI runs in the main thread. Communication via Qt signals/slots. This prevents the GUI from blocking the simulation loop.

---

## 19. Logging and Performance Reports

### 19.1 Per-Frame CSV Log

One row per frame, 19 fields (see §9.2). Header comments include run_id, seed, and timestamp.

### 19.2 Summary Report

| Metric | Description |
|--------|-------------|
| Simulation duration | Total run time |
| Average and minimum FPS | Rendering performance |
| Average and maximum processing time | Per-frame latency |
| Acquisition time | Time to first TRACKING state |
| Detection precision and recall | Detection quality |
| Average and maximum tracking error | Pixel error from center |
| Root-mean-square angular error | Angular tracking quality |
| Lock-retention rate | Percentage of frames in TRACKING |
| Number and duration of target losses | Loss events and recovery time |
| Reacquisition time | Time to recover from loss |
| Percentage of time inside central tracking region | Coarse-alignment quality |
| Controller saturation time | Time spent at rate limit |
| Results by disturbance level and trajectory | Breakdown by scenario |
| Identity-switch rate (v1.1) | Multi-target tracking quality |

### 19.3 Output Formats

| Format | When | Content |
|--------|------|---------|
| CSV | Every run | Per-frame raw data |
| JSON | Every run (v1.1) | Structured summary |
| PNG | Every run | Error-over-time plot |
| HTML/PDF | On export (v1.1) | Full report with config, metrics, plots, limitations |

---

## 20. Evaluation Framework

### 20.1 Evaluation Categories

| Category | Example Cases | Main Metrics |
|----------|-------------|---------------|
| Nominal | Static background, smooth target, no disturbances | FPS, acquisition time, error |
| Motion | Fast and maneuvering target | Lock retention, maximum error |
| Vibration | Low, medium, and high platform vibration | RMS angular error, reacquisition |
| Turbulence | Slow and fast random angular jitter | Lock retention, control stability |
| Sensor degradation | Noise, blur, exposure changes | Precision, recall, FPS |
| Clutter | False beacons and background points | Identification accuracy, false-lock rate |
| Occlusion | Temporary target disappearance | Reacquisition time, loss duration |
| Multi-target | Crossing or nearby beacons | Identity-switch rate |

### 20.2 Evaluation Weighting

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| Detection and identification | 20% | Precision, recall, false-lock rate |
| Acquisition performance | 20% | Time to enter central region |
| Continuous tracking | 25% | Lock-retention rate and tracking error |
| Disturbance robustness | 20% | Performance degradation across disturbance levels |
| Real-time execution | 10% | FPS and processing latency |
| Reproducibility and usability | 5% | Configuration, logging, documentation, repeatability |

### 20.3 Evaluation Protocol

1. Run each scenario at least 5 times with different seeds
2. Report mean, standard deviation, and worst-case values
3. Preserve random seed and configuration with every result
4. Compare detectors on identical sequences
5. Generate the evaluation matrix CSV with all categories and weights

---

## 21. Validation and Testing

### 21.1 Validation Layers

| Layer | Scope | Method |
|-------|-------|--------|
| Unit tests | Coordinate conversion, FOV checks, trajectory generation, disturbance statistics, PID saturation, metric calculations | pytest (v1.1), module self-tests (v1.0) |
| Component tests | Rendering, detection, tracking, logging independently | Module self-tests |
| Closed-loop tests | Controller centers target in ideal conditions | `control.py` self-test |
| Robustness tests | Increase one disturbance at a time, measure degradation | Headless benchmark |
| Stress tests | Increase target count, resolution, scene complexity | Manual |
| Replay tests | Identical seeds produce identical metrics | `FSOC_TEST_MODE=1 python main.py` |
| Packaging tests | Standalone executable on clean machine | Manual (v1.1) |
| Acceptance tests | Complete evaluation matrix and final report | §27 |

### 21.2 Self-Test Coverage (v1.0)

| Module | Tests |
|--------|-------|
| sim.py | Projection (4 assertions), motion models (4 verified), rendering (90-frame video), disturbances (noise std) |
| detect.py | Detector accuracy (sub-pixel), Kalman smoothing, state machine (all transitions), disturbance robustness (60/60 detections) |
| control.py | PID convergence, rate limiting, deadband, closed loop (99.3% lock, 13px error), disturbance robustness (12.5px mean) |

---

## 22. Packaging and Distribution

### 22.1 Repository Structure

```
fsoc-virtual-tracker/
├── app/
│   ├── main.py
│   ├── gui/                (v1.1)
│   ├── simulation/
│   ├── vision/
│   ├── control/
│   ├── evaluation/         (v1.1)
│   └── logging/
├── configs/
├── scenarios/              (v1.1: YAML files)
├── tests/                  (v1.1: pytest)
├── models/                 (v1.1: ONNX models)
├── docs/                   (technical report, user manual)
├── requirements.txt
├── README.md
└── build/                  (PyInstaller output, v1.1)
```

### 22.2 Dependencies

```
# requirements.txt
opencv-python>=4.8
numpy>=1.24
scipy>=1.10          # optional (v1.1)
matplotlib>=3.7
PySide6>=6.5         # v1.1 GUI
onnxruntime>=1.15     # v1.1 AI detector
# PyTorch is only needed for training, not for inference
```

### 22.3 PyInstaller (v1.1)

The release archive contains:
- Standalone executable
- Default scenarios
- Model files (if AI detector used)
- Configuration examples
- User manual

---

## 23. Technical Report Structure

The required 10–15 page technical report follows this structure:

1. **Problem understanding and FSOC coarse-alignment context**
2. **Requirements and assumed parameters** (the parameters and specifications table)
3. **System architecture** (module map, data flow, ground-truth isolation)
4. **Virtual environment and camera model** (coordinate system, projection, rendering)
5. **Disturbance and sensor models** (all 8 disturbance types, implementation approach)
6. **Beacon detection and identification methods** (classical pipeline, AI method)
7. **Tracking filter, state machine, and pan-tilt controller** (Kalman, FSM, PID)
8. **AI method and training data** (if used)
9. **Experimental design and evaluation criteria** (8 categories, weighted scoring)
10. **Performance results and discussion** (measured metrics, comparison plots)
11. **Limitations, risks, and future improvements**
12. **Conclusion**

---

## 24. User Manual Structure

The user manual explains:

1. **Installation** — system requirements, dependency installation
2. **System requirements** — OS, Python version, hardware
3. **Application startup** — how to run
4. **GUI controls** — keyboard and mouse reference
5. **Configuration parameters** — all configurable values
6. **Scenario creation** — how to write a YAML scenario
7. **Detector selection** — classical vs. AI
8. **Disturbance settings** — how to enable/configure each
9. **Experiment execution** — running scenarios
10. **Log export** — CSV, JSON, plots
11. **Report interpretation** — how to read metrics
12. **Troubleshooting** — common issues and fixes
13. **Known limitations**

Includes at least one **complete example experiment** from configuration through report generation.

---

## 25. Scenarios and Evaluation

### 25.1 Scenario Definitions

#### Clean Scenario (Nominal)

| Property | Value |
|----------|-------|
| Motion model | Circular |
| Center (az, el) | (5°, 3°) |
| Radius | 8° |
| Angular speed | 4°/s |
| Disturbances | None |
| Purpose | Baseline tracking performance |

#### Hard Scenario (Disturbed)

| Property | Value |
|----------|-------|
| Motion model | Sinusoidal |
| Azimuth amplitude | 10°, frequency 0.2 Hz |
| Elevation amplitude | 6°, frequency 0.3 Hz |
| Vibration RMS | 0.2° |
| Sensor noise σ | 12.0 |
| Motion blur max | 3.0 px |
| Purpose | Robustness under realistic disturbances |

### 25.2 v1.1 Evaluation Scenarios (8 categories)

| Category | Motion | Disturbances | Duration |
|----------|--------|-------------|----------|
| Nominal | Circular, slow | None | 60s |
| Motion | Random maneuvering, fast | None | 60s |
| Vibration | Circular | Vibration (low/med/high) | 60s each |
| Turbulence | Circular | Turbulence (slow/fast) | 60s each |
| Sensor degradation | Circular | Noise + blur + exposure | 60s |
| Clutter | Circular | False beacons | 60s |
| Occlusion | Circular | Scripted occlusions | 60s |
| Multi-target | Two crossing beacons | None | 60s |

### 25.3 Metrics

| Metric | Formula | Clean target | Hard target |
|--------|---------|-------------|-------------|
| Acquisition time | First TRACKING frame / FPS | < 2s | < 2s |
| Mean pixel error | mean(‖gt - center‖) | < 50px | < 200px |
| Steady-state error | mean(error) last 30 frames | < 15px | < 100px |
| Lock retention | count(TRACKING)/total × 100 | > 95% | > 90% |
| Detection rate | count(detected)/total × 100 | 100% | > 95% |
| RMS angular error | sqrt(mean(ang_err²)) | < 3° | < 8° |
| Loss events | count(TRACKING→REACQUIRING) | 0 | < 3 |
| Identity-switch rate (v1.1) | switches/crossings | N/A | < 10% |

### 25.4 Measured Performance (v1.0, 10s runs, seed=42)

| Metric | Clean | Hard |
|--------|-------|------|
| Acquisition time | 0.07s | 0.07s |
| Mean pixel error | 35px | 113px |
| Steady-state error | 10px | 88px |
| Max pixel error | 264px | 182px |
| Lock retention | 99.3% | 99.3% |
| Detection rate | 100% | 100% |
| RMS angular error | 2.9° | 5.9° |
| Loss events | 0 | 0 |
| Mean FPS | 45 | 24 |

### 25.5 Evaluation Protocol

1. Set `random_seed` to a fixed value
2. Run each scenario for the configured duration (default 60s)
3. Record CSV log, error plot, summary statistics
4. Repeat with at least 3 different seeds
5. Report mean, standard deviation, worst-case
6. Preserve seed and configuration with every result

---

## 26. Performance Requirements

### 26.1 Timing Budget (Per Frame)

| Component | Clean | Hard | Budget |
|-----------|-------|------|--------|
| Target motion update | < 0.1ms | < 0.1ms | 1ms |
| Camera copy + vibration | < 0.1ms | 0.5ms | 2ms |
| Frame rendering | 14ms | 14ms | 20ms |
| Motion blur | 0ms | 3ms | 5ms |
| Sensor noise | 0ms | 15ms | 20ms |
| Detection | 2ms | 3ms | 5ms |
| Kalman + FSM | < 0.1ms | < 0.1ms | 1ms |
| PID controller | < 0.1ms | < 0.1ms | 1ms |
| CSV logging | < 0.1ms | < 0.1ms | 1ms |
| HUD rendering | 1ms | 1ms | 5ms |
| **Total** | **~18ms** | **~38ms** | **< 60ms** |

### 26.2 Accuracy Requirements

| Parameter | Requirement |
|-----------|-------------|
| Projection round-trip error | < 1e-6 degrees |
| Detector centroid error (clean) | < 0.5 pixels |
| Kalman filtered error | ≤ raw error × 1.5 |
| Steady-state error (clean) | < 15 pixels |
| Reproducibility | Identical seed → identical metrics |

---

## 27. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| AC1 | Application runs without specialized hardware | Manual |
| AC2 | Scenario can be configured, started, paused, reset, replayed | Manual |
| AC3 | At least one moving beacon detected and tracked through synthetic video | `python detect.py` |
| AC4 | Virtual camera changes pan/tilt in response to target error | `python control.py` |
| AC5 | System supports target loss and reacquisition | `python detect.py` |
| AC6 | Disturbances can be enabled independently and in combination | `python sim.py` |
| AC7 | Application displays real-time tracking state and statistics | Manual |
| AC8 | Every run produces raw logs and summary performance report | Manual |
| AC9 | Baseline and optional AI detector compared on same scenarios | Headless mode |
| AC10 | Executable, source code, technical report, user manual delivered together | Manual |
| AC11 | Evaluation results include config, seed, metrics, plots, limitations | Manual |
| AC12 | All module self-tests pass | `python sim.py && python detect.py && python control.py` |
| AC13 | System runs headless with CSV + plot output | `FSOC_TEST_MODE=1 python main.py` |
| AC14 | Demo runs reliably for 60+ seconds without crashing | Manual |
| AC15 | 8-category evaluation matrix produced (v1.1) | Headless eval mode |
| AC16 | AI detector trained and benchmarked (v1.1, if included) | Report comparison |
| AC17 | PySide6 GUI with config controls and report export (v1.1) | Manual |
| AC18 | PyInstaller executable runs on clean machine (v1.1) | Clean-machine test |

---

## 28. Technical Constraints

### 28.1 Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | Rapid sim, CV, ML ecosystem |
| Image processing | OpenCV | 4.8+ | Rendering, display, CC, filtering |
| Numerical | NumPy | 1.24+ | Array ops, projection |
| Signal processing | SciPy | 1.10+ (v1.1) | Signal generation, filtering |
| GUI | PySide6 | 6.5+ (v1.1) | Desktop application |
| Plotting | Matplotlib | 3.7+ | Post-run plots |
| Real-time plots | PyQtGraph | 0.13+ (v1.1) | Embedded error plots |
| ML utilities | scikit-learn | 1.3+ (v1.1) | Non-DL evaluation utilities |
| AI inference | ONNX Runtime | 1.15+ (v1.1) | Cross-platform model inference |
| AI training | PyTorch | 2.0+ (v1.1, optional) | Training only, not inference |
| Packaging | PyInstaller | 6.0+ (v1.1) | Standalone executable |

### 28.2 Platform Constraints

- OS: Linux, macOS, Windows
- No GPU required for baseline (v1.0) or AI inference (v1.1 ONNX)
- GPU recommended for AI training only
- No root/sudo access
- No network access required (fully offline)
- Display optional (headless mode)
- Disk: < 10MB source, < 500MB with all dependencies + models

---

## 29. Risisks and Mitigations

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|-----------|------------|
| R1 | Projection math is wrong | High | Low | Self-test: round-trip, boresight-to-center, edge mapping |
| R2 | Controller oscillation | Medium | Medium | Start P-only; add deadband; tune in ideal first; add I/D gradually |
| R3 | GUI blocks simulation | Medium | Low | Run sim in worker thread; Qt signals/slots; v1.0 uses non-blocking imshow |
| R4 | Kalman filter diverges | Low | Low | Conservative process noise; alpha-beta fallback |
| R5 | AI detector overfits synthetic images | High | Medium | Use varied procedural generation; report synthetic-domain limits; benchmark on test seeds not seen in training |
| R6 | Ground-truth leakage | High | Very Low | Strict isolation; dev-mode overlay disabled in eval mode; code review |
| R7 | Non-reproducible experiments | Medium | Low | Seed in CSV header; all RNGs seeded; fixed timestep |
| R8 | Unrealistic disturbance model | Medium | Medium | Parameterize models; document assumptions; allow per-effect tuning |
| R9 | Packaging failures | High | Low | Test release on clean system before submission |
| R10 | Multi-target identity switches | Medium | Medium | Add identity features (color), track continuity, explicit switch metrics |
| R11 | Controller tilt sign error | High | Low (found/fixed) | Negate y in error conversion; verified by closed-loop test |
| R12 | Otsu threshold too low | Medium | Medium | Default to fixed threshold; Otsu mode with configurable floor |
| R13 | Sensor noise performance | Medium | Medium | Use `cv2.randn()` instead of `numpy.Generator.normal`; single-channel + broadcast |
| R14 | Demo crashes on stage | High | Low | Screen capture backup; test from clean venv |
| R15 | Running out of time | High | Medium | Build order ensures demoable loop by Hour 14; everything after is improvement |

---

## 30. Milestones and Timeline

### 30.1 v1.0 (Hackathon, 48 hours)

| Hours | Phase | Deliverable |
|-------|-------|-------------|
| 0–2 | Setup | Project structure, config, stubs |
| 2–6 | Simulator | sim.py: coordinates, projection, rendering, motion models |
| 6–10 | Detector | detect.py: detection pipeline |
| 10–14 | Controller + Loop | control.py + main.py: first closed loop |
| 14–18 | Tracker + FSM | Kalman filter, 4-state FSM |
| 18–22 | Disturbances | Vibration, noise, blur |
| 22–26 | HUD + Logging | HUD overlay, CSV logger, plots |
| 26–30 | Tuning | PID, Kalman, scenario parameters |
| 30–36 | Demo Scenarios | Clean + hard, verify reliability |
| 36–40 | Stretch | Motion blur, PID I/D, trajectory switching |
| 40–44 | Hardening | Edge cases, README, clean venv test |
| 44–48 | Rehearsal | Full demo timed, backup recording |

### 30.2 v1.1 (Post-Hackathon, 6–10 weeks)

| Phase | Duration | Output |
|-------|----------|--------|
| Requirements and architecture | 1 week | Frozen spec, module interfaces |
| Additional disturbances | 1–2 weeks | Turbulence, exposure, occlusion, false beacons |
| Multi-target scenarios | 1–2 weeks | Identity discrimination, switch metrics |
| GUI (PySide6) | 1–2 weeks | Operator interface with config controls |
| AI detector | 2–4 weeks | Trained, ONNX-exported, benchmarked |
| Evaluation matrix | 2 weeks | 8-category results, weighted scoring |
| YAML configuration | 1 week | Versioned scenario files |
| Packaging (PyInstaller) | 1 week | Standalone executable |
| Technical report + user manual | 1–2 weeks | 10–15 page report, complete manual |

### 30.3 v2.0 (Production, 3–6 months)

| Phase | Duration | Output |
|-------|----------|--------|
| 3-D renderer | 4–6 weeks | Realistic depth, occlusion, platform geometry |
| Hardware-in-the-loop | 4–8 weeks | Real pan-tilt gimbal integration |
| Real camera feed support | 2–4 weeks | Process external video |
| Full evaluation framework | 2 weeks | Automated batch evaluation, CI integration |
| Production hardening | 2–4 weeks | Error handling, logging, documentation |

---

## 31. Future Roadmap

### 31.1 Research Directions

- Compare classical vs. AI detector on synthetic and (if available) real beacon data
- Study trade-off between detection confidence and tracking stability
- Evaluate LQR or MPC as alternatives to PID
- Investigate adaptive thresholding for varying backgrounds
- Explore particle filters for non-Gaussian target motion
- Study multi-target data association algorithms (JPDA, MHT)

### 31.2 Integration Opportunities

- Port control layer to real pan-tilt gimbal for hardware-in-the-loop
- Connect to real FSOC terminal for end-to-end PAT demonstration
- Integrate with link-level simulation for end-to-end communication evaluation
- Cloud-based batch evaluation service

---

## 32. Glossary

| Term | Definition |
|------|-----------|
| **FSOC** | Free-Space Optical Communication — data transmission via laser through air or vacuum |
| **Coarse alignment** | Bringing the beacon into the camera's FOV before fine pointing takes over |
| **Fine pointing** | Precise beam steering (sub-milliradian) for communication — not in scope |
| **Beacon** | An optical signal emitted by the remote terminal for acquisition |
| **Boresight** | The optical axis of the camera — center of the FOV |
| **FOV** | Field of View — the angular extent visible to the camera |
| **Pan** | Horizontal rotation (azimuth) of the camera |
| **Tilt** | Vertical rotation (elevation) of the camera |
| **Pinhole projection** | Camera model: angular position maps linearly to pixel position |
| **Kalman filter** | Recursive estimator combining noisy measurements with a motion model |
| **Alpha-beta filter** | Simplified Kalman filter with separate position and velocity gains |
| **PID controller** | Proportional-Integral-Derivative feedback controller |
| **FSM** | Finite-State Machine — transitions between discrete states based on inputs |
| **Connected components** | Image processing: labeling contiguous pixel regions |
| **RMS** | Root Mean Square — measure of average error magnitude |
| **Lock retention** | Percentage of time in TRACKING state |
| **Acquisition time** | Time from start to first TRACKING frame |
| **Reacquisition** | Recovering tracking after temporary loss |
| **Ground truth** | True target position, known to simulator, hidden from detector/controller |
| **Deadband** | Range of small errors ignored by the controller |
| **Anti-windup** | Prevents integral term from accumulating without bound when saturated |
| **ONNX** | Open Neural Network Exchange — cross-platform model format |
| **Identity switch** | Tracker following the wrong beacon after a crossing (multi-target) |
| **False beacon** | Distractor point with similar appearance to the real beacon |
| **Otsu threshold** | Automatic threshold selection method |
| **PAT** | Pointing, Acquisition, and Tracking — the full FSOC alignment system |

---

## 33. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-09-09 | Team | Initial MVP PRD (hackathon scope) |
| 2.0 | 2026-09-09 | Team | Full product PRD. Covers complete vision: AI detector, 8 disturbance types, multi-target, 8-category evaluation matrix with weighted scoring, PySide6 GUI, YAML config, PyInstaller packaging, technical report, user manual, 3-D renderer path, hardware-in-the-loop. Phased: v1.0 (delivered), v1.1 (post-hackathon), v2.0 (production). |
