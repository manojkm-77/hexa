# Technical Report: FSOC Coarse-Alignment Simulator

**AI-Based Coarse-Alignment Simulator for Mobile Free-Space Optical Communication Terminals**

| Field | Value |
|-------|-------|
| Document version | 1.0 |
| Project | Hexaverse - FSOC Virtual Camera Tracking System |
| Date | 2026-09-09 |
| Classification | Internal / Submission |

---

## Table of Contents

1. [Problem Understanding and FSOC Coarse-Alignment Context](#1-problem-understanding-and-fsoc-coarse-alignment-context)
2. [Requirements and Assumed Parameters](#2-requirements-and-assumed-parameters)
3. [System Architecture](#3-system-architecture)
4. [Virtual Environment and Camera Model](#4-virtual-environment-and-camera-model)
5. [Disturbance and Sensor Models](#5-disturbance-and-sensor-models)
6. [Beacon Detection and Identification Methods](#6-beacon-detection-and-identification-methods)
7. [Tracking Filter, State Machine, and Pan-Tilt Controller](#7-tracking-filter-state-machine-and-pan-tilt-controller)
8. [AI Method and Training Data](#8-ai-method-and-training-data)
9. [Experimental Design and Evaluation Criteria](#9-experimental-design-and-evaluation-criteria)
10. [Performance Results and Discussion](#10-performance-results-and-discussion)
11. [Limitations, Risks, and Future Improvements](#11-limitations-risks-and-future-improvements)
12. [Conclusion](#12-conclusion)

---

## 1. Problem Understanding and FSOC Coarse-Alignment Context

### 1.1 Free-Space Optical Communication

Free-Space Optical (FSOC) communication uses modulated laser beams to transmit data through the atmosphere between two line-of-sight terminals. Unlike radio-frequency systems, FSOC offers high bandwidth, low power consumption, and immunity to electromagnetic interference. However, the narrow beam divergence required for efficient optical coupling demands precise pointing between the transmitting and receiving terminals.

### 1.2 The Coarse-Alignment Problem

Establishing and maintaining an FSOC link requires two sequential alignment phases:

1. **Coarse alignment**: Bringing the remote terminal's optical beacon into the field of view (FOV) of the local terminal's tracking camera. This is the domain addressed by the present simulator.
2. **Fine pointing**: Sub-microradian laser beam steering to optimize coupling efficiency after the beacon is acquired. This is explicitly out of scope for the current system.

Coarse alignment is typically performed by a pan-tilt camera that scans the sky to detect the remote beacon, then continuously tracks it while the pan-tilt mechanism keeps the beacon centered in the image. The coarse-alignment controller must operate in real time, handle target motion, reject platform disturbances, recover from temporary track loss, and function without specialized optical or mechanical hardware during development and testing.

### 1.3 Rationale for Simulation

Physical FSOC testbeds require precision pan-tilt gimbals, optical benches, beacon sources, and controlled environments. A software-in-the-loop simulator provides several advantages:

- **Reproducibility**: Identical scenarios with deterministic random seeds enable fair comparison of detection and control algorithms.
- **Parametric control**: Disturbance parameters (vibration amplitude, noise level, turbulence strength) can be swept systematically without hardware wear.
- **Rapid iteration**: Algorithm changes can be evaluated in minutes rather than hours of laboratory time.
- **Safety**: No laser safety hazards or mechanical wear on expensive gimbal hardware.

The simulator generates synthetic camera frames in which a Gaussian optical beacon moves along configurable trajectories. The rendered frames are processed by a detection pipeline that operates without access to ground-truth information, and the resulting position estimates drive a closed-loop pan-tilt controller. This architecture enforces the same information-flow constraints as a real system, ensuring that algorithm performance on synthetic data is indicative of real-world behavior.

---

## 2. Requirements and Assumed Parameters

### 2.1 Functional Requirements Summary

The system must satisfy the following top-level functional requirements:

| ID | Requirement |
|----|-------------|
| FR-S1 | Simulate a 2-D angular environment with azimuth (pan) and elevation (tilt) coordinates in degrees |
| FR-S3 | Render the beacon as a 2-D Gaussian blob with configurable sigma and peak intensity |
| FR-S4 | Support at least four target motion models: constant velocity, sinusoidal, circular, and random maneuvering |
| FR-S7 | Use a fixed simulation time step (1/FPS seconds) regardless of rendering speed |
| FR-S8 | Support a configurable random seed for reproducible experiments |
| FR-D1 | Detect the beacon in each rendered frame using a classical brightness-threshold pipeline |
| FR-T1 | Maintain a Kalman filter (constant-velocity model) for position smoothing and prediction through detection gaps |
| FR-T2 | Implement a finite state machine with at least four states: SEARCHING, ACQUIRING, TRACKING, REACQUIRING |
| FR-C1 | Convert image-space tracking error to angular pan/tilt commands via a PID controller with rate limiting |
| FR-L1 | Log per-frame metrics (pixel error, angular error, tracker state, detection confidence) to CSV |
| FR-E1 | Compute summary statistics: acquisition time, mean error, lock retention, loss events |

### 2.2 Camera and Simulation Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Image width | W | 1280 | px |
| Image height | H | 720 | px |
| Horizontal FOV | h_fov | 60 | deg |
| Vertical FOV | v_fov | 40 | deg |
| Pan range | -- | [-180, +180] | deg |
| Tilt range | -- | [-30, +90] | deg |
| Frame rate | FPS | 30 | fps |
| Duration (default) | -- | 60 | s |
| Random seed | -- | 42 (default) | -- |
| Degrees per pixel (H) | dp_x | 0.04688 | deg/px |
| Degrees per pixel (V) | dp_y | 0.05556 | deg/px |

### 2.3 Beacon Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Gaussian sigma | 4.0 | px |
| Peak intensity | 255 | (0-255) |
| Background top brightness | 40 | (0-255) |
| Background bottom brightness | 10 | (0-255) |

### 2.4 Detection and Tracking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Threshold method | Fixed (80) | Grayscale brightness threshold for binarization |
| Minimum blob area | 2 px^2 | Rejects noise speckles |
| Maximum blob area | 500 px^2 | Rejects overly large regions |
| Morphological kernel | 3x3 ellipse | Noise removal after thresholding |
| Max aspect ratio | 3.0 | Rejects elongated non-beacon blobs |
| Kalman measurement noise (R) | 8.0 | Higher values reduce trust in raw detections |
| Kalman process noise (Q pos) | 1.0 | Position process noise |
| Kalman process noise (Q vel) | 0.5 | Velocity process noise |

### 2.5 Controller Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Kp (proportional) | 1.0 | -- | Proportional gain |
| Ki (integral) | 0.0 | -- | Integral gain (disabled by default) |
| Kd (derivative) | 0.15 | -- | Derivative gain for damping |
| Deadband | 3.0 | px | Errors below this are ignored |
| Rate limit | 60.0 | deg/s | Maximum pan/tilt slew rate |
| Output range (pan) | [-180, +180] | deg | Pan axis limits |
| Output range (tilt) | [-30, +90] | deg | Tilt axis limits |
| Integral limit (anti-windup) | 30.0 | deg | Integral term clamp |

### 2.6 Tracker FSM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Verify threshold | 2 | Consecutive detections to advance from CANDIDATE_VERIFICATION to ACQUIRING |
| Acquire threshold | 3 | Consecutive detections to advance from ACQUIRING to TRACKING |
| Acquire error threshold | 100 px | Maximum Kalman-detection discrepancy for acquisition |
| Lose threshold | 5 | Consecutive misses to enter REACQUIRING |
| Reacquire timeout | 150 frames | Frames in REACQUIRING before reverting to SEARCHING (~5 s at 30 FPS) |

---

## 3. System Architecture

### 3.1 Module Map

The system is organized into five core modules under `src/`, each with a clearly defined responsibility:

| Module | File | Responsibility |
|--------|------|---------------|
| Simulator | `sim.py` | Virtual environment, camera model, motion models, disturbances, frame rendering |
| Detector | `detect.py` | Classical beacon detection, Kalman filter, alpha-beta filter, tracking FSM |
| Controller | `control.py` | PID controller, pan-tilt control, search patterns, integrated controller |
| Configuration | `config.py` | Central dataclass holding all tunable parameters |
| Main | `main.py` | Entry point, scenario assembly, main loop, HUD, logging, plots |
| Evaluation | `evaluation.py` | Metrics calculation, scenario runner, 8-category weighted scoring matrix |
| AI Detector | `ai_detector.py` | ONNX-based neural network detector with classical fallback |
| Report | `report.py` | Structured JSON summary and HTML report generation |

### 3.2 Data Flow

The data flow enforces strict information isolation. Ground truth is produced by the simulator but never exposed to the detector or controller:

```mermaid
graph LR
    subgraph "Simulator (sim.py)"
        A[Motion Model] --> B[Target State]
        B --> C[Frame Renderer]
        D[Camera State] --> C
        C --> E["Frame (BGR image)"]
        B --> F[Ground Truth]
        D --> F
    end

    subgraph "Detector (detect.py)"
        E --> G[BeaconDetector]
        G --> H["Detection (cx, cy, confidence)"]
        H --> I[KalmanFilter2D]
        I --> J["Estimated Position (x, y)"]
        J --> K[Tracker FSM]
        K --> L["TrackerOutput (state, position)"]
    end

    subgraph "Controller (control.py)"
        L --> M[PanTiltController]
        M --> N["ControllerOutput (pan_cmd, tilt_cmd)"]
    end

    N -->|set_camera_pose| D
    F -.->|"NEVER passes to detector/controller"| O[Logger / HUD only]
```

### 3.3 Ground-Truth Isolation

A fundamental design principle of the system is **ground-truth isolation**. The `GroundTruth` dataclass is returned by `Simulator.step()` alongside each rendered frame and contains the true target position in both angular and pixel coordinates. However, this data is never passed to the `BeaconDetector`, `KalmanFilter2D`, or `PanTiltController`. Only the `CSVLogger` and the developer-mode HUD overlay receive ground-truth information. This constraint ensures that the detector and controller operate under the same informational limitations as a real system, where only the camera image is available.

### 3.4 Configuration Architecture

All tunable parameters reside in a single `Config` dataclass in `config.py`. Other modules read from `Config` but never hardcode parameter values. Scenario-specific parameters (clean vs. hard) are stored as named groups within the same dataclass, enabling straightforward scenario switching at runtime. CLI arguments and YAML configuration files can override any parameter.

### 3.5 Main Loop and Scenario Management

The main loop in `main.py` orchestrates the simulation cycle:

1. **Scenario assembly** (`make_scenario`): Constructs the simulator, tracker, and controller from the current `Config`. Builds the appropriate motion model, disturbance objects, detector, and filter based on the selected scenario and configuration parameters.

2. **Per-frame execution**: For each frame within the configured duration:
   - Apply camera-level disturbances (turbulence) to the camera boresight.
   - Call `sim.step(frame_id)` to advance the target motion model and render the frame.
   - Apply frame-level disturbances (occlusion, false beacons, exposure variation) to the rendered image.
   - Feed the processed frame to the integrated controller, which runs detection, Kalman filtering, state-machine logic, and PID control.
   - Apply the resulting pan/tilt commands to the camera via `sim.set_camera_pose()`.
   - Log per-frame metrics to CSV.

3. **Post-run**: Generate summary statistics, error-over-time plots (Matplotlib), structured JSON summaries, and self-contained HTML reports.

Two pre-built scenarios are provided:

- **Clean**: Circular beacon trajectory (center at 5 deg az, 3 deg el, radius 8 deg, speed 4 deg/s) with no disturbances. Serves as the baseline for ideal tracking performance.
- **Hard**: Sinusoidal trajectory (azimuth amplitude 10 deg at 0.2 Hz, elevation amplitude 6 deg at 0.3 Hz) with platform vibration (0.2 deg RMS), sensor noise (sigma = 12), and motion blur (3 px maximum). Tests robustness under realistic operational conditions.

### 3.6 Logging and Reporting

The `CSVLogger` records one row per frame with 19 fields including ground-truth position, estimated position, pixel error, angular error, detection confidence, tracker state, pan/tilt commands, and processing time. The log includes metadata comments (run ID, seed, timestamp) for traceability.

The `SummaryReporter` reads the CSV and computes all PRD-defined metrics, outputting a structured JSON summary. The `ReportGenerator` produces a self-contained HTML report with embedded configuration, metrics tables, and error-over-time plots (base64-encoded PNG). This enables one-command report generation for any completed run.

---

## 4. Virtual Environment and Camera Model

### 4.1 Coordinate System

The simulator operates in a 2-D angular coordinate system:

- **Azimuth (pan axis)**: Horizontal angular position in degrees, increasing to the right.
- **Elevation (tilt axis)**: Vertical angular position in degrees, increasing upward.
- **Image coordinates**: Pixel coordinates (x rightward, y downward), following the standard image convention.

The camera boresight is defined by its (pan, tilt) pose in angular coordinates. Objects in the world are described by their (azimuth, elevation) position. The projection from angular world coordinates to pixel coordinates determines where each object appears in the image.

### 4.2 Pinhole Camera Projection

The camera model uses a simplified pinhole projection with the following key functions:

**Forward projection** (`angular_to_pixel`): Converts an angular world position to image pixel coordinates.

```
delta_az = az_deg - cam.pan_deg
delta_el = el_deg - cam.tilt_deg

pixel_x = cx + delta_az / deg_per_pixel_x
pixel_y = cy - delta_el / deg_per_pixel_y
```

where `cx = width / 2` and `cy = height / 2` are the image center, and:

```
deg_per_pixel_x = h_fov_deg / width
deg_per_pixel_y = v_fov_deg / height
```

The negation of `delta_el` in the `pixel_y` equation reflects the inverted relationship between elevation (increasing upward) and image y-coordinate (increasing downward).

**Inverse projection** (`pixel_to_angular`): Converts pixel coordinates back to angular world position. This is the exact inverse of the forward projection and is used by the controller to convert image-space error into angular commands:

```
delta_az = (px - cx) * deg_per_pixel_x
delta_el = (cy - py) * deg_per_pixel_y

az_deg = cam.pan_deg + delta_az
el_deg = cam.tilt_deg + delta_el
```

**Visibility check** (`is_visible`): A point is visible if its pixel coordinates satisfy `0 <= px < width` and `0 <= py < height`. Points outside the frame are not rendered but still contribute to ground truth for logging purposes.

### 4.3 Frame Rendering

The `FrameRenderer` class produces synthetic camera frames:

1. **Background**: A vertical brightness gradient from `bg_top` (40) at the top to `bg_bottom` (10) at the bottom, simulating a dark sky. The gradient is replicated across all three BGR channels.

2. **Beacon**: Rendered as a 2-D Gaussian blob centered at the projected pixel position. The intensity profile is:

```
I(x, y) = I_peak * exp(-((x - px)^2 + (y - py)^2) / (2 * sigma^2))
```

The beacon is additively blended with the background within a region of +/- 3 sigma from its center. The peak intensity (255) and sigma (4 px) are configurable.

3. **Optional stars**: When enabled, fixed random positions render small 1-pixel white dots to simulate a star field.

### 4.4 Target Motion Models

The simulator supports five motion models, each implementing the `MotionModel` base class:

| Model | Class | Description | Key Parameters |
|-------|-------|-------------|----------------|
| Constant velocity | `ConstantVelocity` | Linear motion at fixed angular rates | az_rate, el_rate (deg/s) |
| Sinusoidal | `SinusoidalMotion` | Oscillatory motion per axis | amplitude, frequency, phase per axis |
| Circular | `CircularMotion` | Uniform motion along a circle | center (az, el), radius, angular speed |
| Random maneuvering | `RandomManeuvering` | Bounded-acceleration random walk | max_acc, change_interval, max_vel |
| Scripted trajectory | `ScriptedTrajectory` | Replay from (az, el) position list | position array, fps |

Each model provides full state: position (az, el), velocity (vel_az, vel_el), and acceleration (acc_az, acc_el) at every time step.

### 4.5 Motion Model Mathematical Details

**Constant velocity**: The position at time *t* is computed by direct integration:

```
az(t) = az(0) + az_rate * t
el(t) = el(0) + el_rate * t
```

The velocity is constant (az_rate, el_rate) and acceleration is zero.

**Sinusoidal motion**: Position follows independent sinusoidal oscillations on each axis:

```
az(t) = az_amp * sin(2 * pi * az_freq * t + az_phase)
el(t) = el_amp * sin(2 * pi * el_freq * t + el_phase)
```

Velocity and acceleration are the first and second time derivatives, respectively. The two axes can have different amplitudes, frequencies, and phase offsets, producing Lissajous-like trajectories that stress the controller with continuously varying acceleration.

**Circular motion**: Position traces a circle in angular space:

```
az(t) = center_az + radius * cos(omega * t + start_angle)
el(t) = center_el + radius * sin(omega * t + start_angle)
```

where omega = angular_speed (converted to radians). The velocity components are the time derivatives, and acceleration is centripetal (directed toward the center with magnitude radius * omega^2).

**Random maneuvering**: Acceleration is redrawn from a uniform distribution U(-max_acc, +max_acc) at random intervals drawn from U(0.5 * change_interval, 1.5 * change_interval). Between changes, velocity is integrated forward and clamped to [-max_vel, +max_vel], and position is integrated from velocity. The seeded random number generator ensures reproducibility across runs.

**Scripted trajectory**: Position is set by direct lookup from a pre-recorded array of (az, el) tuples, indexed by frame number. This enables replay of exact trajectories for regression testing.

### 4.6 Multi-Target Support

The `MultiTargetSimulator` extends the single-target simulator to render multiple beacons simultaneously. Each beacon has a unique identity, color (BGR), intensity, sigma, and independent motion model. The `MultiTracker` maintains one `Tracker` per beacon and uses nearest-neighbor data association to assign detections to tracks. Identity switches (where two tracks swap which detection they follow) are counted as a performance metric.

The multi-target data association works as follows:

1. The detector extracts all qualifying blobs from the frame (not just the brightest).
2. Each tracker predicts its next position using its Kalman filter.
3. For each beacon, the detection nearest to its predicted position (within `association_max_distance` = 200 px) is assigned.
4. Detections are assigned greedily: the beacon with the closest prediction claims its detection first.
5. Identity switches are detected by comparing pairwise distance ratios between the current and previous frame assignments. A switch occurs when two beacons' assigned detections swap relative proximity.

---

## 5. Disturbance and Sensor Models

The system models eight distinct disturbance types, organized by the layer at which they are applied.

### 5.1 Camera-Level Disturbances (Angular Perturbation)

These disturbances modify the camera boresight angular position before rendering, simulating physical effects that cause pointing errors.

| # | Disturbance | Class | Model | Key Parameters |
|---|-------------|-------|-------|----------------|
| 1 | Platform vibration | `PlatformVibration` | Sum of sinusoidal modes plus Gaussian jitter | RMS = 0.15 deg; frequencies = [5, 12, 25, 47] Hz |
| 2 | Atmospheric turbulence | `AtmosphericTurbulence` | Ornstein-Uhlenbeck process | RMS = 0.1 deg; correlation time = 1.0 s |

**Platform vibration** models mechanical vibrations from motors, wind, or structural resonances. It superimposes sinusoidal oscillations at multiple frequencies (5, 12, 25, 47 Hz) with equal amplitude, plus broadband Gaussian jitter. The composite offset is added to both pan and tilt axes at each frame.

**Atmospheric turbulence** models slowly varying pointing errors caused by refractive index fluctuations along the optical path. It uses an Ornstein-Uhlenbeck (OU) process, which produces smooth, temporally correlated random walk behavior. The OU process is parameterized by its RMS amplitude and correlation time, with the discrete-time update:

```
offset(t) = exp(-dt/tau) * offset(t-dt) + N(0, sigma_n)
```

where `sigma_n = RMS * sqrt(1 - exp(-2*dt/tau))`.

### 5.2 Frame-Level Disturbances (Image Perturbation)

These disturbances modify the rendered image before it is passed to the detector, simulating sensor and optical effects.

| # | Disturbance | Class | Model | Key Parameters |
|---|-------------|-------|-------|----------------|
| 3 | Sensor noise | `SensorNoise` | Additive Gaussian noise, optional Poisson noise | sigma = 8.0-12.0 |
| 4 | Motion blur | `MotionBlur` | Directional convolution kernel | max_blur = 3.0-5.0 px |
| 5 | Exposure variation | `ExposureVariation` | Sinusoidal brightness modulation | rate = 0.1 Hz; amplitude = 0.3 |
| 6 | Occlusion | `Occlusion` | Scripted black rectangle mask | duration = 1.0 s; frequency = 0.05 Hz |
| 7 | False beacons | `FalseBeacons` | Additional Gaussian blobs at random positions | count = 2; brightness = 80-180 |

**Sensor noise** adds zero-mean Gaussian noise with configurable standard deviation to all pixel values. Optionally, Poisson noise is mixed in at a 30% weight to simulate photon shot noise. The implementation uses OpenCV's `cv2.randn` for performance on large arrays.

**Motion blur** applies a directional convolution kernel whose angle and length are proportional to the relative angular velocity between the target and the camera. This elongates and weakens the beacon image during fast motion.

**Exposure variation** modulates overall frame brightness sinusoidally at 0.1 Hz with 30% amplitude, simulating slow scintillation and detector gain instability.

**Occlusion** places a black rectangular mask (15% of frame area) at a random position during periodic intervals. The timing follows a deterministic sine-wave threshold, producing predictable occlusion periods that force temporary track loss.

**False beacons** inject additional Gaussian blobs at fixed random positions with per-frame brightness flicker. These distractors test the detector's ability to discriminate the real beacon from spurious targets.

### 5.3 Disturbance Application Sequence

Disturbances are applied in a specific order to ensure physical consistency:

1. Camera-level disturbances (vibration, turbulence) modify the camera boresight.
2. The frame is rendered with the disturbed camera state.
3. Frame-level disturbances (motion blur, sensor noise, occlusion, false beacons, exposure variation) are applied to the rendered image.
4. The processed frame is passed to the detector.
5. Ground truth is computed using the disturbed camera state (not the nominal state), ensuring error metrics reflect what the detector actually sees.

### 5.4 Disturbance Model Mathematical Details

**Platform vibration** computes angular offsets as a sum of sinusoidal components:

```
offset_az(t) = sum_i(A_i * sin(2 * pi * f_i * t + phi_i)) + N(0, sigma_jitter)
offset_el(t) = sum_i(A_i * cos(2 * pi * f_i * t + phi_i)) + N(0, sigma_jitter)
```

where A_i = RMS * 0.5 / sqrt(N_modes) is the amplitude of each mode, f_i are the vibration frequencies [5, 12, 25, 47] Hz, phi_i are random phase offsets (seeded), and sigma_jitter = RMS * 0.2 is the broadband Gaussian jitter component. The use of sin for azimuth and cos for elevation decorrelates the two axes.

**Atmospheric turbulence** uses the Ornstein-Uhlenbeck (OU) process, a stationary Gauss-Markov process:

```
theta(t) = exp(-dt/tau) * theta(t-dt) + sigma_n * N(0, 1)
```

where tau is the correlation time (1.0 s), dt is the internal step size (tau/10), and sigma_n = RMS * sqrt(1 - exp(-2*dt/tau)) is chosen so that the process has the target RMS in steady state. The OU process is preferred over a simple random walk because it produces bounded, physically realistic fluctuations with a well-defined correlation structure.

**Sensor noise** uses OpenCV's `cv2.randn` for performance on large arrays:

```
noisy_frame = clip(frame + N(0, sigma), 0, 255)
```

The optional Poisson component models photon shot noise: a Poisson-distributed frame is generated from the integer pixel intensities, then blended at 30% weight with the Gaussian-noised frame.

**Motion blur** constructs a directional convolution kernel:

```
kernel_size = ceil(blur_pixels)
kernel[y, x] = 1/kernel_size  for points along a line at angle_deg
blurred = cv2.filter2D(frame, -1, kernel)
```

The kernel length is proportional to the relative angular velocity (converted to pixels via deg_per_pixel_x * dt), and the angle is the direction of relative motion. This produces realistic streaking of the beacon along its direction of travel.

**Exposure variation** modulates pixel values by a time-varying gain factor:

```
factor(t) = 1.0 + amplitude * sin(2 * pi * rate * t + phase)
modulated_frame = clip(frame * factor, 0, 255)
```

**Occlusion** activates a black rectangular mask (15% of frame area) at a random position during periodic intervals. The on/off timing is determined by `sin(2 * pi * frequency * t) > 0` combined with a fractional phase check, producing predictable 1-second occlusion windows separated by longer clear periods.

**False beacons** render additional Gaussian blobs at fixed random positions with per-frame brightness flicker:

```
brightness_i(t) = base_brightness_i * (1.0 + 0.1 * sin(2 * pi * 0.3 * t + i * 1.5))
```

The flicker frequency (0.3 Hz) and phase offsets ensure that different false beacons flicker independently, making them more challenging to reject than static distractors.

---

## 6. Beacon Detection and Identification Methods

### 6.1 Classical Detection Pipeline

The `BeaconDetector` implements a classical computer-vision pipeline with the following stages:

```mermaid
graph TD
    A[Input Frame BGR] --> B[Grayscale Conversion]
    B --> C{Threshold Method}
    C -->|Fixed| D["Binary Threshold (T=80)"]
    C -->|Otsu| E[Otsu Auto-Threshold with Floor]
    D --> F{HSV Filtering?}
    E --> F
    F -->|Yes| G[HSV Color Mask AND]
    F -->|No| H[Morphological Opening]
    G --> H
    H --> I[Connected Components with Stats]
    I --> J["Filter: Area (2-500 px^2)"]
    J --> K["Filter: Aspect Ratio (< 3.0)"]
    K --> L{Persistence Filter?}
    L -->|Yes| M[Reject Non-Persistent Candidates]
    L -->|No| N[Select Brightest Candidate]
    M --> N
    N --> O["Detection Output (cx, cy, confidence)"]
```

**Grayscale conversion**: The BGR input frame is converted to single-channel grayscale for thresholding.

**Thresholding**: Two methods are supported:
- *Fixed threshold*: Binarizes at a configurable intensity level (default 80). Simple and fast, suitable for controlled scenarios.
- *Otsu's method*: Automatically selects the threshold that maximizes inter-class variance between foreground and background. A configurable floor prevents the Otsu threshold from dropping too low when the background dominates.

**Optional HSV color filtering**: When enabled, a color mask restricts detection to beacons of a specific hue range, which is useful for multi-target scenarios with colored beacons.

**Morphological opening**: An elliptical structuring element (3x3 default) with one iteration removes isolated noise speckles from the binary mask.

**Connected component analysis**: OpenCV's `connectedComponentsWithStats` (8-connectivity) extracts all foreground regions along with their area, bounding box, and centroid.

**Filtering**: Candidates are filtered by:
- Area: regions smaller than `min_area` (2 px^2) or larger than `max_area` (500 px^2) are rejected.
- Aspect ratio: elongated blobs (aspect ratio > 3.0) are rejected, as the beacon is circular.
- Persistence (optional): candidates that have not appeared in a sufficient number of recent frames are rejected, reducing false positives from transient noise.

**Selection**: Among remaining candidates, the one with the highest peak brightness within its bounding box is selected. This heuristic favors the real beacon, which is the brightest object in the scene.

**Confidence scoring**: The confidence of each detection is computed as the ratio of the candidate's mean grayscale intensity to the threshold value, clamped to [0, 1]:

```
confidence = min(1.0, mean_intensity / threshold)
```

A confidence of 1.0 indicates the candidate is at least as bright as the threshold everywhere within its bounding box, while lower values indicate weaker detections that may be noise artifacts.

### 6.2 DetectorBase Interface

All detectors implement the abstract `DetectorBase` interface:

```python
class DetectorBase(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        ...
```

This common interface enables seamless swapping between classical and AI detectors, and ensures that the tracker and controller remain agnostic to the detection method.

### 6.3 Detection Output

The `Detection` dataclass contains:
- `detected` (bool): Whether a beacon was found.
- `cx`, `cy` (float): Centroid position in pixels.
- `area` (float): Blob area in pixel-squared.
- `bbox` (tuple): Bounding box (x, y, w, h).
- `confidence` (float): 0-1 score based on mean brightness relative to threshold.

---

## 7. Tracking Filter, State Machine, and Pan-Tilt Controller

### 7.1 Kalman Filter

The `KalmanFilter2D` implements a constant-velocity Kalman filter with a four-element state vector:

```
x = [px, py, vx, vy]^T
```

where `(px, py)` is the position in pixels and `(vx, vy)` is the velocity in pixels per frame.

**State transition model** (constant velocity):

```
F = [1 0 1 0]    (px' = px + vx)
    [0 1 0 1]    (py' = py + vy)
    [0 0 1 0]    (vx' = vx)
    [0 0 0 1]    (vy' = vy)
```

**Measurement model**: The detector provides only position (px, py):

```
H = [1 0 0 0]
    [0 1 0 0]
```

**Process noise covariance** (Q): Diagonal matrix with configurable position noise (1.0) and velocity noise (0.5). Higher values cause the filter to trust predictions less and measurements more.

**Measurement noise covariance** (R): Diagonal matrix with configurable measurement noise (default 8.0). This parameter controls how much the filter smooths the detector output.

**Predict-update cycle**: The filter predicts every frame regardless of whether a detection is available. When a detection is present, the Kalman gain `K` is computed and the state is updated:

```
K = P * H^T * (H * P * H^T + R)^{-1}
x = x + K * (z - H * x)
P = (I - K * H) * P
```

During detection gaps (e.g., occlusion), the filter extrapolates from the last velocity estimate, maintaining a smooth position estimate through temporary track loss.

### 7.2 Alpha-Beta Filter

An alternative to the Kalman filter, the `AlphaBetaFilter` uses fixed gains (alpha = 0.5 for position, beta = 0.1 for velocity) instead of computing optimal gains from covariance matrices. It is simpler and computationally lighter but does not adapt to varying noise conditions. The filter is selectable via the `filter_type` configuration parameter.

### 7.3 Seven-State Tracking FSM

The tracker implements a seven-state finite state machine that governs the lifecycle of beacon tracking:

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> SEARCHING : start()

    SEARCHING --> CANDIDATE_VERIFICATION : detection

    CANDIDATE_VERIFICATION --> ACQUIRING : verify_threshold consecutive detections
    CANDIDATE_VERIFICATION --> SEARCHING : miss

    ACQUIRING --> TRACKING : acquire_threshold consecutive detections AND error OK
    ACQUIRING --> SEARCHING : miss OR error too large

    TRACKING --> REACQUIRING : lose_threshold consecutive misses

    REACQUIRING --> TRACKING : detection
    REACQUIRING --> SEARCHING : reacquire timeout (150 frames)

    TRACKING --> FAILED : fatal error
    REACQUIRING --> FAILED : fatal error

    FAILED --> IDLE : reset()
```

**State descriptions**:

| State | Behavior |
|-------|----------|
| IDLE | Initial state. No processing until `start()` is called. |
| SEARCHING | Camera sweeps the sky using a raster search pattern. No tracking. |
| CANDIDATE_VERIFICATION | A detection was observed. The filter is initialized and the tracker verifies the detection persists for `verify_threshold` (2) consecutive frames. |
| ACQUIRING | Candidate verified. The tracker accumulates `acquire_threshold` (3) consecutive detections while verifying that the Kalman-detection discrepancy is below `acquire_error_threshold` (100 px). |
| TRACKING | Locked on. Full closed-loop feedback. The controller continuously centers the beacon. |
| REACQUIRING | Track lost. The Kalman filter predicts through the gap. If a detection reappears, tracking resumes. If `reacquire_timeout` (150 frames) elapses without recovery, the tracker reverts to SEARCHING. |
| FAILED | Fatal error state. Requires explicit reset. |

### 7.4 Per-Frame Tracker Logic

Each call to `Tracker.track(frame)` executes:

1. **Predict**: Advance the Kalman filter one step (always, regardless of detection).
2. **Detect**: Run the beacon detector on the input frame.
3. **Update**: If a detection is available, update the Kalman filter with the measured position; reset the consecutive detection counter. Otherwise, increment the miss counter.
4. **State transition**: Apply FSM transition logic based on the current state and detection outcome.
5. **Output**: Return `TrackerOutput` with the current state, Kalman-estimated position, confidence, and diagnostic counters.

### 7.5 PID Pan-Tilt Controller

The `PIDController` implements a classical PID control law for each axis (pan and tilt independently):

```
output = current_output + (Kp * error + Ki * integral + Kd * derivative) * dt
```

**Error computation**: The pixel-space error is converted to angular error:

```
error_pan = (target_x - cx) * deg_per_pixel_x
error_tilt = (cy - target_y) * deg_per_pixel_y
```

The negation of the y-error accounts for the inverted image-y convention (y increases downward while elevation increases upward).

**Rate limiting**: The command rate of change is clamped to `rate_limit_deg_s` (60 deg/s). This prevents instantaneous jumps that would be physically impossible for a real pan-tilt mechanism.

**Deadband**: Errors below `deadband_pixels` (3 px) are zeroed out, preventing oscillation around the target when the error is within acceptable bounds.

**Anti-windup**: The integral term is clamped to `integral_limit` (30 deg) to prevent windup during prolonged saturation.

**Output saturation**: The final command is clamped to the physical range of the pan and tilt axes.

### 7.6 Integrated Controller

The `IntegratedController` ties together the tracker, PID controller, and search pattern generator:

| Tracker State | Controller Behavior |
|---------------|-------------------|
| SEARCHING | Generates a raster sweep pattern (horizontal pan at fixed tilt, stepping upward) |
| ACQUIRING | Feeds the Kalman estimate to the PID controller |
| TRACKING | Full closed-loop PID feedback |
| REACQUIRING | Uses Kalman prediction (no new detections) to hold position |

The search pattern (`SearchPattern`) performs a systematic raster scan: it sweeps horizontally at the current tilt level, bounces at the pan range limits, steps the tilt upward, and repeats. This covers the full camera FOV and ensures the beacon is found regardless of its initial position.

---

## 8. AI Method and Training Data

### 8.1 AI Detector Architecture

The `AIDetector` class wraps an ONNX Runtime inference session for a lightweight object detection model. The design specifies a YOLO-nano or SSD-lite architecture suitable for real-time inference (target: under 30 ms per frame).

**Inference pipeline**:

1. **Preprocessing**: The input frame is resized to the model's input dimensions (e.g., 640x360 for speed), transposed to CHW format, and normalized to [0, 1].
2. **Inference**: The ONNX model is executed via ONNX Runtime, producing a tensor of shape `[batch, num_detections, 6]`, where each detection contains `(x1, y1, x2, y2, confidence, class_id)`.
3. **Postprocessing**: Detections are filtered by a confidence threshold (0.5), the highest-confidence detection is selected, and coordinates are scaled back to the original frame resolution.
4. **Output**: A `Detection` object with centroid, area, bounding box, and confidence.

### 8.2 Fallback Mechanism

If the ONNX model is unavailable (not loaded, inference error, or confidence below threshold), the AI detector falls back to the classical `BeaconDetector`. This ensures the closed loop is never broken by AI failures:

```python
def detect(self, frame):
    if self.session is None:
        return self.fallback.detect(frame)
    try:
        return self._run_inference(frame)
    except Exception:
        return self.fallback.detect(frame)
```

### 8.3 Training Data Generation

Training data is generated synthetically from the simulator:

| Parameter | Value |
|-----------|-------|
| Frame count | 5,000-10,000 |
| Beacon positions | Full FOV coverage |
| Beacon sizes | 3-20 pixels sigma |
| Brightness range | 50-255 peak intensity |
| Disturbance levels | None, low, medium, high |
| Backgrounds | Gradient, gradient + stars |
| Labels | Auto-generated from ground truth (bounding box + class) |
| Train/val/test split | 70% / 15% / 15% (split by scenario seed, not adjacent frames) |

The split-by-seed strategy ensures that temporally adjacent frames do not appear in both training and test sets, preventing data leakage and providing a realistic evaluation of generalization.

### 8.4 Benchmarking Protocol

AI detection improvement is claimed only when demonstrated through repeatable metrics computed on identical test sequences with identical seeds:

- Precision and recall
- Acquisition time
- Lock-retention rate
- False-lock rate
- Identity-switch rate (multi-target)

The classical detector serves as the baseline, and the AI detector is evaluated against it on the same scenarios.

---

## 9. Experimental Design and Evaluation Criteria

### 9.1 Evaluation Scenarios

The evaluation framework defines eight categories, each isolating a specific operational challenge:

| # | Category | Motion Model | Disturbances | Purpose |
|---|----------|-------------|--------------|---------|
| 1 | Nominal | Circular, slow | None | Baseline tracking performance |
| 2 | Motion | Random maneuvering, fast | None | Response to unpredictable trajectories |
| 3 | Vibration | Circular | Platform vibration (0.2 deg RMS) | Mechanical vibration rejection |
| 4 | Turbulence | Circular | Atmospheric turbulence (0.1 deg RMS) | Slow pointing error rejection |
| 5 | Sensor degradation | Circular | Noise + blur + exposure variation | Image quality robustness |
| 6 | Clutter | Circular | 2 false beacons | Distractor discrimination |
| 7 | Occlusion | Circular | Periodic blockage (1.0 s duration) | Track loss recovery |
| 8 | Multi-target | Sinusoidal | Two crossing beacons | Identity maintenance |

### 9.2 Metrics

The following metrics are computed from per-frame CSV logs:

| Metric | Formula | Clean Target | Hard Target |
|--------|---------|-------------|-------------|
| Acquisition time | First TRACKING frame / FPS | < 2.0 s | < 2.0 s |
| Mean pixel error | mean(sqrt((gt_x - cx)^2 + (gt_y - cy)^2)) | < 50 px | < 200 px |
| Steady-state error | mean(error) over last 30 frames | < 15 px | < 100 px |
| Max pixel error | max(sqrt((gt_x - cx)^2 + (gt_y - cy)^2)) | < 300 px | < 500 px |
| Lock retention | count(TRACKING) / total_frames * 100 | > 95% | > 90% |
| Detection rate | count(detected) / total_frames * 100 | 100% | > 95% |
| RMS angular error | sqrt(mean(error_pan^2 + error_tilt^2)) | < 3.0 deg | < 8.0 deg |
| Loss events | count(TRACKING -> REACQUIRING transitions) | 0 | < 3 |
| Reacquisition time | mean duration of REACQUIRING episodes | < 0.5 s | < 3.0 s |
| Mean FPS | mean(fps_counter / elapsed_time) | >= 30 | >= 15 |

### 9.3 Evaluation Matrix

The `EvaluationMatrix` class implements an 8-category weighted scoring system. Each category contributes to the overall score according to its assigned weight:

| Category | Weight | Primary Metrics |
|----------|--------|-----------------|
| Nominal | 0.15 | Acquisition time, mean pixel error |
| Motion | 0.15 | Lock retention, max pixel error |
| Vibration | 0.12 | RMS angular error, lock retention |
| Turbulence | 0.10 | Lock retention, mean pixel error |
| Sensor degradation | 0.12 | Detection rate, mean FPS |
| Clutter | 0.10 | Detection rate, lock retention |
| Occlusion | 0.12 | Reacquisition time, lock retention |
| Multi-target | 0.14 | Lock retention, mean pixel error |

Each metric is scored on a 0-100 scale via linear interpolation between a *baseline* value (score 0) and a *target* value (score 100). The category score is the mean of its constituent metric scores, and the overall score is the weighted sum across categories.

**Metric scoring reference**:

| Metric | Target (100) | Baseline (0) | Direction |
|--------|-------------|-------------|-----------|
| Acquisition time | 0.5 s | 5.0 s | Lower is better |
| Mean pixel error | 25 px | 200 px | Lower is better |
| Max pixel error | 100 px | 500 px | Lower is better |
| Lock retention | 95% | 50% | Higher is better |
| Detection rate | 100% | 60% | Higher is better |
| RMS angular error | 0.5 deg | 5.0 deg | Lower is better |
| Reacquisition time | 0.2 s | 3.0 s | Lower is better |
| Mean FPS | 30.0 | 10.0 | Higher is better |

### 9.4 Evaluation Protocol

1. Set `random_seed` to a fixed value.
2. Run each scenario for the configured duration (default 10 s for short tests, 60 s for full evaluation).
3. Record per-frame CSV log, error-over-time plot, and summary statistics.
4. Repeat with at least 3 different seeds (default set: [42, 123, 456, 789, 1024]).
5. Report mean, standard deviation, and worst-case across seeds.
6. Preserve seed and configuration with every result for reproducibility.

---

## 10. Performance Results and Discussion

### 10.1 Measured Performance

The following results were obtained from 10-second runs with seed 42, using the default parameter configuration:

| Metric | Clean Scenario | Hard Scenario | Target (Clean) | Target (Hard) |
|--------|---------------|---------------|----------------|---------------|
| Acquisition time | 0.07 s | 0.07 s | < 2.0 s | < 2.0 s |
| Mean pixel error | 35 px | 113 px | < 50 px | < 200 px |
| Steady-state error | 10 px | 88 px | < 15 px | < 100 px |
| Max pixel error | 264 px | 182 px | < 300 px | < 500 px |
| Lock retention | 99.3% | 99.3% | > 95% | > 90% |
| Detection rate | 100% | 100% | 100% | > 95% |
| RMS angular error | 2.9 deg | 5.9 deg | < 3.0 deg | < 8.0 deg |
| Loss events | 0 | 0 | 0 | < 3 |
| Mean FPS | 45 | 24 | >= 30 | >= 15 |

### 10.2 Analysis

**Clean scenario**: The system achieves all target metrics comfortably. Acquisition time of 0.07 s (approximately 2 frames) demonstrates rapid initial lock. The steady-state error of 10 px confirms that the PID controller centers the beacon well within the 15-pixel target. Zero loss events and 99.3% lock retention indicate robust continuous tracking. The mean FPS of 45 exceeds the 30 FPS target, confirming real-time capability.

**Hard scenario**: Under sinusoidal motion with vibration (0.2 deg RMS), sensor noise (sigma = 12), and motion blur (3 px max), the system still meets all targets. The mean pixel error of 113 px (target: < 200 px) reflects the larger disturbances but remains well within bounds. The steady-state error of 88 px (target: < 100 px) is tighter, suggesting that the PID gains could be further tuned for the hard scenario. The RMS angular error of 5.9 deg (target: < 8 deg) demonstrates acceptable pointing accuracy. The mean FPS of 24 is above the 15 FPS minimum but below the 30 FPS target, indicating that the noise computation (sigma = 12) is the dominant computational cost.

**Acquisition performance**: Both scenarios achieve acquisition in 0.07 s, indicating that the classical detector reliably identifies the beacon on the second or third frame, and the FSM transitions through CANDIDATE_VERIFICATION and ACQUIRING rapidly.

**Detection robustness**: The 100% detection rate in both scenarios confirms that the fixed threshold of 80 is well-suited to the beacon intensity (peak 255) against the dark background (10-40). The detector does not miss any frames, which is expected for synthetic data with a high signal-to-noise ratio.

**Controller behavior**: The zero loss events in both scenarios indicate that the 5-frame lose threshold is sufficient to absorb transient detection gaps caused by noise and vibration. The Kalman filter's prediction through these gaps, combined with the PID controller's rate limiting, produces smooth camera motion that maintains the beacon within the FOV.

### 10.3 Timing Budget

The per-frame timing breakdown from PRD section 26:

| Component | Clean | Hard | Budget |
|-----------|-------|------|--------|
| Target motion update | < 0.1 ms | < 0.1 ms | 1 ms |
| Camera copy + vibration | < 0.1 ms | 0.5 ms | 2 ms |
| Frame rendering | 14 ms | 14 ms | 20 ms |
| Motion blur | 0 ms | 3 ms | 5 ms |
| Sensor noise | 0 ms | 15 ms | 20 ms |
| Detection | 2 ms | 3 ms | 5 ms |
| Kalman + FSM | < 0.1 ms | < 0.1 ms | 1 ms |
| PID controller | < 0.1 ms | < 0.1 ms | 1 ms |
| CSV logging | < 0.1 ms | < 0.1 ms | 1 ms |
| HUD rendering | 1 ms | 1 ms | 5 ms |
| **Total** | **~18 ms** | **~38 ms** | **< 60 ms** |

The clean-scenario total of ~18 ms supports 45+ FPS, while the hard-scenario total of ~38 ms supports approximately 24 FPS. Both are within the 60 ms budget.

---

## 11. Limitations, Risks, and Future Improvements

### 11.1 Current Limitations

| Limitation | Impact | Mitigation / Future Work |
|-----------|--------|-------------------------|
| 2-D angular environment only | No range or depth information; occlusion is modeled as image masks rather than 3-D geometry | 3-D renderer planned for v2.0 |
| Ideal camera model (no lens distortion) | Real lenses introduce radial/tangential distortion that shifts beacon positions | Add Brown-Conrady distortion model |
| Fixed PID gains | No adaptation to varying disturbance levels | Online gain scheduling or LQR/MPC |
| Fixed brightness threshold | Detector sensitivity is not adaptive to changing illumination | Otsu method available but not default |
| Single-target tracking in control loop | Multi-target scenarios track all beacons but only control on the first | Per-target control or priority selection |
| Gaussian beacon model | Real FSOC beacons may have asymmetric, saturated, or structured intensity profiles | Textured sprite rendering for v2.0 |
| No atmospheric propagation model | Turbulence is modeled as angular jitter, not as scintillation, beam wander, or beam spread | Physics-based propagation for v2.0 |
| Ground-truth isolation is enforced by convention | A future developer could accidentally pass GroundTruth to the detector | Consider making GroundTruth a private return type |

### 11.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Classical detector fails under heavy noise | Low | High | HSV color filtering and persistence filter reduce false positives; AI fallback |
| Kalman filter diverges with poor initial conditions | Low | Medium | Filter reinitialization on acquisition; covariance reset |
| PID oscillation at high gains | Medium | Medium | Rate limiting and deadband prevent oscillation; Ki = 0 by default |
| Frame rate drops below real-time on slow hardware | Medium | Low | Headless mode disables display; configurable resolution |
| Reproducibility broken by library version changes | Low | High | All random operations use NumPy's Generator with explicit seeds |

### 11.3 Future Improvements

**v1.1 enhancements**:
- PySide6 GUI with configuration panels, real-time plots, and report export
- YAML-based scenario configuration for reproducible experiment definitions
- 7-state FSM with candidate verification and failure states (delivered)
- Alpha-beta filter as an alternative to Kalman (delivered)
- Eight-category evaluation matrix with weighted scoring (delivered)
- AI detector with ONNX export and benchmarking against classical baseline
- Multi-target scenarios with identity-switch tracking
- Atmospheric turbulence, exposure variation, occlusion, and false-beacon disturbances (delivered)
- Structured JSON and HTML report generation (delivered)

**v2.0 roadmap**:
- 3-D rendering with perspective projection, depth, and realistic occlusion
- Real camera input (USB/IP camera feed)
- Hardware-in-the-loop mode with physical pan-tilt gimbal
- Adaptive thresholding and multi-spectral detection
- Machine-learning-based controller (reinforcement learning or model-predictive control)
- Atmospheric propagation modeling (scintillation, beam wander, beam spread)
- Link budget integration and communication-layer simulation

---

## 12. Conclusion

This report presented the design, implementation, and evaluation of a software-in-the-loop FSOC coarse-alignment simulator. The system generates synthetic camera frames containing a moving optical beacon, processes them through a classical detection pipeline without access to ground truth, estimates the beacon position with a Kalman filter, and commands a virtual pan-tilt camera to center the beacon using a PID controller.

The architecture enforces strict ground-truth isolation, ensuring that the detector and controller operate under the same informational constraints as a real system. The modular design, with four core modules (sim, detect, control, config) and strict information-flow rules, enables independent development and testing of each component.

Eight disturbance models -- platform vibration, atmospheric turbulence, sensor noise, motion blur, exposure variation, occlusion, and false beacons -- provide comprehensive coverage of real-world challenges facing FSOC terminals. The 7-state tracking FSM with candidate verification, acquisition validation, and reacquisition timeout provides robust lifecycle management.

Performance results demonstrate that the system meets all target metrics for both clean and hard scenarios. The clean scenario achieves a mean pixel error of 35 px, steady-state error of 10 px, and 99.3% lock retention with zero loss events. The hard scenario, under sinusoidal motion with vibration, noise, and blur, maintains a mean pixel error of 113 px, steady-state error of 88 px, and identical 99.3% lock retention. Both scenarios achieve sub-0.1-second acquisition time and real-time or near-real-time frame rates.

The 8-category evaluation framework with weighted scoring provides a systematic method for comparing detection methods, controller configurations, and disturbance handling. The classical detector, ONNX-based AI detector with classical fallback, and the full evaluation pipeline are implemented and functional.

The simulator provides a reproducible, parametric platform for developing and benchmarking FSOC coarse-alignment algorithms without physical hardware, enabling rapid iteration and fair comparison of approaches.

### 12.2 Design Lessons

Several architectural decisions proved particularly valuable during implementation and testing:

1. **Centralized configuration in a single dataclass**: Placing all tunable parameters in the `Config` dataclass (115 lines in `src/config.py`) eliminated an entire class of bugs where modules used inconsistent parameter values. Every tunable value lives in one place, making parameter sweeps, scenario comparison, and debugging straightforward. This design also simplifies YAML configuration file loading in v1.1, since the dataclass can be constructed directly from a dictionary.

2. **Flat module structure with clear import hierarchy**: The flat layout of five core source files allowed rapid development without the overhead of package management. The import dependencies form a clean directed acyclic graph: `sim.py` has no internal imports; `detect.py` imports from `sim.py`; `control.py` imports from both `sim.py` and `detect.py`; `main.py` imports from all three plus `config.py`. This layering means each module can be tested independently.

3. **Embedded module self-tests**: Placing test functions in the `if __name__ == "__main__"` block of each module provided immediate feedback during development without requiring a separate test framework. The closed-loop test in `control.py` (lines 489-561) is particularly valuable because it exercises the entire sim-detect-control-camera pipeline end-to-end and verifies that the beacon is brought to within 100 pixels of center within 10 seconds.

4. **Disturbance isolation via the apply() pattern**: Each disturbance class implements an `apply()` method on a common interface, making it trivial to add, remove, or debug individual effects. The ability to run the system with zero, one, or many disturbances enabled was essential for the progressive tuning protocol: tune PID gains in the clean scenario first, then add disturbances one at a time and verify that performance degrades gracefully.

5. **Ground-truth separation by architectural convention**: The strict separation between ground truth (available only to the logger and HUD) and frame data (available to the detector and controller) ensured that performance numbers are trustworthy. The dev-mode HUD overlay is explicitly disabled during evaluation runs to prevent accidental information leakage.

### 12.3 Transferability to Real FSOC Systems

Although this simulator operates in a simplified two-dimensional angular domain, the algorithms and architecture are directly transferable to real FSOC terminals. The detection pipeline (brightness thresholding, morphological filtering, connected-component analysis) is a standard approach used in many optical tracking systems. The Kalman filter with constant-velocity model is a well-understood estimator that has been validated in numerous real-world tracking applications. The PID controller with rate limiting, deadband, and anti-windup is the industry-standard approach for pan-tilt servo control.

The key differences between the simulator and a real deployment would be the input source (a physical camera rather than a synthetic renderer) and the output target (a real pan-tilt gimbal rather than simulated camera pose updates). The `DetectorBase` abstract interface is designed precisely for this transition: an AI or classical detector trained on synthetic data can be deployed on real camera frames through the same interface, and the controller output (pan and tilt angle commands) can drive a real gimbal through a serial or network interface.

The synthetic-to-real gap remains a challenge for AI-based detectors, since models trained exclusively on synthetic data may not generalize to real sensor characteristics (lens distortion, rolling shutter, color cast, etc.). The classical detector, by contrast, operates directly on pixel intensities and is inherently more portable across sensor types. The evaluation framework, with its systematic disturbance injection and multi-seed testing protocol, provides the infrastructure needed to quantify this synthetic-to-real gap when real hardware becomes available.

---

*End of Technical Report*
