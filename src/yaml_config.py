"""
yaml_config.py — YAML scenario configuration loader for the FSOC simulator.

Loads a structured YAML scenario file and converts it into the flat
Config dataclass used by the rest of the simulator.

Usage:
    from yaml_config import load_scenario, scenario_to_config

    scenario = load_scenario("scenarios/clean.yaml")
    config = scenario_to_config(scenario)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config import Config


# ── Required top-level sections ──────────────────────────────────────────
# Only simulation, camera, and motion are truly required.
# Other sections (beacon, detector, kalman, controller, tracker) use Config defaults.
_REQUIRED_SECTIONS = {"simulation", "camera"}

# ── Required keys inside each section ────────────────────────────────────
_REQUIRED_KEYS: dict[str, list[str]] = {
    "simulation": ["fps", "duration_s"],
    "camera": ["width", "height"],
    "beacon": [],
    "motion": ["type"],
    "disturbances": [],
    "detector": [],
    "kalman": [],
    "controller": [],
    "tracker": [],
}


# ── Public API ───────────────────────────────────────────────────────────

def load_scenario(yaml_path: str) -> dict:
    """Load and validate a YAML scenario file.

    Parameters
    ----------
    yaml_path : str
        Path to a YAML scenario file.

    Returns
    -------
    dict
        Parsed and validated scenario dictionary.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    ValueError
        If the YAML is malformed or fails schema validation.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {yaml_path}")

    # ── Parse YAML ──
    try:
        with open(path, "r", encoding="utf-8") as fh:
            scenario = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"[yaml_config] YAML parse error in {yaml_path}: {exc}")
        raise ValueError(f"Invalid YAML in {yaml_path}: {exc}") from exc

    if not isinstance(scenario, dict):
        print(f"[yaml_config] {yaml_path} must be a mapping at the top level, got {type(scenario).__name__}")
        raise ValueError(f"Scenario file must be a YAML mapping: {yaml_path}")

    # ── Version check (optional, informational) ──
    version = scenario.get("version")
    if version:
        print(f"[yaml_config] Scenario version: {version}")

    # ── Validate ──
    errors = _validate(scenario)
    if errors:
        msg = "\n  - ".join(errors)
        print(f"[yaml_config] Validation errors in {yaml_path}:\n  - {msg}")
        raise ValueError(f"Scenario validation failed:\n  - {msg}")

    return scenario


def scenario_to_config(scenario: dict) -> Config:
    """Convert a parsed scenario dict into a :class:`Config` dataclass.

    The YAML uses nested sections; this flattens them into the Config
    fields expected by the simulator.

    Parameters
    ----------
    scenario : dict
        A validated scenario dictionary (from :func:`load_scenario`).

    Returns
    -------
    Config
        A fully-populated Config instance.
    """
    sim = scenario.get("simulation", {})
    cam = scenario.get("camera", {})
    bcn = scenario.get("beacon", {})
    mot = scenario.get("motion", {})
    dst = scenario.get("disturbances", {})
    det = scenario.get("detector", {})
    klm = scenario.get("kalman", {})
    ctr = scenario.get("controller", {})
    trk = scenario.get("tracker", {})

    motion_type = mot.get("type", "circular")

    # Build kwargs for Config
    kwargs: dict[str, Any] = {}

    # ── Simulation ──
    kwargs["fps"] = _int(sim.get("fps", 30))
    kwargs["duration_s"] = _int(sim.get("duration_s", 60))
    if "random_seed" in sim:
        kwargs["random_seed"] = _int(sim["random_seed"])

    # ── Camera ──
    kwargs["width"] = _int(cam.get("width", 1280))
    kwargs["height"] = _int(cam.get("height", 720))
    if "h_fov_deg" in cam:
        kwargs["h_fov_deg"] = _float(cam["h_fov_deg"])
    if "v_fov_deg" in cam:
        kwargs["v_fov_deg"] = _float(cam["v_fov_deg"])
    if "pan_range" in cam:
        kwargs["pan_range"] = tuple(cam["pan_range"])
    if "tilt_range" in cam:
        kwargs["tilt_range"] = tuple(cam["tilt_range"])
    if "rate_limit_deg_s" in cam:
        kwargs["rate_limit_deg_s"] = _float(cam["rate_limit_deg_s"])

    # ── Beacon ──
    if "sigma_px" in bcn:
        kwargs["beacon_sigma_px"] = _float(bcn["sigma_px"])
    if "peak" in bcn:
        kwargs["beacon_peak"] = _float(bcn["peak"])

    # ── Detector ──
    if "threshold" in det:
        kwargs["threshold"] = _int(det["threshold"])
    if "min_area" in det:
        kwargs["min_area"] = _int(det["min_area"])
    if "max_area" in det:
        kwargs["max_area"] = _int(det["max_area"])

    # ── Kalman ──
    if "measurement_noise" in klm:
        kwargs["measurement_noise"] = _float(klm["measurement_noise"])

    # ── Controller ──
    if "kp" in ctr:
        kwargs["kp"] = _float(ctr["kp"])
    if "ki" in ctr:
        kwargs["ki"] = _float(ctr["ki"])
    if "kd" in ctr:
        kwargs["kd"] = _float(ctr["kd"])
    if "deadband_pixels" in ctr:
        kwargs["deadband_pixels"] = _float(ctr["deadband_pixels"])

    # ── Tracker ──
    if "acquire_threshold" in trk:
        kwargs["acquire_threshold"] = _int(trk["acquire_threshold"])
    if "lose_threshold" in trk:
        kwargs["lose_threshold"] = _int(trk["lose_threshold"])
    if "reacquire_timeout" in trk:
        kwargs["reacquire_timeout"] = _int(trk["reacquire_timeout"])

    # ── Filter selection ──
    if "filter_type" in klm:
        kwargs["filter_type"] = str(klm["filter_type"])
    if "alpha" in klm:
        kwargs["alpha"] = _float(klm["alpha"])
    if "beta" in klm:
        kwargs["beta"] = _float(klm["beta"])

    # ── Detector selection ──
    if "detector_type" in det:
        kwargs["detector_type"] = str(det["detector_type"])
    if "ai_model_path" in det:
        kwargs["ai_model_path"] = str(det["ai_model_path"])

    # ── Scenario-specific motion + disturbance fields ──

    kwargs["clean_motion"] = "circular" if motion_type == "circular" else motion_type
    kwargs["hard_motion"] = "sinusoidal" if motion_type == "sinusoidal" else motion_type

    # Motion parameters
    if motion_type == "circular":
        kwargs["clean_center_az"] = _float(mot.get("center_az_deg", 5.0))
        kwargs["clean_center_el"] = _float(mot.get("center_el_deg", 3.0))
        kwargs["clean_radius"] = _float(mot.get("radius_deg", 8.0))
        kwargs["clean_speed"] = _float(mot.get("speed_deg_s", 4.0))
    elif motion_type == "sinusoidal":
        kwargs["hard_az_amp"] = _float(mot.get("az_amp_deg", 10.0))
        kwargs["hard_az_freq"] = _float(mot.get("az_freq_hz", 0.2))
        kwargs["hard_el_amp"] = _float(mot.get("el_amp_deg", 6.0))
        kwargs["hard_el_freq"] = _float(mot.get("el_freq_hz", 0.3))
    elif motion_type == "random":
        kwargs["clean_motion"] = "random"
        kwargs["clean_max_acc"] = _float(mot.get("max_acc_deg_s2", 10.0))
        kwargs["clean_change_interval"] = _float(mot.get("change_interval_s", 2.0))
        kwargs["clean_max_vel"] = _float(mot.get("max_vel_deg_s", 20.0))
    elif motion_type == "constant":
        kwargs["clean_motion"] = "constant"
        kwargs["clean_az_rate"] = _float(mot.get("az_rate_deg_s", 5.0))
        kwargs["clean_el_rate"] = _float(mot.get("el_rate_deg_s", 2.0))
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    # Disturbances
    if motion_type == "circular":
        kwargs["clean_vibration_rms"] = _float(dst.get("vibration_rms_deg", 0.0))
        kwargs["clean_noise_sigma"] = _float(dst.get("noise_sigma", 0.0))
        kwargs["clean_blur_pixels"] = _float(dst.get("blur_pixels", 0.0))
    else:
        kwargs["hard_vibration_rms"] = _float(dst.get("vibration_rms_deg", 0.0))
        kwargs["hard_noise_sigma"] = _float(dst.get("noise_sigma", 0.0))
        kwargs["hard_blur_pixels"] = _float(dst.get("blur_pixels", 0.0))

    # ── v1.1 disturbances (scenario-independent) ──
    if "turbulence_rms_deg" in dst:
        kwargs["turbulence_rms_deg"] = _float(dst["turbulence_rms_deg"])
    if "turbulence_correlation_s" in dst:
        kwargs["turbulence_correlation_s"] = _float(dst["turbulence_correlation_s"])
    if "exposure_rate_hz" in dst:
        kwargs["exposure_rate_hz"] = _float(dst["exposure_rate_hz"])
    if "exposure_amplitude" in dst:
        kwargs["exposure_amplitude"] = _float(dst["exposure_amplitude"])
    if "occlusion_duration_s" in dst:
        kwargs["occlusion_duration_s"] = _float(dst["occlusion_duration_s"])
    if "occlusion_frequency_hz" in dst:
        kwargs["occlusion_frequency_hz"] = _float(dst["occlusion_frequency_hz"])
    if "false_beacon_count" in dst:
        kwargs["false_beacon_count"] = _int(dst["false_beacon_count"])
    if "false_beacon_brightness_range" in dst:
        kwargs["false_beacon_brightness_range"] = tuple(dst["false_beacon_brightness_range"])

    return Config(**kwargs)


def config_to_scenario(config: Config, scenario_name: str = "custom") -> dict:
    """Convert a flat Config dataclass into a structured scenario dict conforming to YAML schema."""
    motion_type = config.clean_motion if scenario_name != "hard" else config.hard_motion
    if motion_type not in ("circular", "sinusoidal", "random", "constant"):
        motion_type = "circular"

    mot_dict: dict[str, Any] = {"type": motion_type}
    if motion_type == "circular":
        mot_dict["center_az_deg"] = float(config.clean_center_az)
        mot_dict["center_el_deg"] = float(config.clean_center_el)
        mot_dict["radius_deg"] = float(config.clean_radius)
        mot_dict["speed_deg_s"] = float(config.clean_speed)
    elif motion_type == "sinusoidal":
        mot_dict["az_amp_deg"] = float(config.hard_az_amp)
        mot_dict["az_freq_hz"] = float(config.hard_az_freq)
        mot_dict["el_amp_deg"] = float(config.hard_el_amp)
        mot_dict["el_freq_hz"] = float(config.hard_el_freq)
    elif motion_type == "random":
        mot_dict["max_acc_deg_s2"] = float(config.clean_max_acc)
        mot_dict["change_interval_s"] = float(config.clean_change_interval)
        mot_dict["max_vel_deg_s"] = float(config.clean_max_vel)
    elif motion_type == "constant":
        mot_dict["az_rate_deg_s"] = float(config.clean_az_rate)
        mot_dict["el_rate_deg_s"] = float(config.clean_el_rate)

    vibration_rms = config.hard_vibration_rms if scenario_name == "hard" else config.clean_vibration_rms
    noise_sigma = config.hard_noise_sigma if scenario_name == "hard" else config.clean_noise_sigma
    blur_pixels = config.hard_blur_pixels if scenario_name == "hard" else config.clean_blur_pixels

    dst_dict = {
        "vibration_rms_deg": float(vibration_rms),
        "noise_sigma": float(noise_sigma),
        "blur_pixels": float(blur_pixels),
        "turbulence_rms_deg": float(config.turbulence_rms_deg),
        "turbulence_correlation_s": float(config.turbulence_correlation_s),
        "exposure_rate_hz": float(config.exposure_rate_hz),
        "exposure_amplitude": float(config.exposure_amplitude),
        "occlusion_duration_s": float(config.occlusion_duration_s),
        "occlusion_frequency_hz": float(config.occlusion_frequency_hz),
        "false_beacon_count": int(config.false_beacon_count),
    }

    scenario = {
        "name": scenario_name,
        "version": "1.1",
        "description": f"Scenario {scenario_name} exported from FSOC Simulator",
        "simulation": {
            "fps": int(config.fps),
            "duration_s": int(config.duration_s),
            "random_seed": int(config.random_seed),
        },
        "camera": {
            "width": int(config.width),
            "height": int(config.height),
            "h_fov_deg": float(config.h_fov_deg),
            "v_fov_deg": float(config.v_fov_deg),
            "pan_range": list(config.pan_range),
            "tilt_range": list(config.tilt_range),
            "rate_limit_deg_s": float(config.rate_limit_deg_s),
        },
        "beacon": {
            "sigma_px": float(config.beacon_sigma_px),
            "peak": float(config.beacon_peak),
        },
        "motion": mot_dict,
        "disturbances": dst_dict,
        "detector": {
            "threshold": int(config.threshold),
            "min_area": int(config.min_area),
            "max_area": int(config.max_area),
            "detector_type": str(config.detector_type),
        },
        "kalman": {
            "measurement_noise": float(config.measurement_noise),
            "filter_type": str(config.filter_type),
            "alpha": float(config.alpha),
            "beta": float(config.beta),
        },
        "controller": {
            "kp": float(config.kp),
            "ki": float(config.ki),
            "kd": float(config.kd),
            "deadband_pixels": float(config.deadband_pixels),
        },
        "tracker": {
            "acquire_threshold": int(config.acquire_threshold),
            "verify_threshold": int(config.verify_threshold),
            "acquire_error_threshold": float(config.acquire_error_threshold),
            "lose_threshold": int(config.lose_threshold),
            "reacquire_timeout": int(config.reacquire_timeout),
        },
    }
    return scenario


def save_scenario(config: Config, yaml_path: str, scenario_name: str = "custom") -> None:
    """Save a Config instance to a YAML scenario file."""
    scenario_dict = config_to_scenario(config, scenario_name)
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(scenario_dict, fh, default_flow_style=False, sort_keys=False)



# ── Internal helpers ─────────────────────────────────────────────────────

def _validate(scenario: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors: list[str] = []

    # Check required top-level sections
    for section in _REQUIRED_SECTIONS:
        if section not in scenario:
            errors.append(f"Missing required section: '{section}'")

    # Check required keys within each present section
    for section, keys in _REQUIRED_KEYS.items():
        if section not in scenario:
            continue  # already flagged above
        sec = scenario[section]
        if not isinstance(sec, dict):
            errors.append(f"Section '{section}' must be a mapping, got {type(sec).__name__}")
            continue
        for key in keys:
            if key not in sec:
                errors.append(f"Section '{section}' is missing required key: '{key}'")

    # Validate types of critical values where present
    _check_type(scenario, "simulation", "fps", int, errors)
    _check_type(scenario, "simulation", "duration_s", int, errors)
    _check_type(scenario, "camera", "width", int, errors)
    _check_type(scenario, "camera", "height", int, errors)
    _check_type(scenario, "detector", "threshold", int, errors)
    _check_type(scenario, "controller", "kp", (int, float), errors)
    _check_type(scenario, "controller", "ki", (int, float), errors)
    _check_type(scenario, "controller", "kd", (int, float), errors)

    # Validate motion section
    mot = scenario.get("motion", {})
    if isinstance(mot, dict):
        mt = mot.get("type", "circular")
        if mt not in ("circular", "sinusoidal", "random", "constant"):
            errors.append(f"Unknown motion type: '{mt}' (must be 'circular', 'sinusoidal', 'random', or 'constant')")

    return errors


def _check_type(scenario: dict, section: str, key: str,
                expected: type | tuple[type, ...], errors: list[str]) -> None:
    """Append an error if a value has the wrong type."""
    sec = scenario.get(section, {})
    if not isinstance(sec, dict):
        return
    val = sec.get(key)
    if val is not None and not isinstance(val, expected):
        errors.append(
            f"Section '{section}', key '{key}': expected {expected}, "
            f"got {type(val).__name__}"
        )


def _int(val: Any) -> int:
    """Coerce a value to int, letting non-numeric types raise naturally."""
    return int(val)


def _float(val: Any) -> float:
    """Coerce a value to float."""
    return float(val)
