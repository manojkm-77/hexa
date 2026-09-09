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
_REQUIRED_SECTIONS = {"simulation", "camera", "beacon", "detector",
                      "kalman", "controller", "tracker"}

# ── Required keys inside each section ────────────────────────────────────
_REQUIRED_KEYS: dict[str, list[str]] = {
    "simulation": ["fps", "duration_s"],
    "camera": ["width", "height", "h_fov_deg", "v_fov_deg"],
    "beacon": ["sigma_px", "peak"],
    "motion": ["type"],
    "disturbances": [],
    "detector": ["threshold", "min_area", "max_area"],
    "kalman": ["measurement_noise"],
    "controller": ["kp", "ki", "kd", "deadband_pixels"],
    "tracker": ["acquire_threshold", "lose_threshold", "reacquire_timeout"],
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

    # ── Scenario-specific motion + disturbance fields ──
    prefix = "clean" if motion_type == "circular" else "hard"

    kwargs["clean_motion"] = "circular"
    kwargs["hard_motion"] = "sinusoidal"

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

    return Config(**kwargs)


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
        if mt not in ("circular", "sinusoidal"):
            errors.append(f"Unknown motion type: '{mt}' (must be 'circular' or 'sinusoidal')")

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
