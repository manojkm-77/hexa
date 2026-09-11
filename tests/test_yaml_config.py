"""Tests for the YAML scenario configuration loader."""

from pathlib import Path
import textwrap

import pytest
import yaml

from yaml_config import load_scenario, scenario_to_config
from config import Config

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_temp_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temporary file and return the path."""
    p = tmp_path / "test_scenario.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Test: valid YAML produces a valid Config ────────────────────────────

class TestLoadValidScenario:
    """Load real scenario files and convert to Config."""

    def test_clean_yaml_loads_and_converts(self):
        scenario = load_scenario(str(SCENARIOS_DIR / "clean.yaml"))
        config = scenario_to_config(scenario)

        assert isinstance(config, Config)
        assert config.fps == 30
        assert config.duration_s == 60
        assert config.width == 1280
        assert config.height == 720
        assert config.h_fov_deg == 60.0
        assert config.v_fov_deg == 40.0
        assert config.pan_range == (-180.0, 180.0)
        assert config.tilt_range == (-30.0, 90.0)
        assert config.beacon_sigma_px == 4.0
        assert config.beacon_peak == 255.0
        assert config.threshold == 80
        assert config.min_area == 2
        assert config.max_area == 500
        assert config.measurement_noise == 8.0
        assert config.kp == 1.0
        assert config.ki == 0.0
        assert config.kd == 0.15
        assert config.deadband_pixels == 3.0
        assert config.acquire_threshold == 3
        assert config.lose_threshold == 5
        assert config.reacquire_timeout == 150

    def test_hard_yaml_loads_and_converts(self):
        scenario = load_scenario(str(SCENARIOS_DIR / "hard.yaml"))
        config = scenario_to_config(scenario)

        assert isinstance(config, Config)
        assert config.fps == 30
        assert config.duration_s == 60
        assert config.width == 1280
        assert config.height == 720

        # Sinusoidal motion fields
        assert config.hard_az_amp == 10.0
        assert config.hard_az_freq == 0.2
        assert config.hard_el_amp == 6.0
        assert config.hard_el_freq == 0.3

        # Disturbances
        assert config.hard_vibration_rms == 0.2
        assert config.hard_noise_sigma == 12.0
        assert config.hard_blur_pixels == 3.0

    def test_clean_has_no_disturbances(self):
        scenario = load_scenario(str(SCENARIOS_DIR / "clean.yaml"))
        config = scenario_to_config(scenario)

        assert config.clean_vibration_rms == 0.0
        assert config.clean_noise_sigma == 0.0
        assert config.clean_blur_pixels == 0.0

    def test_hard_has_disturbances(self):
        scenario = load_scenario(str(SCENARIOS_DIR / "hard.yaml"))
        config = scenario_to_config(scenario)

        assert config.hard_vibration_rms == 0.2
        assert config.hard_noise_sigma == 12.0
        assert config.hard_blur_pixels == 3.0


# ── Test: invalid YAML raises ValueError ────────────────────────────────

class TestInvalidYAML:
    """Malformed YAML must raise ValueError."""

    def test_syntax_error(self, tmp_path):
        path = _write_temp_yaml(tmp_path, """
            simulation:
              fps: 30
              duration_s: 60
            camera:
              width: 1280
              broken yaml [[[
        """)
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_scenario(str(path))

    def test_not_a_mapping(self, tmp_path):
        path = _write_temp_yaml(tmp_path, """
            - item1
            - item2
        """)
        with pytest.raises(ValueError, match="mapping"):
            load_scenario(str(path))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_scenario("nonexistent_path.yaml")


# ── Test: missing required fields are caught ─────────────────────────────

class TestMissingRequiredFields:
    """Missing required sections or keys must be caught."""

    def test_missing_simulation_section(self, tmp_path):
        path = _write_temp_yaml(tmp_path, """
            camera:
              width: 1280
              height: 720
              h_fov_deg: 60.0
              v_fov_deg: 40.0
            beacon:
              sigma_px: 4.0
              peak: 255.0
            motion:
              type: circular
            disturbances: {}
            detector:
              threshold: 80
              min_area: 2
              max_area: 500
            kalman:
              measurement_noise: 8.0
            controller:
              kp: 1.0
              ki: 0.0
              kd: 0.15
              deadband_pixels: 3.0
            tracker:
              acquire_threshold: 3
              lose_threshold: 5
              reacquire_timeout: 150
        """)
        with pytest.raises(ValueError, match="simulation"):
            load_scenario(str(path))

    def test_missing_required_key_in_section(self, tmp_path):
        path = _write_temp_yaml(tmp_path, """
            simulation:
              fps: 30
            camera:
              width: 1280
              height: 720
              h_fov_deg: 60.0
              v_fov_deg: 40.0
            beacon:
              sigma_px: 4.0
              peak: 255.0
            motion:
              type: circular
            disturbances: {}
            detector:
              threshold: 80
              min_area: 2
              max_area: 500
            kalman:
              measurement_noise: 8.0
            controller:
              kp: 1.0
              ki: 0.0
              kd: 0.15
              deadband_pixels: 3.0
            tracker:
              acquire_threshold: 3
              lose_threshold: 5
              reacquire_timeout: 150
        """)
        with pytest.raises(ValueError, match="duration_s"):
            load_scenario(str(path))

    def test_missing_controller_section_uses_defaults(self, tmp_path):
        """Controller section is optional; Config defaults apply."""
        path = _write_temp_yaml(tmp_path, """
            simulation:
              fps: 30
              duration_s: 60
            camera:
              width: 1280
              height: 720
              h_fov_deg: 60.0
              v_fov_deg: 40.0
            beacon:
              sigma_px: 4.0
              peak: 255.0
            motion:
              type: circular
            disturbances: {}
            detector:
              threshold: 80
              min_area: 2
              max_area: 500
            kalman:
              measurement_noise: 8.0
            tracker:
              acquire_threshold: 3
              lose_threshold: 5
              reacquire_timeout: 150
        """)
        scenario = load_scenario(str(path))
        config = scenario_to_config(scenario)
        assert config.kp == 1.0  # Config default

    def test_invalid_motion_type(self, tmp_path):
        path = _write_temp_yaml(tmp_path, """
            simulation:
              fps: 30
              duration_s: 60
            camera:
              width: 1280
              height: 720
              h_fov_deg: 60.0
              v_fov_deg: 40.0
            beacon:
              sigma_px: 4.0
              peak: 255.0
            motion:
              type: linear
            disturbances: {}
            detector:
              threshold: 80
              min_area: 2
              max_area: 500
            kalman:
              measurement_noise: 8.0
            controller:
              kp: 1.0
              ki: 0.0
              kd: 0.15
              deadband_pixels: 3.0
            tracker:
              acquire_threshold: 3
              lose_threshold: 5
              reacquire_timeout: 150
        """)
        with pytest.raises(ValueError, match="motion type"):
            load_scenario(str(path))


# ── Test: clean.yaml and hard.yaml both load successfully ────────────────

class TestBothScenariosLoad:
    """Verify both shipped scenario files load and convert cleanly."""

    def test_clean_loads(self):
        path = str(SCENARIOS_DIR / "clean.yaml")
        scenario = load_scenario(path)
        config = scenario_to_config(scenario)
        assert config.fps > 0
        assert config.duration_s > 0

    def test_hard_loads(self):
        path = str(SCENARIOS_DIR / "hard.yaml")
        scenario = load_scenario(path)
        config = scenario_to_config(scenario)
        assert config.fps > 0
        assert config.duration_s > 0

    def test_both_configs_are_distinct(self):
        clean = scenario_to_config(load_scenario(str(SCENARIOS_DIR / "clean.yaml")))
        hard = scenario_to_config(load_scenario(str(SCENARIOS_DIR / "hard.yaml")))

        # Clean has no disturbances; hard does
        assert clean.clean_vibration_rms == 0.0
        assert hard.hard_vibration_rms == 0.2
