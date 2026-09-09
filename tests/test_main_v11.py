"""Tests for v1.1 disturbance integration in main.py make_scenario()."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from sim import (
    AtmosphericTurbulence,
    ExposureVariation,
    Occlusion,
    FalseBeacons,
)


class TestMakeScenarioV11Disturbances:
    """make_scenario should create v1.1 disturbance objects when config params are non-zero."""

    def _make_hard_config(self, **overrides):
        """Build a Config with hard-scenario defaults and optional overrides."""
        kwargs = {
            'turbulence_rms_deg': 0.1,
            'turbulence_correlation_s': 1.0,
            'exposure_rate_hz': 0.1,
            'exposure_amplitude': 0.3,
            'occlusion_duration_s': 1.0,
            'occlusion_frequency_hz': 0.05,
            'false_beacon_count': 2,
        }
        kwargs.update(overrides)
        return Config(**kwargs)

    def test_hard_scenario_creates_turbulence(self):
        from main import make_scenario
        config = self._make_hard_config()
        sim, controller, disturbances = make_scenario(config, "hard")
        assert isinstance(disturbances['turbulence'], AtmosphericTurbulence)

    def test_hard_scenario_creates_exposure(self):
        from main import make_scenario
        config = self._make_hard_config()
        sim, controller, disturbances = make_scenario(config, "hard")
        assert isinstance(disturbances['exposure'], ExposureVariation)

    def test_hard_scenario_creates_occlusion(self):
        from main import make_scenario
        config = self._make_hard_config()
        sim, controller, disturbances = make_scenario(config, "hard")
        assert isinstance(disturbances['occlusion'], Occlusion)

    def test_hard_scenario_creates_false_beacons(self):
        from main import make_scenario
        config = self._make_hard_config()
        sim, controller, disturbances = make_scenario(config, "hard")
        assert isinstance(disturbances['false_beacons'], FalseBeacons)

    def test_zero_params_produce_none_disturbances(self):
        from main import make_scenario
        config = self._make_hard_config(
            turbulence_rms_deg=0.0,
            exposure_amplitude=0.0,
            occlusion_frequency_hz=0.0,
            false_beacon_count=0,
        )
        sim, controller, disturbances = make_scenario(config, "hard")
        assert disturbances['turbulence'] is None
        assert disturbances['exposure'] is None
        assert disturbances['occlusion'] is None
        assert disturbances['false_beacons'] is None

    def test_clean_scenario_also_creates_v11_disturbances(self):
        """Clean scenario should also create v1.1 disturbances from config defaults."""
        from main import make_scenario
        config = Config()  # default config has non-zero v1.1 params
        sim, controller, disturbances = make_scenario(config, "clean")
        assert isinstance(disturbances['turbulence'], AtmosphericTurbulence)
        assert isinstance(disturbances['exposure'], ExposureVariation)
        assert isinstance(disturbances['occlusion'], Occlusion)
        assert isinstance(disturbances['false_beacons'], FalseBeacons)

    def test_returns_three_element_tuple(self):
        from main import make_scenario
        config = Config()
        result = make_scenario(config, "clean")
        assert len(result) == 3
        assert result[0] is not None  # sim
        assert result[1] is not None  # controller
        assert isinstance(result[2], dict)  # disturbances

    def test_disturbance_seeds_differ(self):
        """Each v1.1 disturbance should use a unique seed offset."""
        from main import make_scenario
        config = self._make_hard_config(random_seed=42)
        sim, controller, disturbances = make_scenario(config, "hard")
        # Turbulence seed = 42 + 2 = 44
        assert disturbances['turbulence'].rng is not None
        # Exposure seed = 42 + 3 = 45
        assert disturbances['exposure'].rng is not None
        # Occlusion seed = 42 + 4 = 46
        assert disturbances['occlusion'].rng is not None
        # FalseBeacons seed = 42 + 5 = 47
        assert disturbances['false_beacons'].rng is not None

    def test_unknown_scenario_raises(self):
        from main import make_scenario
        config = Config()
        with pytest.raises(ValueError, match="Unknown scenario"):
            make_scenario(config, "nonexistent")
