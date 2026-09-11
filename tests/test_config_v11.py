"""Tests for v1.1 Config fields and YAML config with all motion types."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config


class TestConfigV11Fields:
    """Tests for new Config dataclass fields added in v1.1."""

    def test_filter_type_default(self):
        """filter_type defaults to kalman."""
        cfg = Config()
        assert cfg.filter_type == "kalman"

    def test_filter_type_alpha_beta(self):
        """filter_type can be set to alpha_beta."""
        cfg = Config(filter_type="alpha_beta")
        assert cfg.filter_type == "alpha_beta"

    def test_detector_type_default(self):
        """detector_type defaults to classical."""
        cfg = Config()
        assert cfg.detector_type == "classical"

    def test_detector_type_ai(self):
        """detector_type can be set to ai."""
        cfg = Config(detector_type="ai", ai_model_path="model.onnx")
        assert cfg.detector_type == "ai"
        assert cfg.ai_model_path == "model.onnx"

    def test_alpha_beta_params(self):
        """Alpha-beta filter params are stored."""
        cfg = Config(alpha=0.7, beta=0.2)
        assert cfg.alpha == 0.7
        assert cfg.beta == 0.2

    def test_alpha_beta_defaults(self):
        """Alpha-beta params have correct defaults."""
        cfg = Config()
        assert cfg.alpha == 0.5
        assert cfg.beta == 0.1

    def test_multi_target_fields(self):
        """Multi-target config fields exist."""
        cfg = Config(multi_target=True, num_targets=3)
        assert cfg.multi_target is True
        assert cfg.num_targets == 3

    def test_v11_disturbance_fields(self):
        """v1.1 disturbance config fields exist with defaults."""
        cfg = Config()
        assert hasattr(cfg, 'turbulence_rms_deg')
        assert hasattr(cfg, 'exposure_rate_hz')
        assert hasattr(cfg, 'occlusion_duration_s')
        assert hasattr(cfg, 'false_beacon_count')

    def test_v11_disturbance_custom(self):
        """v1.1 disturbance fields can be customized."""
        cfg = Config(
            turbulence_rms_deg=0.2,
            exposure_rate_hz=0.5,
            occlusion_duration_s=2.0,
            false_beacon_count=5,
        )
        assert cfg.turbulence_rms_deg == 0.2
        assert cfg.exposure_rate_hz == 0.5
        assert cfg.occlusion_duration_s == 2.0
        assert cfg.false_beacon_count == 5

    def test_all_fields_present(self):
        """Config has all expected field groups."""
        cfg = Config()
        # Simulation
        assert hasattr(cfg, 'fps')
        assert hasattr(cfg, 'duration_s')
        assert hasattr(cfg, 'random_seed')
        # Camera
        assert hasattr(cfg, 'width')
        assert hasattr(cfg, 'height')
        assert hasattr(cfg, 'h_fov_deg')
        # PID
        assert hasattr(cfg, 'kp')
        assert hasattr(cfg, 'ki')
        assert hasattr(cfg, 'kd')
        # Tracker
        assert hasattr(cfg, 'acquire_threshold')
        assert hasattr(cfg, 'lose_threshold')
        # v1.1 additions
        assert hasattr(cfg, 'filter_type')
        assert hasattr(cfg, 'detector_type')
