"""Tests for ScriptedTrajectory motion model."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sim import ScriptedTrajectory, TargetState


class TestScriptedTrajectory:
    """Tests for the ScriptedTrajectory motion model."""

    def test_basic_trajectory(self):
        """Positions are returned in order."""
        positions = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)]
        traj = ScriptedTrajectory(positions, fps=1.0)
        state = TargetState()
        traj.update(state, dt=1.0, t=0.0)
        assert state.azimuth_deg == pytest.approx(0.0)
        assert state.elevation_deg == pytest.approx(0.0)

    def test_intermediate_time(self):
        """Time between frames indexes to correct position."""
        positions = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)]
        traj = ScriptedTrajectory(positions, fps=1.0)
        state = TargetState()
        traj.update(state, dt=1.0, t=1.5)
        # t=1.5 at fps=1 -> index = 1 (floor of 1.5)
        assert state.azimuth_deg == pytest.approx(1.0)
        assert state.elevation_deg == pytest.approx(0.5)

    def test_clamped_at_end(self):
        """Time beyond trajectory length clamps to last position."""
        positions = [(5.0, 3.0), (6.0, 4.0)]
        traj = ScriptedTrajectory(positions, fps=1.0)
        state = TargetState()
        traj.update(state, dt=1.0, t=100.0)
        assert state.azimuth_deg == pytest.approx(6.0)
        assert state.elevation_deg == pytest.approx(4.0)

    def test_fps_scaling(self):
        """Higher fps means faster index advancement."""
        positions = [(0.0, 0.0), (10.0, 5.0)]
        traj = ScriptedTrajectory(positions, fps=10.0)
        state = TargetState()
        # t=0.1 at fps=10 -> index = floor(1.0) = 1
        traj.update(state, dt=0.1, t=0.1)
        assert state.azimuth_deg == pytest.approx(10.0)

    def test_reset(self):
        """Reset returns to beginning."""
        positions = [(1.0, 2.0), (3.0, 4.0)]
        traj = ScriptedTrajectory(positions, fps=1.0)
        state = TargetState()
        traj.update(state, dt=1.0, t=1.0)
        traj.reset()
        state2 = TargetState()
        traj.update(state2, dt=1.0, t=0.0)
        assert state2.azimuth_deg == pytest.approx(1.0)
        assert state2.elevation_deg == pytest.approx(2.0)

    def test_negative_time(self):
        """Negative time clamps to first position."""
        positions = [(10.0, 20.0), (30.0, 40.0)]
        traj = ScriptedTrajectory(positions, fps=1.0)
        state = TargetState()
        traj.update(state, dt=1.0, t=-5.0)
        assert state.azimuth_deg == pytest.approx(10.0)
        assert state.elevation_deg == pytest.approx(20.0)

    def test_single_position(self):
        """Single position always returns that position."""
        traj = ScriptedTrajectory([(7.0, 8.0)], fps=30.0)
        state = TargetState()
        traj.update(state, dt=1.0/30.0, t=0.0)
        assert state.azimuth_deg == pytest.approx(7.0)
        assert state.elevation_deg == pytest.approx(8.0)
        state2 = TargetState()
        traj.update(state2, dt=1.0/30.0, t=5.0)
        assert state2.azimuth_deg == pytest.approx(7.0)
        assert state2.elevation_deg == pytest.approx(8.0)
