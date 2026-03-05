"""Shared fixtures for rosnav_rl unit tests.

Run tests with the ROS environment sourced:
    cd ~/arena5_ws && source arena
    cd src/Arena/arena_training/deps/rosnav_rl/rosnav_rl
    python -m pytest tests/ -v
"""

from unittest.mock import MagicMock
from dataclasses import dataclass

import numpy as np
import pytest

# Structured dtype matching rosnav_rl.observations.utils.types.Pose2DType
Pose2DType = np.dtype([("x", np.float32), ("y", np.float32), ("yaw", np.float32)])


def make_pose2d(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """Create a structured Pose2D numpy scalar matching Pose2DType."""
    pose = np.zeros(1, dtype=Pose2DType)[0]
    pose["x"] = x
    pose["y"] = y
    pose["yaw"] = yaw
    return pose


# ============================================================
# Simulation state container stubs
# ============================================================

@dataclass
class _RobotState:
    radius: float = 0.3
    safety_distance: float = 0.5

@dataclass
class _TaskState:
    goal_radius: float = 0.3
    max_steps: int = 500

@dataclass
class SimulationStateContainerStub:
    robot: _RobotState = None
    task: _TaskState = None

    def __post_init__(self):
        if self.robot is None:
            self.robot = _RobotState()
        if self.task is None:
            self.task = _TaskState()


# ============================================================
# Shared pytest fixtures
# ============================================================

@pytest.fixture
def sim_state():
    """Default simulation state container."""
    return SimulationStateContainerStub()


@pytest.fixture
def make_reward_function():
    """Factory that builds a minimal mock RewardFunction for unit testing."""
    def _factory():
        rf = MagicMock()
        rf.add_reward = MagicMock()
        rf.add_info = MagicMock()
        rf.state = MagicMock()
        rf.state.current_reward = 0.0
        rf.state.info = {}
        rf.state.reward_overview = {}
        rf.verbose = 0
        rf.parallel = False
        rf._lock = None

        # Track cumulative reward
        _reward_accum = {"total": 0.0}

        def _add_reward(value, called_by=None):
            _reward_accum["total"] += value
            rf.state.current_reward = _reward_accum["total"]

        def _add_info(info):
            rf.state.info.update(info)

        def _reset():
            _reward_accum["total"] = 0.0
            rf.state.current_reward = 0.0
            rf.state.info.clear()

        rf.add_reward.side_effect = _add_reward
        rf.add_info.side_effect = _add_info
        rf._reset = _reset
        rf._reward_accum = _reward_accum

        return rf

    return _factory
