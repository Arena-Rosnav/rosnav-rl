"""
Rosnav-RL: Reinforcement Learning Framework

This module provides classes and utilities for training and deploying Reinforcement Learning (RL)
agents for robot navigation tasks in ROS environments.

The module includes:
- RL agent and model implementations (including DreamerV3 and StableBaselines3 support)
- Observation space management and collection
- Custom reward function framework
- Action space handling
- State management for simulation
- Configuration structures

Version: 0.1.0
"""

__version__ = "0.1.0"
__all__ = [
    # ── Agent ──────────────────────────────────────────────
    "RL_Agent",
    # ── Config (single source of truth) ───────────────────
    "AgentConfig",
    "AgentCfg",   # backward-compat alias for AgentConfig
    "AgentParameters",
    "RewardCfg",
    # ── Typed action spaces ───────────────────────────────
    "ActionSpaceSpec",
    "DifferentialDriveActionSpace",
    "OmnidirectionalActionSpace",
    "DiscretizationCfg",
    "DiscretizationStrategy",
    # ── Models ────────────────────────────────────────────
    "RL_Model",
    "StableBaselinesModel",
    "DreamerV3Model",
    "dreamerv3",
    "stable_baselines3",
    # ── Spaces ────────────────────────────────────────────
    "ActionSpaceManager",
    "BaseSpaceManager",
    "ObservationSpaceManager",
    # ── Observations ──────────────────────────────────────
    "ObservationManager",
    # ── Reward ────────────────────────────────────────────
    "RewardFunction",
    "reward_units",

    # ── Utils ─────────────────────────────────────────────
    "SupportedRLFrameworks",
    "tuning",
]


from .model import RL_Model
from .reward import RewardFunction, reward_units
from .rl_agent import RL_Agent
from .spaces import ActionSpaceManager, BaseSpaceManager, ObservationSpaceManager
from .cfg import (
    AgentConfig,
    AgentParameters,
    RewardCfg,
    ActionSpaceSpec,
    DifferentialDriveActionSpace,
    OmnidirectionalActionSpace,
    DiscretizationCfg,
    DiscretizationStrategy,
)
AgentCfg = AgentConfig  # backward-compat alias
from .utils.type_aliases import SupportedRLFrameworks


def __getattr__(name: str):
    # ObservationManager pulls in rclpy transitively (.observations -> ROS
    # topic subscribers); deferred so `import rosnav_rl` works without ROS.
    if name == "ObservationManager":
        from .observations import ObservationManager

        return ObservationManager
    # Backend model classes/submodules are deferred to .model's own
    # __getattr__ so importing rosnav_rl doesn't pay for both
    # StableBaselinesModel's and DreamerV3Model's dependency chains.
    if name in ("StableBaselinesModel", "DreamerV3Model", "dreamerv3", "stable_baselines3"):
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
