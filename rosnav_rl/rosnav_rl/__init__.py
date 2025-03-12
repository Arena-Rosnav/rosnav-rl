"""
Rosnav-RL: Reinforcement Learning Framework

This module provides classes and utilities for training and deploying Reinforcement Learning (RL)
agents for robot navigation tasks in ROS environments.

The module includes:
- RL agent and model implementations (including DreamerV3 and StableBaselines3 support)
- Observation space management and collection
- Custom reward function framework
- Action space handling
- State management for agent and simulation
- Configuration structures

Version: 0.1.0
"""

__version__ = "0.1.0"
__all__ = [
    "RL_Agent",
    "RL_Model",
    "StableBaselinesModel",
    "DreamerV3Cfg",
    "DreamerV3Model",
    "dreamerv3",
    "stable_baselines3",
    "ActionSpaceManager",
    "BaseSpaceManager",
    "ObservationSpaceManager",
    "ObservationManager",
    "collectors",
    "generators",
    "static",
    "RewardFunction",
    "reward_units",
    "AgentStateContainer",
    "SimulationStateContainer",
    "AgentCfg",
    "RewardCfg",
    "SupportedRLFrameworks",
]


from .model import (
    # DreamerV3Cfg,
    DreamerV3Model,
    RL_Model,
    StableBaselinesModel,
    dreamerv3,
    stable_baselines3,
    
)
from .observations import ObservationManager, collectors, generators, static
from .reward import RewardFunction, reward_units
from .rl_agent import RL_Agent
from .spaces import ActionSpaceManager, BaseSpaceManager, ObservationSpaceManager
from .states import AgentStateContainer, SimulationStateContainer
from .cfg import AgentCfg, RewardCfg
from .utils.type_aliases import SupportedRLFrameworks
