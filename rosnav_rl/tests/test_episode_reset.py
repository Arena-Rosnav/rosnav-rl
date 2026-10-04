from __future__ import annotations

from typing import Type

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import rosnav_rl.spaces.observation_space.spaces as spaces
from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Algorithm_Cfg, PPO_Cfg
from rosnav_rl.model.stable_baselines3.cfg.framework import StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory
from rosnav_rl.model.stable_baselines3.policy.base_policy import (
    StableBaselinesPolicyDescription,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.classic import (
    EXTRACTOR_5,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

_NUM_BEAMS = 360

_OBS_KWARGS = {
    "normalize": True,
    "goal_max_dist": 10,
    "subgoal_max_dist": 10,
    "reduced_num_beams": _NUM_BEAMS,
    "laser_stack_size": 4,
    "feature_map_size": 40,
}

_OBS_SPACES = [
    spaces.perception.ReducedLaserScanSpace,
    spaces.navigation.DistAngleToSubgoalSpace,
    spaces.dynamics.LastActionSpace,
]


@AgentFactory.register("TEST_RESET_STACKED_LASER")
class _StackedLaserAgent(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = PPO
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = [*_OBS_SPACES, spaces.environment.StackedLaserMapSpace]
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = dict(pi=[32], vf=[32])
    activation_fn = nn.ReLU


@AgentFactory.register("TEST_RESET_FRAME_STACK")
class _FrameStackAgent(StableBaselinesPolicyDescription):
    algorithm_class: Type[BaseAlgorithm] = PPO
    stack_size = 3
    observation_space_kwargs = _OBS_KWARGS
    observation_spaces = _OBS_SPACES
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = dict(pi=[32], vf=[32])
    activation_fn = nn.ReLU


def _spec(name: str, algorithm) -> AgentConfig:
    return AgentConfig(
        name=name,
        action_space=DifferentialDriveActionSpace(
            linear_range=(-0.5, 0.5),
            angular_range=(-1.0, 1.0),
        ),
        parameters=AgentParameters(
            laser_num_beams=_NUM_BEAMS,
            laser_max_range=30.0,
            normalize=True,
        ),
        framework=StableBaselinesCfg(algorithm=algorithm),
    )


def _ppo(architecture_name: str) -> PPO_Cfg:
    return PPO_Cfg(
        architecture_name=architecture_name,
        parameters=PPO_Algorithm_Cfg(total_batch_size=64, batch_size=64),
    )


def _initialized_agent(spec: AgentConfig) -> RL_Agent:
    agent = RL_Agent(spec)
    agent.initialize_model(
        make_mock_env(
            ns="",
            space_manager=agent.space_manager,
            stack_size=agent.model._policy_description.stack_size,
        )
    )
    return agent


def _raw_observation(laser_range: float, subgoal_dist: float) -> dict:
    return {
        "front_laser": np.full(_NUM_BEAMS, laser_range, dtype=np.float32),
        "dist_angle_to_subgoal": np.array([subgoal_dist, 0.3], dtype=np.float32),
        "last_action": np.zeros(3, dtype=np.float32),
        "is_terminal": False,
    }


def test_agent_reset_clears_stacked_laser_history():
    agent = RL_Agent(_spec("reset_stacked_laser", _ppo("TEST_RESET_STACKED_LASER")))
    stacked = next(
        space
        for space in agent.space_manager.observation_space_list
        if isinstance(space, spaces.environment.StackedLaserMapSpace)
    )
    first = stacked.encode_observation(front_laser=np.full(_NUM_BEAMS, 1.0, dtype=np.float32), is_terminal=False).copy()
    stacked.encode_observation(front_laser=np.full(_NUM_BEAMS, 7.0, dtype=np.float32), is_terminal=False)

    agent.reset()

    after_reset = stacked.encode_observation(front_laser=np.full(_NUM_BEAMS, 1.0, dtype=np.float32), is_terminal=False)
    np.testing.assert_array_equal(after_reset, first)


def test_reset_restarts_the_frame_stack_from_the_next_observation():
    agent = _initialized_agent(_spec("reset_frame_stack", _ppo("TEST_RESET_FRAME_STACK")))
    probe = _raw_observation(laser_range=2.0, subgoal_dist=3.0)

    fresh_action = agent.get_action(probe).copy()
    agent.get_action(_raw_observation(laser_range=9.0, subgoal_dist=8.0))
    agent.get_action(_raw_observation(laser_range=0.5, subgoal_dist=1.0))
    continued_action = agent.get_action(probe).copy()

    agent.reset()

    np.testing.assert_allclose(agent.get_action(probe), fresh_action)
    assert not np.allclose(continued_action, fresh_action)
