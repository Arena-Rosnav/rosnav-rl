from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
from rosnav_rl.model.dreamerv3.exploration import Random
from rosnav_rl.model.dreamerv3.networks import MLP
from rosnav_rl.rl_agent import RL_Agent

_NUM_BEAMS = 360


@pytest.mark.parametrize("dist", ["onehot", "normal"])
def test_random_exploration_samples_one_action_per_feature_row(dist: str):
    config = DreamerV3Cfg()
    config.model.actor.dist = dist
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    actor = Random(config, act_space).actor(torch.zeros(5, 16))

    assert actor.sample().shape == (5, 2)


def test_huber_head_clamps_to_absmax():
    head = MLP(inp_dim=4, shape=(2,), layers=1, units=8, dist="huber", absmax=0.25, device="cpu")

    mode = head(torch.full((3, 4), 50.0)).mode()

    assert mode.shape == (3, 2)
    assert torch.all(mode.abs() <= 0.25 + 1e-6)


def _dreamer_agent(tmp_path, exploration: str = "greedy") -> RL_Agent:
    framework = DreamerV3Cfg()
    framework.general.logdir = tmp_path
    framework.model.exploration.behavior = exploration
    agent = RL_Agent(
        AgentConfig(
            name="dreamer_deploy",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=_NUM_BEAMS, laser_max_range=30.0),
            framework=framework,
        )
    )
    agent.initialize_model()
    return agent


def _raw_observation(laser_range: float) -> dict:
    return {
        "front_laser": np.full(_NUM_BEAMS, laser_range, dtype=np.float32),
        "pedestrian_relative_locations": np.zeros((0, 2), dtype=np.float32),
        "pedestrian_vel_x": np.zeros(0, dtype=np.float32),
        "pedestrian_vel_y": np.zeros(0, dtype=np.float32),
        "pedestrian_social_states": np.zeros(0, dtype=np.int32),
        "pedestrian_types": np.zeros(0, dtype=np.int32),
        "dist_angle_to_goal": np.array([3.0, 0.2], dtype=np.float32),
        "last_action": np.zeros(3, dtype=np.float32),
    }


def test_dreamer_get_action_returns_a_decoded_action_and_keeps_state(tmp_path):
    agent = _dreamer_agent(tmp_path)

    first = agent.get_action(_raw_observation(2.0))
    agent.get_action(_raw_observation(5.0))

    assert first.shape == (3,)
    assert agent.model._policy_state is not None
    agent.reset()
    assert agent.model._policy_state is None


def test_dreamer_builds_plan2explore_exploration(tmp_path):
    agent = _dreamer_agent(tmp_path, exploration="plan2explore")

    assert agent.get_action(_raw_observation(2.0)).shape == (3,)


def test_dreamer_agent_config_without_name_gets_generated_name():
    spec = AgentConfig(
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.0, 1.0)),
        parameters=AgentParameters(laser_num_beams=_NUM_BEAMS, laser_max_range=30.0),
        framework=DreamerV3Cfg(),
    )

    assert spec.name
