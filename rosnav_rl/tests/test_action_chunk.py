"""Tests for chunked velocity actions on the rosnav_rl side."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

ACTION_SPACE = DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0))


def _spec(chunk: int, **algorithm) -> AgentConfig:
    return AgentConfig(
        name=f"chunk_{chunk}",
        action_space=ACTION_SPACE,
        parameters=AgentParameters(laser_num_beams=360),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_1", parameters=PPO_Algorithm_Cfg(device="cpu", **algorithm))
        ),
        action_chunk=chunk,
    )


def test_policy_space_tiles_the_command_bounds_per_tick():
    manager = RL_Agent(_spec(5)).space_manager

    space = manager.policy_action_space

    assert space.shape == (10,)
    np.testing.assert_allclose(space.low, np.tile([-0.5, -1.0], 5))
    np.testing.assert_allclose(space.high, np.tile([1.0, 1.0], 5))
    assert manager.action_space.shape == (2,)


def test_unchunked_policy_space_is_the_command_space():
    manager = RL_Agent(_spec(1)).space_manager

    assert manager.policy_action_space is manager.action_space


def test_a_chunk_decodes_into_one_command_per_tick():
    manager = RL_Agent(_spec(3)).space_manager

    commands = manager.decode_policy_action(np.array([0.1, 0.2, 0.3, -0.4, 0.5, 0.6]))

    np.testing.assert_allclose(commands, [[0.1, 0.0, 0.2], [0.3, 0.0, -0.4], [0.5, 0.0, 0.6]])


def test_chunking_is_refused_outside_stable_baselines3(tmp_path):
    framework = DreamerV3Cfg()
    framework.general.logdir = tmp_path

    with pytest.raises(ValidationError, match="stable_baselines3"):
        AgentConfig(name="dreamer_chunk", action_space=ACTION_SPACE, framework=framework, action_chunk=5)


def test_chunked_model_discounts_per_decision_and_acts_a_whole_chunk():
    agent = RL_Agent(_spec(5, gamma=0.99, gae_lambda=0.95, seed=0))
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    observation = {
        "front_laser": np.full(360, 4.0, dtype=np.float32),
        "dist_angle_to_subgoal": np.array([2.0, 0.3], dtype=np.float32),
        "last_action": np.array([0.2, 0.1], dtype=np.float32),
    }
    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }
    assert required <= set(observation)

    commands = agent.get_action(observation)

    assert agent.model.model.gamma == pytest.approx(0.99**5)
    assert agent.model.model.gae_lambda == pytest.approx(0.95**5)
    assert agent.model.model.action_space.shape == (10,)
    assert commands.shape == (5, 3)
    assert np.all(commands[:, 0] >= -0.5) and np.all(commands[:, 0] <= 1.0)
    assert np.all(commands[:, 1] == 0.0)


def test_saved_agent_config_keeps_the_chunk(tmp_path):
    spec = _spec(5)
    path = tmp_path / "agent.yaml"

    spec.to_yaml(path)
    loaded = AgentConfig.from_yaml(path)

    assert loaded.action_chunk == 5
