"""Tests for normalized policy actions."""

from __future__ import annotations

import numpy as np

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent

ACTION_SPACE = DifferentialDriveActionSpace(linear_range=(-2.0, 2.0), angular_range=(-4.0, 4.0))


def _spec(normalized: bool, chunk: int = 1, **algorithm) -> AgentConfig:
    return AgentConfig(
        name=f"normalized_{normalized}_{chunk}",
        action_space=ACTION_SPACE,
        parameters=AgentParameters(laser_num_beams=360),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_1", parameters=PPO_Algorithm_Cfg(device="cpu", **algorithm))
        ),
        normalized_actions=normalized,
        action_chunk=chunk,
    )


def test_policy_acts_in_the_unit_box():
    manager = RL_Agent(_spec(True)).space_manager

    np.testing.assert_allclose(manager.action_space.low, [-1.0, -1.0])
    np.testing.assert_allclose(manager.action_space.high, [1.0, 1.0])


def test_unit_actions_map_linearly_onto_the_robot_limits():
    manager = RL_Agent(_spec(True)).space_manager

    np.testing.assert_allclose(manager.decode_action(np.array([-1.0, 1.0])), [-2.0, 0.0, 4.0])
    np.testing.assert_allclose(manager.decode_action(np.array([0.0, 0.0])), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(manager.decode_action(np.array([0.5, -0.25])), [1.0, 0.0, -1.0])
    np.testing.assert_allclose(manager.decode_action(np.array([3.0, -7.0])), [2.0, 0.0, -4.0])


def test_unnormalized_agents_keep_the_robot_limits():
    manager = RL_Agent(_spec(False)).space_manager

    np.testing.assert_allclose(manager.action_space.high, [2.0, 4.0])
    np.testing.assert_allclose(manager.decode_action(np.array([0.5, -0.25])), [0.5, 0.0, -0.25])


def test_chunks_of_unit_actions_decode_per_tick():
    manager = RL_Agent(_spec(True, chunk=2)).space_manager

    np.testing.assert_allclose(manager.policy_action_space.high, np.ones(4))
    np.testing.assert_allclose(
        manager.decode_policy_action(np.array([1.0, 0.0, -1.0, 0.5])), [[2.0, 0.0, 0.0], [-2.0, 0.0, 2.0]]
    )
