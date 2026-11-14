from __future__ import annotations

import numpy as np
import torch

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.dreamerv3.cfg import DreamerV3Cfg
from rosnav_rl.model.dreamerv3.exploration import Random
from rosnav_rl.rl_agent import RL_Agent

_ACTION_SPACE = DifferentialDriveActionSpace(linear_range=(0.0, 0.5), angular_range=(-1.5, 1.5))


def _agent(tmp_path, normalized: bool) -> RL_Agent:
    framework = DreamerV3Cfg()
    framework.general.logdir = tmp_path
    framework.model.actor.dist = "normal"
    agent = RL_Agent(
        AgentConfig(
            name=f"dreamer_normalized_{normalized}",
            action_space=_ACTION_SPACE,
            parameters=AgentParameters(laser_num_beams=360, laser_max_range=30.0),
            framework=framework,
            normalized_actions=normalized,
        )
    )
    agent.initialize_model()
    return agent


def test_dreamer_is_built_on_the_policy_action_space(tmp_path):
    agent = _agent(tmp_path, True)

    act_space = agent.model.model._act_space

    np.testing.assert_allclose(act_space.low, agent.space_manager.policy_action_space.low)
    np.testing.assert_allclose(act_space.high, agent.space_manager.policy_action_space.high)
    np.testing.assert_allclose(act_space.low, [-1.0, -1.0])
    np.testing.assert_allclose(act_space.high, [1.0, 1.0])


def test_random_explorer_and_actor_share_the_policy_bounds(tmp_path):
    agent = _agent(tmp_path, True)
    dreamer = agent.model.model
    actor = dreamer._task_behavior.actor
    feat = torch.full((64, next(actor.parameters()).shape[1]), 50.0)

    explored = Random(dreamer._config, dreamer._act_space).actor(feat).sample()
    acted = actor(feat).mode()

    assert explored.abs().max() <= 1.0
    assert acted.abs().max() <= 1.0 + 1e-6


def test_unit_actor_output_maps_onto_the_configured_bounds(tmp_path):
    manager = _agent(tmp_path, True).space_manager

    np.testing.assert_allclose(manager.decode_action(np.array([-1.0, -1.0])), [0.0, 0.0, -1.5])
    np.testing.assert_allclose(manager.decode_action(np.array([1.0, 1.0])), [0.5, 0.0, 1.5])
    np.testing.assert_allclose(manager.decode_action(np.array([0.0, 0.0])), [0.25, 0.0, 0.0])


def test_unnormalized_dreamer_agents_keep_the_raw_action_space(tmp_path):
    agent = _agent(tmp_path, False)

    np.testing.assert_allclose(agent.model.model._act_space.high, [0.5, 1.5])
    np.testing.assert_allclose(agent.space_manager.decode_action(np.array([0.5, 1.0])), [0.5, 0.0, 1.0])
