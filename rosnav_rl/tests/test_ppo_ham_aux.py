"""PPO_HamAux trains end to end on a mock env and fits the predictor to a synthetic crowd."""

from __future__ import annotations

import pytest
import torch
from stable_baselines3.common.utils import configure_logger

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env


def _model(name):
    agent = RL_Agent(
        AgentConfig(
            name=name.lower(),
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=720, robot_radius=0.4),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name=name, parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    model = agent.model.model
    model.n_steps = 64
    model.batch_size = 32
    model.n_epochs = 1
    model.rollout_buffer.buffer_size = 64
    model.rollout_buffer.reset()
    model.set_logger(configure_logger(0, None, None, False))
    return model


@pytest.mark.parametrize("name", ["AGENT_5_HAM", "AGENT_5_PLAIN"])
def test_train_runs_on_a_filled_buffer(name):
    model = _model(name)
    rb = model.rollout_buffer
    for key, arr in rb.observations.items():
        arr[...] = 0
    rb.full = True
    model.train()


def test_aux_loss_drops_when_fitting_a_constant_velocity_crowd():
    model = _model("AGENT_5_HAM")
    rb = model.rollout_buffer
    crowd = rb.observations["CrowdSetSpace"]
    for t in range(64):
        crowd[t, 0, 0] = [2.0 + 0.5 * 0.1 * t, 0.0, 0.5, 0.0, 1.0, 3.0]
    rb.full = True
    pred = model.policy.features_extractor.predictor
    args = [torch.as_tensor(x) for x in model._aux_data()]
    before = float(model._aux_loss(pred, *args).detach())
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        loss = model._aux_loss(pred, *args)
        loss.backward()
        opt.step()
    after = float(model._aux_loss(pred, *args).detach())
    assert after < before
