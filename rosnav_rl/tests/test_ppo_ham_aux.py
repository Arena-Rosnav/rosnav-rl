"""PPO_HamAux trains end to end on a mock env and fits the predictor to a synthetic crowd."""

from __future__ import annotations

import numpy as np
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
                algorithm=PPO_Cfg(
                    architecture_name=name,
                    parameters=PPO_Algorithm_Cfg(
                        device="cpu", total_batch_size=64, batch_size=32, n_epochs=1, total_timesteps=64, aux_weight=0.37
                    ),
                )
            ),
        )
    )
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    model = agent.model.model
    assert model.n_steps == 64
    assert model.batch_size == 32
    assert model.n_epochs == 1
    assert model.aux_weight == 0.37
    assert model.rollout_buffer.buffer_size == 64
    model.set_logger(configure_logger(0, None, None, False))
    return model


@pytest.mark.parametrize("name", ["AGENT_5_HAM", "AGENT_5_PLAIN"])
def test_train_runs_on_a_filled_buffer(name):
    model = _model(name)
    rb = model.rollout_buffer
    for key, arr in rb.observations.items():
        arr[...] = 0
    if "PedestrianTypeSpace" in rb.observations:
        rb.observations["PedestrianTypeSpace"][:rb.buffer_size] = -1.0
    model._ham_rollout_crowd = np.zeros((rb.buffer_size, rb.n_envs, 24, 6), dtype=np.float32)
    model._ham_rollout_pose = np.zeros((rb.buffer_size, rb.n_envs, 3), dtype=np.float32)
    rb.full = True
    aux_data = model._aux_data()
    assert aux_data[3].shape == (rb.buffer_size * rb.n_envs, 3)
    model.train()


def test_aux_loss_drops_when_fitting_a_constant_velocity_crowd():
    model = _model("AGENT_5_HAM")
    rb = model.rollout_buffer
    crowd = np.zeros((rb.buffer_size, rb.n_envs, 24, 6), dtype=np.float32)
    maps = rb.observations
    type_map = maps["PedestrianTypeSpace"]
    vx_map = maps["PedestrianVelXSpace"]
    vy_map = maps["PedestrianVelYSpace"]
    type_map[...] = -1.0
    vx_map[...] = 0.0
    vy_map[...] = 0.0
    roi, size = 24.0, type_map.shape[-1]
    cell = roi / size
    for t in range(rb.buffer_size):
        x = 2.0 + 0.5 * 0.1 * t
        crowd[t, 0, 0] = [x, 0.0, 0.5, 0.0, 1.0, 3.0]
        gx = int(np.floor(x / cell + size // 2))
        gy = size // 2
        # AGENT_5's pedestrian map encoders emit raw type codes and m/s values;
        # only LastActionSpace applies normalization.
        type_map[t, 0, gy, gx] = 0.0
        vx_map[t, 0, gy, gx] = 0.5
        vy_map[t, 0, gy, gx] = 0.0
    model._ham_rollout_crowd = crowd
    model._ham_rollout_pose = np.zeros((rb.buffer_size, rb.n_envs, 3), dtype=np.float32)
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


def test_vecenv_reset_infos_skips_empty_wrapper_state():
    from types import SimpleNamespace
    from rosnav_rl.model.stable_baselines3.policy.ham_aux import (
        CROWD_INFO_KEY,
        POSE_INFO_KEY,
        PPO_HamAux,
    )

    valid_info = {
        CROWD_INFO_KEY: np.zeros((24, 6), dtype=np.float32),
        POSE_INFO_KEY: np.zeros(3, dtype=np.float32),
    }
    inner = SimpleNamespace(reset_infos=[valid_info], venv=None)
    outer = SimpleNamespace(reset_infos=[{}], venv=inner)

    infos = PPO_HamAux._vecenv_reset_infos(outer)
    assert infos == [valid_info]


def test_active_vecenv_reference_is_excluded_from_save(tmp_path):
    from multiprocessing import current_process

    model = _model("AGENT_5_HAM")
    # Reproduce the non-picklable object that caused the real save failure.
    model._ham_active_env = current_process().authkey
    path = tmp_path / "ham_save_test.zip"
    model.save(path)
    assert path.is_file()
    assert path.stat().st_size > 0
