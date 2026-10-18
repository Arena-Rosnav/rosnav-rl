"""A policy loaded for inference acts as the saved policy predicts, on its trained BatchNorm statistics, and inference leaves those statistics untouched."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

BEAMS = 720


def _spec() -> AgentConfig:
    return AgentConfig(
        name="eval_mode",
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.5, 1.5)),
        parameters=AgentParameters(
            laser_num_beams=BEAMS, robot_radius=0.4, min_translational_vel=0.0, max_translational_vel=0.0
        ),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_5", parameters=PPO_Algorithm_Cfg(device="cpu", seed=3))
        ),
        normalized_actions=True,
    )


def _scene() -> dict:
    velocities = np.array([[0.4, 0.3], [-0.6, -0.2]], dtype=np.float32)
    return {
        "front_laser": np.linspace(1.0, 9.0, BEAMS, dtype=np.float32),
        "is_terminal": False,
        "pedestrian_relative_locations": np.array([[1.4, -1.1], [4.1, 2.6]], dtype=np.float32),
        "pedestrian_relative_velocities": velocities,
        "pedestrian_vel_x": velocities[:, 0],
        "pedestrian_vel_y": velocities[:, 1],
        "pedestrian_types": np.zeros(2, dtype=np.int32),
        "pedestrian_social_states": np.array([1, 2], dtype=np.int32),
        "dist_angle_to_subgoal": np.array([3.0, 0.6], dtype=np.float32),
        "last_action": np.array([0.2, 0.0, -0.4], dtype=np.float32),
    }


def _batch_norms(policy: torch.nn.Module) -> list[torch.nn.modules.batchnorm._BatchNorm]:
    return [m for m in policy.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def test_a_loaded_agent_acts_on_the_trained_batch_statistics(tmp_path):
    trained = RL_Agent(_spec())
    trained.initialize_model(make_mock_env(ns="", space_manager=trained.space_manager))
    generator = torch.Generator().manual_seed(0)
    for norm in _batch_norms(trained.model.model.policy):
        norm.running_mean.copy_(torch.rand(norm.running_mean.shape, generator=generator) - 0.5)
        norm.running_var.copy_(torch.rand(norm.running_var.shape, generator=generator) + 0.5)
    trained.model.model.save(tmp_path / "policy")

    agent = RL_Agent(_spec())
    agent.load_model(path=tmp_path / "policy.zip")
    saved = [norm.running_mean.clone() for norm in _batch_norms(agent.model.model.policy)]
    expected, _ = PPO.load(tmp_path / "policy.zip", device="cpu").predict(
        RL_Agent(_spec()).space_manager.encode_observation(_scene()), deterministic=True
    )

    command = agent.get_action(_scene())
    for _ in range(3):
        agent.get_action(_scene())

    np.testing.assert_allclose(command, agent.space_manager.decode_policy_action(expected), atol=1e-5)
    assert all(
        torch.equal(norm.running_mean, before)
        for norm, before in zip(_batch_norms(agent.model.model.policy), saved, strict=True)
    )
