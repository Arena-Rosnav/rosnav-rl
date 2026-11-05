"""Smoke tests for policy registration and the AGENT_5 observation contract."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest
import torch
import yaml

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env


def _agent(name: str) -> RL_Agent:
    return RL_Agent(
        AgentConfig(
            name=name.lower(),
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=720, robot_radius=0.4),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name=name, parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )


@pytest.mark.parametrize("name", ["AGENT_5_HAM", "AGENT_5_PLAIN"])
def test_bundled_observations_provide_what_the_agent_requires(name):
    agent = _agent(name)
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    provided = (set(config["datasources"]) - set(config["aliases"].values())) | set(config["aliases"]) | {"is_terminal"}
    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }

    assert required <= provided
    baseline = _agent("AGENT_5")
    baseline_keys = set(baseline.space_manager.observation_space.spaces)
    assert set(agent.space_manager.observation_space.spaces) == baseline_keys
    assert "CrowdSetSpace" not in agent.space_manager.observation_space.spaces
    assert "CrowdRobotPoseSpace" not in agent.space_manager.observation_space.spaces
    # Same named spaces and tensor shapes as AGENT_5, with a shared 24 m map ROI
    # so the predictor can observe the full 12 m radius on all three experiment arms.
    from rosnav_rl.model.stable_baselines3.policy.sb3_policy.paper import AGENT_5, AGENT_5_HAM
    assert AGENT_5_HAM.observation_spaces == AGENT_5.observation_spaces
    assert AGENT_5.observation_space_kwargs["roi_in_m"] == 24
    assert AGENT_5_HAM.observation_space_kwargs["roi_in_m"] == 24


@pytest.mark.parametrize("name", ["AGENT_5_HAM", "AGENT_5_PLAIN"])
def test_agent_builds_and_extractor_runs(name):
    agent = _agent(name)
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    extractor = agent.model.model.policy.features_extractor
    observation = {
        key: torch.zeros((1, *space.shape), dtype=torch.float32)
        for key, space in agent.space_manager.observation_space.spaces.items()
    }
    # The empty physical pedestrian type is -1, represented by -1 under max_abs.
    observation["PedestrianTypeSpace"].fill_(-1.0)
    extractor.eval()
    out = extractor(observation)

    assert set(agent.space_manager.observation_space.spaces) == set(_agent("AGENT_5").space_manager.observation_space.spaces)
    assert out.shape[0] == 1 and torch.isfinite(out).all()
