"""The inference node runs at the agent's trained control rate unless the control_rate parameter is set."""

from __future__ import annotations

from pathlib import Path

import pytest
import rclpy
import yaml
from rclpy.node import Node
from rclpy.parameter import Parameter

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.inference.arena_inference_node import ArenaInferenceNode
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import make_mock_env

TRAINED_HZ = 20.0


def _saved_agent(directory: Path, general: dict) -> Path:
    spec = AgentConfig(
        name="rate_agent",
        action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 0.5), angular_range=(-1.5, 1.5)),
        parameters=AgentParameters(laser_num_beams=360, robot_radius=0.4),
        framework=StableBaselinesCfg(
            algorithm=PPO_Cfg(architecture_name="AGENT_5", parameters=PPO_Algorithm_Cfg(device="cpu", seed=3))
        ),
        normalized_actions=True,
    )
    agent = RL_Agent(spec)
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    agent.model.model.save(directory / "best_model")
    (directory / "training_config.yaml").write_text(
        yaml.safe_dump({"arena_cfg": {"general": general}, "agent_config": spec.model_dump(mode="json")})
    )
    return directory


def _inference(agent_dir: Path, *overrides: Parameter) -> ArenaInferenceNode:
    if not rclpy.ok():
        rclpy.init()
    node = Node(
        "rosnav_rl_inference",
        namespace="/rate_env/jackal",
        parameter_overrides=[
            Parameter("agent", value=str(agent_dir)),
            Parameter("arena_env_node", value="/rate_env/arena_env_node"),
            *overrides,
        ],
    )
    return ArenaInferenceNode(node)


def test_unset_control_rate_is_the_trained_control_hz(tmp_path):
    inference = _inference(_saved_agent(tmp_path, {"control_hz": TRAINED_HZ}))

    assert inference.control_rate == pytest.approx(TRAINED_HZ)
    inference.node.destroy_node()


def test_unset_control_rate_under_trained_lockstep_is_the_step_rate(tmp_path):
    inference = _inference(_saved_agent(tmp_path, {"lockstep": True, "lockstep_step_seconds": 0.05}))

    assert inference.control_rate == pytest.approx(20.0)
    inference.node.destroy_node()


def test_set_control_rate_overrides_the_trained_rate(tmp_path):
    inference = _inference(_saved_agent(tmp_path, {"control_hz": TRAINED_HZ}), Parameter("control_rate", value=5.0))

    assert inference.control_rate == pytest.approx(5.0)
    inference.node.destroy_node()
