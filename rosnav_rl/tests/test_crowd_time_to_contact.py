"""Tests for the crowd time-to-contact pipeline: odometry twist, generator, space and AGENT_5_TTC."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest
import torch
import yaml
from nav_msgs.msg import Odometry

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.observations.data_sources.collectors import OdometryTwistCollector
from rosnav_rl.observations.data_sources.generators import CrowdTimeToContactGenerator
from rosnav_rl.observations.utils.semantic import time_to_contact
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdTimeToContactSpace
from rosnav_rl.utils.utils import make_mock_env

PARAMS = AgentParameters(robot_radius=0.4)


def _ttc(locations, velocities, twist=(0.0, 0.0, 0.0), **kwargs):
    return CrowdTimeToContactGenerator(name="crowd_ttc", **kwargs)._generate(
        np.asarray(locations, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        np.asarray(twist, dtype=np.float32),
        PARAMS,
    )


def test_odometry_twist_is_the_measured_base_frame_velocity():
    msg = Odometry()
    msg.twist.twist.linear.x = 0.8
    msg.twist.twist.linear.y = -0.1
    msg.twist.twist.angular.z = 0.5

    twist = OdometryTwistCollector(name="robot_twist", topic="odom")._preprocess(msg)

    np.testing.assert_allclose(twist, [0.8, -0.1, 0.5], rtol=1e-6)


def test_head_on_disc_closes_the_gap_minus_the_contact_radius():
    t = time_to_contact(np.array([[5.0, 0.0]]), np.array([[-1.0, 0.0]]), contact_radius=1.0, horizon=10.0)

    assert t[0] == pytest.approx(4.0)


def test_diverging_and_overlapping_discs():
    t = time_to_contact(
        np.array([[5.0, 0.0], [0.5, 0.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        contact_radius=1.0,
        horizon=10.0,
    )

    assert t.tolist() == [10.0, 0.0]


def test_a_standing_pedestrian_is_reached_only_through_the_measured_robot_velocity():
    standing = _ttc([[4.7, 0.0]], [[0.0, 0.0]])
    driving = _ttc([[4.7, 0.0]], [[0.0, 0.0]], twist=(1.0, 0.0, 0.0))

    assert standing[0] == pytest.approx(10.0)
    assert driving[0] == pytest.approx(4.0, abs=1e-5)


def test_summary_orders_counts_and_keeps_the_pedestrians_ahead():
    out = _ttc(
        [[2.7, 0.0], [0.0, 3.9], [8.7, 0.5], [30.0, 0.0]],
        [[-1.0, 0.0], [0.0, -1.0], [-2.0, 0.0], [-10.0, 0.0]],
    )

    np.testing.assert_allclose(out, [2.0, 3.2, 1.0, 2.0], atol=1e-5)


def test_ahead_ignores_a_closer_pedestrian_beside_the_robot():
    out = _ttc([[0.0, 2.7], [6.7, 0.0]], [[0.0, -1.0], [-1.0, 0.0]])

    np.testing.assert_allclose(out, [2.0, 6.0, 1.0, 6.0], atol=1e-5)


def test_no_pedestrians_reads_the_horizon_everywhere():
    out = _ttc(np.empty((0, 2)), np.empty((0, 2)))

    assert out.tolist() == [10.0, 10.0, 0.0, 10.0]


def test_space_clips_the_count_and_normalizes_into_the_unit_box():
    space = CrowdTimeToContactSpace(ttc_horizon=10.0, ttc_count_cap=10, normalize=True)

    encoded = space.encode_observation(np.array([0.0, 10.0, 25.0, 5.0], dtype=np.float32))

    np.testing.assert_allclose(encoded, [-1.0, 1.0, 1.0, 0.0], atol=1e-6)


def _ttc_agent() -> RL_Agent:
    return RL_Agent(
        AgentConfig(
            name="ttc_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=720, robot_radius=0.4),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name="AGENT_5_TTC", parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )


def test_bundled_observations_provide_what_the_ttc_agent_requires():
    agent = _ttc_agent()
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    provided = (set(config["datasources"]) - set(config["aliases"].values())) | set(config["aliases"]) | {"is_terminal"}

    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }

    assert required <= provided
    assert config["datasources"]["crowd_ttc"]["type"] == "CrowdTimeToContactGenerator"
    assert config["datasources"]["robot_twist"]["type"] == "OdometryTwistCollector"


def test_ttc_agent_head_reads_the_scalars_next_to_the_robot_state():
    agent = _ttc_agent()
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    extractor = agent.model.model.policy.features_extractor
    observation = {
        key: np.zeros((1, *space.shape), dtype=np.float32)
        for key, space in agent.space_manager.observation_space.spaces.items()
    }

    near = dict(observation, CrowdTimeToContactSpace=np.array([[-1.0, -1.0, 1.0, -1.0]], dtype=np.float32))

    extractor.eval()
    with torch.no_grad():
        far_features = extractor({k: torch.as_tensor(v) for k, v in observation.items()})
        near_features = extractor({k: torch.as_tensor(v) for k, v in near.items()})

    assert extractor._last_action_size == agent.space_manager.observation_space["LastActionSpace"].shape[-1] + 4
    assert not torch.allclose(far_features, near_features)
