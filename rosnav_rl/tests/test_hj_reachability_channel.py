"""Tests for the HJ reachability-tube pipeline: generators, space, AGENT_5_HJ, mirroring."""

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
from rosnav_rl.observations.data_sources.generators import CrowdReachabilityTubeGenerator, PedestrianTubeValueGenerator
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdReachabilityTubeSpace
from rosnav_rl.utils.utils import make_mock_env

PARAMS = AgentParameters(robot_radius=0.267)


def test_pedestrian_tube_value_generator_returns_empty_for_no_pedestrians():
    out = PedestrianTubeValueGenerator(name="pedestrian_tube_values")._generate(
        np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), PARAMS
    )
    assert out.shape == (0,)


def test_pedestrian_tube_value_generator_reads_a_close_standing_pedestrian_as_less_safe_than_a_far_one():
    gen = PedestrianTubeValueGenerator(name="pedestrian_tube_values")
    locations = np.array([[0.4, 0.0], [5.0, 0.0]], dtype=np.float32)
    velocities = np.zeros((2, 2), dtype=np.float32)
    close, far = gen._generate(locations, velocities, PARAMS)
    assert close < far


def test_pedestrian_tube_value_generator_reads_walking_toward_the_robot_as_less_safe_than_walking_away():
    gen = PedestrianTubeValueGenerator(name="pedestrian_tube_values")
    locations = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    velocities = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)  # toward the robot, then away
    toward, away = gen._generate(locations, velocities, PARAMS)
    assert toward < away


def test_pedestrian_tube_value_generator_value_changes_sensibly_as_the_pedestrian_turns():
    """Same position and speed, only the walking direction (psi) turns from head-on to sideways to away."""
    gen = PedestrianTubeValueGenerator(name="pedestrian_tube_values")
    position = np.array([2.0, 0.0], dtype=np.float32)
    headings = [np.pi, np.pi / 2, 0.0]  # walking: straight at the robot, sideways, straight away
    values = [
        gen._generate(position[None, :], np.array([[0.5 * np.cos(h), 0.5 * np.sin(h)]], dtype=np.float32), PARAMS)[0]
        for h in headings
    ]
    assert values[0] < values[2]  # head-on is less safe than walking away; sideways (values[1]) is in between


def test_crowd_reachability_tube_generator_reads_the_cap_with_no_pedestrians():
    gen = CrowdReachabilityTubeGenerator(name="crowd_tube", value_cap=1.5)
    out = gen._generate(np.empty((0, 2), dtype=np.float32), np.array([], dtype=np.float32), PARAMS)
    np.testing.assert_allclose(out, [1.5, 1.5, 0.0, 1.5])


def test_crowd_reachability_tube_generator_orders_counts_and_keeps_the_pedestrians_ahead():
    gen = CrowdReachabilityTubeGenerator(name="crowd_tube", value_cap=1.5, ahead_half_width=1.5)
    # the third pedestrian is within range but outside the ahead corridor
    locations = np.array([[2.0, 0.0], [-1.0, 0.5], [0.5, 3.0]], dtype=np.float32)
    values = np.array([-0.3, 0.2, 1.0], dtype=np.float32)
    out = gen._generate(locations, values, PARAMS)
    # minimum, second minimum, count inside the tube, minimum ahead (only the first pedestrian is ahead)
    np.testing.assert_allclose(out, [-0.3, 0.2, 1.0, -0.3])


def test_crowd_reachability_tube_generator_filters_by_max_range():
    gen = CrowdReachabilityTubeGenerator(name="crowd_tube", value_cap=1.5, max_range=12.0)
    locations = np.array([[100.0, 0.0]], dtype=np.float32)
    values = np.array([-5.0], dtype=np.float32)  # would read as very unsafe if not filtered out
    out = gen._generate(locations, values, PARAMS)
    np.testing.assert_allclose(out, [1.5, 1.5, 0.0, 1.5])


def _hj_agent() -> RL_Agent:
    return RL_Agent(
        AgentConfig(
            name="hj_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(
                laser_num_beams=640,
                laser_angle_min=-3.14159,
                laser_angle_increment=0.009817477,
                robot_radius=0.267,
                min_linear_vel=-0.5,
                max_linear_vel=0.5,
                min_angular_vel=-1.5,
                max_angular_vel=1.5,
                min_translational_vel=0.0,
                max_translational_vel=0.0,
            ),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name="AGENT_5_HJ", parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )


def test_bundled_observations_provide_what_the_hj_agent_requires():
    agent = _hj_agent()
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    provided = (set(config["datasources"]) - set(config["aliases"].values())) | set(config["aliases"]) | {"is_terminal"}

    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }

    assert required <= provided
    assert config["datasources"]["pedestrian_tube_values"]["type"] == "PedestrianTubeValueGenerator"
    assert config["datasources"]["crowd_tube"]["type"] == "CrowdReachabilityTubeGenerator"


def test_hj_agent_head_reads_the_tube_scalars_next_to_the_robot_state():
    agent = _hj_agent()
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    extractor = agent.model.model.policy.features_extractor
    observation = {
        key: np.zeros((1, *space.shape), dtype=np.float32)
        for key, space in agent.space_manager.observation_space.spaces.items()
    }

    near = dict(observation, CrowdReachabilityTubeSpace=np.array([[-1.5, -1.5, 1.0, -1.5]], dtype=np.float32))

    extractor.eval()
    with torch.no_grad():
        far_features = extractor({k: torch.as_tensor(v) for k, v in observation.items()})
        near_features = extractor({k: torch.as_tensor(v) for k, v in near.items()})

    assert extractor._last_action_size == agent.space_manager.observation_space["LastActionSpace"].shape[-1] + 4
    assert not torch.allclose(far_features, near_features)


def test_mirror_manager_accepts_the_hj_agent():
    agent = RL_Agent(
        AgentConfig(
            name="hj_agent_mirrored",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(
                laser_num_beams=640,
                laser_angle_min=-3.14159,
                laser_angle_increment=0.009817477,
                robot_radius=0.267,
                min_linear_vel=-0.5,
                max_linear_vel=0.5,
                min_angular_vel=-1.5,
                max_angular_vel=1.5,
                min_translational_vel=0.0,
                max_translational_vel=0.0,
            ),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name="AGENT_5_HJ", parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
            mirror="canonical",
        )
    )
    assert agent.space_manager.encode_observation is not None  # constructed without "no reflection rule"


def test_a_crowd_reachability_tube_space_refuses_a_mismatched_robot():
    with pytest.raises(ValueError, match="rated speed"):
        CrowdReachabilityTubeSpace(robot_id="jackal", max_linear_vel=99.0, max_angular_vel=1.5)
