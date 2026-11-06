"""Tests for the crowd reachability pipeline: candidate velocities, scalars, generator, space and AGENT_5_REACH."""

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
from rosnav_rl.observations.data_sources.generators import CrowdReachabilityGenerator
from rosnav_rl.observations.utils.semantic import candidate_velocities, reachability_scalars
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdReachabilitySpace
from rosnav_rl.utils.utils import make_mock_env

PARAMS = AgentParameters(robot_radius=0.4, max_linear_vel=1.0)
RADIUS = 0.7  # robot 0.4 plus pedestrian 0.3
CANDIDATES = candidate_velocities(1.0)


def _scalars(locations, velocities, **kwargs):
    return reachability_scalars(
        np.asarray(locations, dtype=np.float64).reshape(-1, 2),
        np.asarray(velocities, dtype=np.float64).reshape(-1, 2),
        CANDIDATES,
        contact_radius=RADIUS,
        horizon=3.0,
        **kwargs,
    )


def _reachability(locations, velocities, **kwargs):
    return CrowdReachabilityGenerator(name="crowd_reachability", **kwargs)._generate(
        np.asarray(locations, dtype=np.float32), np.asarray(velocities, dtype=np.float32), PARAMS
    )


def test_candidates_are_three_speeds_times_sixteen_headings():
    assert CANDIDATES.shape == (48, 2)
    np.testing.assert_allclose(np.hypot(*CANDIDATES.T), np.repeat([0.0, 0.5, 1.0], 16), atol=1e-12)
    assert len({tuple(np.round(c, 9)) for c in CANDIDATES}) == 33  # the 16 standing candidates coincide


def test_candidate_headings_are_symmetric_about_the_forward_axis():
    mirrored = CANDIDATES * np.array([1.0, -1.0])

    for c in mirrored:
        assert np.isclose(np.hypot(*(CANDIDATES - c).T), 0.0, atol=1e-9).any()


def test_no_pedestrians_blocks_nothing():
    assert _scalars(np.empty((0, 2)), np.empty((0, 2))).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_a_distant_standing_pedestrian_blocks_nothing():
    assert _scalars([[10.0, 0.0]], [[0.0, 0.0]]).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_overlapping_pedestrian_blocks_every_candidate():
    np.testing.assert_allclose(_scalars([[0.3, 0.0]], [[0.0, 0.0]]), [1.0, 0.0, 1.0, 1.0])


def test_a_standing_pedestrian_ahead_blocks_only_the_candidates_that_reach_it_in_time():
    out = _scalars([[2.0, 0.0]], [[0.0, 0.0]])

    # contact needs 1.3 m: full speed (1.3 s) and half speed (2.6 s) straight ahead, nothing that stands or backs off
    assert out[0] == out[3] > 0.0
    assert out[0] < 0.5
    assert out[2] == 0.0


def test_a_pedestrian_walking_in_blocks_the_standing_candidates_too():
    out = _scalars([[3.0, 0.0]], [[-1.0, 0.0]])

    assert out[0] >= 16 / 48


def test_contact_exactly_at_the_horizon_counts_as_blocked():
    out = _scalars([[3.0 + RADIUS, 0.0]], [[0.0, 0.0]])

    assert out[0] == pytest.approx(1 / 48)


def test_a_pedestrian_behind_is_never_reached():
    assert _scalars([[-4.0, 0.0]], [[0.0, 0.0]])[0] == 0.0


def test_summary_orders_the_shares_and_the_union_bounds_them():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 8))
        out = _scalars(rng.uniform(-8, 8, (n, 2)), rng.uniform(-1.6, 1.6, (n, 2)))

        assert np.isfinite(out).all()
        assert out[0] >= out[1]
        assert out[0] <= out[3] <= 1.0
        assert out[3] <= n * out[0]


def test_reflecting_the_scene_leaves_the_scalars_unchanged():
    rng = np.random.default_rng(1)
    flip = np.array([1.0, -1.0])
    for _ in range(100):
        locations, velocities = rng.uniform(-8, 8, (5, 2)), rng.uniform(-1.5, 1.5, (5, 2))

        np.testing.assert_allclose(
            _scalars(locations, velocities), _scalars(locations * flip, velocities * flip), atol=1e-6
        )


def test_generator_ignores_pedestrians_beyond_the_range():
    near = _reachability([[2.0, 0.0]], [[0.0, 0.0]])
    both = _reachability([[2.0, 0.0], [30.0, 0.0]], [[0.0, 0.0], [-10.0, 0.0]])

    np.testing.assert_allclose(both, near)


def test_generator_takes_pedestrian_velocities_as_they_are_not_relative_to_the_robot():
    # the candidate replaces the robot velocity, so there is no robot twist to subtract and none is required
    assert "robot_twist" not in CrowdReachabilityGenerator.requires
    assert set(CrowdReachabilityGenerator.requires) == {
        "pedestrian_relative_locations",
        "pedestrian_relative_velocities",
    }


def test_generator_scales_the_candidates_with_the_rated_speed():
    slow = _reachability([[2.0, 0.0]], [[0.0, 0.0]], max_speed=0.3)
    fast = _reachability([[2.0, 0.0]], [[0.0, 0.0]], max_speed=1.0)

    assert slow[0] == 0.0 < fast[0]


def test_generator_without_pedestrians_returns_zeros():
    assert _reachability(np.empty((0, 2)), np.empty((0, 2))).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_space_clips_the_count_and_normalizes_into_the_unit_box():
    space = CrowdReachabilitySpace(reach_count_cap=10, normalize=True)

    encoded = space.encode_observation(np.array([0.0, 1.0, 25.0, 0.5], dtype=np.float32))

    np.testing.assert_allclose(encoded, [-1.0, 1.0, 1.0, 0.0], atol=1e-6)


def _reach_agent() -> RL_Agent:
    return RL_Agent(
        AgentConfig(
            name="reach_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=720, robot_radius=0.4),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name="AGENT_5_REACH", parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )


def test_bundled_observations_provide_what_the_reachability_agent_requires():
    agent = _reach_agent()
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    provided = (set(config["datasources"]) - set(config["aliases"].values())) | set(config["aliases"]) | {"is_terminal"}

    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }

    assert required <= provided
    assert config["datasources"]["crowd_reachability"]["type"] == "CrowdReachabilityGenerator"


def test_reachability_agent_differs_from_the_ttc_agent_only_in_its_scalar_space():
    from rosnav_rl.model.stable_baselines3.policy.agent_factory import AgentFactory

    reach, ttc = AgentFactory.registry["AGENT_5_REACH"], AgentFactory.registry["AGENT_5_TTC"]

    assert [s.name for s in reach.observation_spaces[:-1]] == [s.name for s in ttc.observation_spaces[:-1]]
    assert reach.observation_spaces[-1].name == "CrowdReachabilitySpace"
    assert reach.features_extractor_class is ttc.features_extractor_class
    assert reach.net_arch == ttc.net_arch


def test_reachability_agent_head_reads_the_scalars_next_to_the_robot_state():
    agent = _reach_agent()
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    extractor = agent.model.model.policy.features_extractor
    observation = {
        key: np.zeros((1, *space.shape), dtype=np.float32)
        for key, space in agent.space_manager.observation_space.spaces.items()
    }

    blocked = dict(observation, CrowdReachabilitySpace=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))

    extractor.eval()
    with torch.no_grad():
        free_features = extractor({k: torch.as_tensor(v) for k, v in observation.items()})
        blocked_features = extractor({k: torch.as_tensor(v) for k, v in blocked.items()})

    assert extractor._last_action_size == agent.space_manager.observation_space["LastActionSpace"].shape[-1] + 4
    assert not torch.allclose(free_features, blocked_features)
