"""Advanced spaces built from non-default AgentParameters the way the space manager builds them."""

import importlib.resources

import numpy as np
import pytest
from rosnav_rl.utils.yaml_io import loads_yaml

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.spaces.observation_space.spaces.dynamics.advanced_dynamics_spaces import (
    KinematicStateSpace,
    MotionStateSpace,
    TrajectoryStateSpace,
)
from rosnav_rl.spaces.observation_space.spaces.environment.advanced_environment_spaces import (
    EnvironmentContextSpace,
    ObstacleProximitySpace,
    SpatialAwarenessSpace,
)
from rosnav_rl.spaces.observation_space.spaces.localization.advanced_localization_spaces import (
    LocalizationCombinedSpace,
    RobustOdometrySpace,
)
from rosnav_rl.spaces.observation_space.spaces.meta.advanced_meta_spaces import (
    MissionContextSpace,
    PerformanceContextSpace,
    SafetyContextSpace,
)
from rosnav_rl.spaces.observation_space.spaces.meta.basic_meta_spaces import EpisodeStepSpace
from rosnav_rl.spaces.observation_space.spaces.perception.laser.laser_spaces import MultiLaserFusionSpace

NUM_BEAMS = 541
FIELD_OF_VIEW = 1.5 * np.pi


@pytest.fixture
def parameters():
    return AgentParameters(
        laser_num_beams=NUM_BEAMS,
        laser_angle_min=-FIELD_OF_VIEW / 2,
        laser_angle_increment=FIELD_OF_VIEW / (NUM_BEAMS - 1),
        laser_max_range=3.5,
        max_linear_vel=0.5,
        max_angular_vel=4.0,
        robot_radius=0.2,
        safety_distance=0.5,
        max_steps=600,
        normalize=False,
    )


@pytest.fixture
def kwargs(parameters):
    return parameters.observation_kwargs()


def action(linear, angular):
    return np.array([linear, 0.0, angular])


def beam_at(degrees):
    return NUM_BEAMS // 2 + int(round(degrees * 2))


def test_observation_kwargs_carry_the_robot_and_episode_fields(kwargs):
    assert (kwargs["robot_radius"], kwargs["safety_distance"], kwargs["max_steps"]) == (0.2, 0.5, 600)
    assert "goal_radius" not in kwargs


def test_safety_context_judges_surface_clearance_against_the_safety_distance(kwargs):
    space = SafetyContextSpace(**kwargs)

    result = space.encode_observation(np.array([0.6, 3.0]), action(0.0, 0.0))

    assert result[0] == pytest.approx((0.5 - 0.4) / (0.5 - 0.3))


def test_safety_context_collides_when_the_surface_touches(kwargs):
    space = SafetyContextSpace(**kwargs)

    assert space.encode_observation(np.array([0.25]), action(0.0, 0.0))[0] == 1.0


def test_spatial_awareness_sectors_follow_the_scan_geometry(kwargs):
    space = SpatialAwarenessSpace(**kwargs)

    front = space._get_sector_indices(NUM_BEAMS, -30, 30)
    back = space._get_sector_indices(NUM_BEAMS, 150, 210)

    np.testing.assert_array_equal(front, np.arange(beam_at(-30), beam_at(30) + 1))
    assert len(back) == 0


@pytest.mark.parametrize(("degrees", "direction"), [(0, 0), (90, 1), (-90, 2)])
def test_spatial_awareness_clearance_lands_in_the_sector_of_the_beam_angle(kwargs, degrees, direction):
    space = SpatialAwarenessSpace(**kwargs)
    scan = np.full(NUM_BEAMS, 5.0)
    scan[beam_at(degrees)] = 0.25

    result = space.encode_observation(scan)

    expected = np.ones(4)
    expected[direction] = 0.5
    np.testing.assert_allclose(result[:4], expected)


def test_spatial_awareness_reads_nothing_beyond_the_laser_range(kwargs):
    space = SpatialAwarenessSpace(**kwargs)

    np.testing.assert_allclose(space.encode_observation(np.full(NUM_BEAMS, 5.0))[:4], np.ones(4))


def test_spatial_awareness_without_an_increment_spreads_the_beams_over_a_full_circle():
    space = SpatialAwarenessSpace(laser_angle_min=-np.pi)

    np.testing.assert_array_equal(space._get_sector_indices(360, -30, 30), np.arange(150, 211))


def test_environment_context_and_obstacle_proximity_normalize_by_the_laser_range(kwargs):
    scan = np.array([1.75, 5.0])

    proximity = ObstacleProximitySpace(**kwargs).encode_observation(scan)
    context = EnvironmentContextSpace(density_sectors=2, **kwargs).encode_observation(scan)

    assert proximity[-2] == pytest.approx(0.5)
    assert context[-2] == pytest.approx(0.5)
    assert context[-1] == pytest.approx(0.5)


def test_episode_step_space_is_bounded_by_the_episode_cap(kwargs):
    space = EpisodeStepSpace(**kwargs)

    assert space.space.high[0] == 600


def test_mission_time_is_normalized_by_the_episode_cap(kwargs):
    space = MissionContextSpace(**kwargs)

    for _ in range(3):
        result = space.encode_observation(np.array([5.0, 0.0]))

    assert result[1] == pytest.approx(2 / 600)


def test_motion_state_normalizes_each_velocity_by_its_own_limit(kwargs):
    space = MotionStateSpace(**kwargs)

    result = space.encode_observation(action(0.5, 4.0))

    np.testing.assert_allclose(result[:2], [np.tanh(np.sqrt(2.0)), 0.25], atol=1e-6)


@pytest.mark.parametrize("space_class", [KinematicStateSpace, TrajectoryStateSpace])
def test_kinematic_spaces_normalize_each_velocity_by_its_own_limit(kwargs, space_class):
    space = space_class(**kwargs)

    result = space.encode_observation(np.zeros(3), action(0.25, 2.0))

    np.testing.assert_allclose(result[4:6], [0.5, 0.5])


def test_trajectory_speed_consistency_uses_limit_normalized_velocities(kwargs):
    space = TrajectoryStateSpace(**kwargs)

    space.encode_observation(np.zeros(3), action(0.5, 0.0))
    result = space.encode_observation(np.zeros(3), action(0.0, 4.0))

    assert result[8] == pytest.approx(1.0)


def test_performance_energy_normalizes_each_velocity_by_its_own_limit(kwargs):
    space = PerformanceContextSpace(**kwargs)

    linear = space.encode_observation(np.array([5.0, 0.0]), action(0.25, 0.0))[1]
    space.reset()
    angular = space.encode_observation(np.array([5.0, 0.0]), action(0.0, 2.0))[1]

    assert linear == pytest.approx(0.5)
    assert angular == pytest.approx(0.5)


def test_multi_laser_fusion_follows_the_laser_geometry(kwargs):
    space = MultiLaserFusionSpace(**kwargs)

    result = space.encode_observation(np.full(NUM_BEAMS, 1.75))

    assert space.space.shape == (NUM_BEAMS,)
    np.testing.assert_allclose(result, 0.5)


def test_robust_odometry_reads_the_measured_twist(kwargs):
    space = RobustOdometrySpace(**kwargs)

    np.testing.assert_allclose(space.encode_observation(np.array([0.25, 0.0, 2.0])), [0.5, 0.5])


def test_localization_combined_reads_the_measured_twist(kwargs):
    space = LocalizationCombinedSpace(**kwargs)

    result = space.encode_observation(np.zeros(3), np.array([0.25, 0.0, 2.0]))

    np.testing.assert_allclose(result[4:], [0.5, 0.5])


@pytest.mark.parametrize("space_class", [RobustOdometrySpace, LocalizationCombinedSpace])
def test_bundled_observations_provide_the_measured_twist(space_class):
    config = loads_yaml((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())

    assert "robot_twist" in space_class.requires
    assert set(space_class.requires) <= set(config["datasources"]) | set(config["aliases"])
