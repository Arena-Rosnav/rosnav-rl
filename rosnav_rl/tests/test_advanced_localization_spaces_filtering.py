"""Odometry filtering, pose stabilization and their combination on known inputs."""

import numpy as np
import pytest
from nav_msgs.msg import Odometry

from rosnav_rl.observations.data_sources.collectors import OdometryCollector
from rosnav_rl.observations.utils.pose import Pose2DType
from rosnav_rl.spaces.observation_space.spaces.localization.advanced_localization_spaces import (
    LocalizationCombinedSpace,
    PoseStabilizedSpace,
    RobustOdometrySpace,
)


def pose(x, y, yaw):
    return np.array([(x, y, yaw)], dtype=Pose2DType)[0]


def odometry_pose(x, y, yaw):
    collector = OdometryCollector("odom", topic="odom")
    odom = Odometry()
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.orientation.z = np.sin(yaw / 2)
    odom.pose.pose.orientation.w = np.cos(yaw / 2)
    collector.update(odom)
    return collector.get_observation()


def action(linear, angular):
    return np.array([linear, 0.0, angular])


@pytest.mark.parametrize(("include_acceleration", "dims"), [(False, 2), (True, 4)])
def test_odometry_space_dims(include_acceleration, dims):
    space = RobustOdometrySpace(include_acceleration=include_acceleration).space

    assert space.shape == (dims,)
    assert (space.low.min(), space.high.max()) == (-1.0, 1.0)


def test_odometry_filters_velocities_and_reports_filtered_acceleration():
    space = RobustOdometrySpace(
        max_linear_vel=2.0, max_angular_vel=1.0, velocity_filter_alpha=0.5, include_acceleration=True
    )

    first = space.encode_observation(action(1.0, 0.5))
    second = space.encode_observation(action(3.0, -0.5))

    np.testing.assert_allclose(first, [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(second, [1.0, 0.0, 0.25, -0.25])


def test_odometry_clips_velocity_and_acceleration_to_the_unit_box():
    space = RobustOdometrySpace(
        max_linear_vel=2.0, max_angular_vel=1.0, velocity_filter_alpha=0.0, include_acceleration=True
    )
    space.encode_observation(action(0.0, 0.0))

    np.testing.assert_allclose(space.encode_observation(action(-10.0, 10.0)), [-1.0, 1.0, -1.0, 1.0])


def test_odometry_without_acceleration_returns_only_velocities():
    space = RobustOdometrySpace(max_linear_vel=2.0, max_angular_vel=2.0)

    np.testing.assert_allclose(space.encode_observation(action(1.0, -1.0)), [0.5, -0.5])


def test_odometry_reset_restarts_the_filter():
    space = RobustOdometrySpace(
        max_linear_vel=2.0, max_angular_vel=2.0, velocity_filter_alpha=0.9, include_acceleration=True
    )
    space.encode_observation(action(2.0, 2.0))

    space.reset()

    np.testing.assert_allclose(space.encode_observation(action(0.0, 0.0)), [0.0, 0.0, 0.0, 0.0])
    assert space.last_velocities == (0.0, 0.0)


def test_pose_space_squashes_position_and_encodes_heading_as_cos_sin():
    space = PoseStabilizedSpace(position_scale=10.0)

    np.testing.assert_allclose(
        space.encode_observation(pose(5.0, -5.0, np.pi / 2)), [np.tanh(0.5), np.tanh(-0.5), 0.0, 1.0], atol=1e-6
    )
    assert space.space.shape == (4,)


def test_pose_space_relative_coords_wrap_heading_difference():
    space = PoseStabilizedSpace(position_scale=10.0, use_relative_coords=True)

    origin = space.encode_observation(pose(2.0, 3.0, 3.0))
    moved = space.encode_observation(pose(4.0, 3.0, -3.0))

    wrapped = -6.0 + 2 * np.pi
    np.testing.assert_allclose(origin, [0.0, 0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(moved, [np.tanh(0.2), 0.0, np.cos(wrapped), np.sin(wrapped)], atol=1e-6)


def test_pose_space_reset_picks_a_new_reference():
    space = PoseStabilizedSpace(use_relative_coords=True)
    space.encode_observation(pose(2.0, 3.0, 1.0))

    space.reset()

    np.testing.assert_allclose(space.encode_observation(pose(-7.0, 1.0, -2.0)), [0.0, 0.0, 1.0, 0.0], atol=1e-6)
    assert space.reference_pose == {"x": -7.0, "y": 1.0, "yaw": -2.0}


def test_pose_space_confidence_channel_is_fixed():
    space = PoseStabilizedSpace(include_confidence=True)

    result = space.encode_observation(pose(0.0, 0.0, 0.0))

    assert space.space.shape == (5,)
    assert result[4] == pytest.approx(0.9)


def test_combined_space_concatenates_pose_and_odometry():
    space = LocalizationCombinedSpace(position_scale=10.0, max_linear_vel=2.0, max_angular_vel=2.0)

    result = space.encode_observation(pose(5.0, -5.0, np.pi / 2), action(1.0, 0.5))

    assert space.space.shape == (6,)
    np.testing.assert_allclose(result, [np.tanh(0.5), np.tanh(-0.5), 0.0, 1.0, 0.5, 0.25], atol=1e-6)


def test_combined_space_filters_odometry_across_steps():
    space = LocalizationCombinedSpace(max_linear_vel=2.0, velocity_filter_alpha=0.5)
    space.encode_observation(pose(0.0, 0.0, 0.0), action(2.0, 0.0))

    assert space.encode_observation(pose(0.0, 0.0, 0.0), action(0.0, 0.0))[4] == pytest.approx(0.5)


def test_combined_space_reset_restarts_the_odometry_filter():
    space = LocalizationCombinedSpace(max_linear_vel=2.0, velocity_filter_alpha=0.8)
    space.encode_observation(pose(0.0, 0.0, 0.0), action(2.0, 0.0))

    space.reset()

    assert space.encode_observation(pose(0.0, 0.0, 0.0), action(0.0, 0.0))[4] == 0.0


def test_pose_space_accepts_the_pose_produced_by_the_odometry_collector():
    expected = PoseStabilizedSpace().encode_observation(pose(2.0, -1.0, 0.5))

    result = PoseStabilizedSpace().encode_observation(odometry_pose(2.0, -1.0, 0.5))

    np.testing.assert_allclose(result, expected, atol=1e-6)
