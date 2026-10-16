"""Motion, kinematic and trajectory state vectors on constructed pose and action sequences."""

import numpy as np
import pytest
from nav_msgs.msg import Odometry

from rosnav_rl.observations.data_sources.collectors import OdometryCollector
from rosnav_rl.observations.utils.pose import Pose2DType
from rosnav_rl.spaces.observation_space.spaces.dynamics.advanced_dynamics_spaces import (
    KinematicStateSpace,
    MotionStateSpace,
    TrajectoryStateSpace,
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


def test_motion_state_magnitude_direction_stability_and_acceleration():
    space = MotionStateSpace(max_velocity=2.0, stability_window=3, include_acceleration=True)

    moving = space.encode_observation(action(0.6, 0.8))
    stopped = space.encode_observation(action(0.0, 0.0))

    np.testing.assert_allclose(moving, [np.tanh(0.5), np.arctan2(0.8, 0.6) / np.pi, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(stopped, [0.0, 0.0, np.exp(-0.25 / 4.0), np.tanh(0.5)], atol=1e-6)


def test_motion_state_stability_uses_only_the_last_window():
    space = MotionStateSpace(max_velocity=2.0, stability_window=3)

    for linear in (2.0, 0.0, 1.0, 1.0):
        result = space.encode_observation(action(linear, 0.0))

    assert space.velocity_history == [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    assert result[2] == pytest.approx(np.exp(-np.var([0.0, 1.0, 1.0]) / 4.0))


def test_motion_state_reverse_motion_points_backwards():
    space = MotionStateSpace()

    assert space.encode_observation(action(-1.0, 0.0))[1] == pytest.approx(1.0)


def test_kinematic_state_without_alignment_has_six_dims():
    space = KinematicStateSpace(max_velocity=2.0, position_scale=10.0, include_motion_alignment=False)

    result = space.encode_observation(pose(10.0, -5.0, np.pi), action(1.0, -3.0))

    assert space.space.shape == (6,)
    np.testing.assert_allclose(result, [np.tanh(1.0), np.tanh(-0.5), -1.0, 0.0, 0.5, -1.0], atol=1e-6)


def test_kinematic_state_alignment_is_cosine_of_velocity_direction():
    space = KinematicStateSpace(max_velocity=2.0)

    turning = space.encode_observation(pose(0.0, 0.0, 0.0), action(0.0, 1.0))
    still = space.encode_observation(pose(0.0, 0.0, 0.0), action(0.0, 0.0))

    assert turning[6] == pytest.approx(0.0, abs=1e-7)
    assert still[6] == 0.0


def test_trajectory_state_space_is_a_nine_dim_box():
    space = TrajectoryStateSpace().space

    assert space.shape == (9,)
    assert (space.low.min(), space.high.max()) == (-1.0, 1.0)


def test_trajectory_state_features_over_a_three_step_history():
    space = TrajectoryStateSpace(history_length=3, max_velocity=2.0, position_scale=5.0)
    behind = -np.pi + 0.1

    first = space.encode_observation(pose(0.0, 0.0, 0.0), action(1.0, 0.0))
    second = space.encode_observation(pose(3.0, 4.0, np.pi / 2), action(1.0, 0.0))
    third = space.encode_observation(pose(3.0, 4.0, behind), action(0.0, 0.0))
    fourth = space.encode_observation(pose(3.0, 4.0, behind), action(0.0, 0.0))

    np.testing.assert_allclose(first, [0, 0, 1, 0, 0.5, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(second, [np.tanh(0.6), np.tanh(0.8), 0, 1, 0.5, 0, np.tanh(1.0), 0, 1], atol=1e-6)
    np.testing.assert_allclose(third[6:], [np.tanh(1.0), 0.5 + 0.05 / np.pi, np.exp(-(2 / 9) / 4.0)], atol=1e-5)
    np.testing.assert_allclose(fourth[6:], [0.0, (np.pi / 2 + 0.1) / 2 / np.pi, np.exp(-(2 / 9) / 4.0)], atol=1e-5)
    assert len(space.pose_history) == 3


def test_trajectory_state_clips_velocities():
    space = TrajectoryStateSpace(max_velocity=1.0)

    np.testing.assert_allclose(space.encode_observation(pose(0.0, 0.0, 0.0), action(-3.0, 2.0))[4:6], [-1.0, 1.0])


def test_trajectory_state_reset_clears_history():
    space = TrajectoryStateSpace()
    space.encode_observation(pose(0.0, 0.0, 0.0), action(1.0, 0.0))
    space.encode_observation(pose(1.0, 0.0, 0.0), action(1.0, 0.0))

    space.reset()

    assert (space.pose_history, space.velocity_history) == ([], [])
    np.testing.assert_allclose(space.encode_observation(pose(2.0, 0.0, 0.0), action(1.0, 0.0))[6:], [0, 0, 0])


@pytest.mark.parametrize("space_class", [KinematicStateSpace, TrajectoryStateSpace])
def test_pose_spaces_accept_the_pose_produced_by_the_odometry_collector(space_class):
    expected = space_class().encode_observation(pose(2.0, -1.0, 0.5), action(1.0, 0.0))

    result = space_class().encode_observation(odometry_pose(2.0, -1.0, 0.5), action(1.0, 0.0))

    np.testing.assert_allclose(result, expected, atol=1e-6)
