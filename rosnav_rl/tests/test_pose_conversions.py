"""Quaternion, planar pose and goal geometry helpers on known orientations."""

import numpy as np
import pytest
from geometry_msgs.msg import Point, Pose, Pose2D, PoseWithCovariance, Quaternion
from tf_transformations import quaternion_from_euler

from rosnav_rl.observations.utils.pose import (
    euler_from_quaternion,
    false_params,
    get_goal_pose_in_robot_frame,
    pose3d_to_pose2d,
    pose_with_covariance_to_pose2d,
)


@pytest.mark.parametrize(
    ("roll", "pitch", "yaw"),
    [(0.0, 0.0, 0.0), (0.0, 0.0, 2.5), (0.3, -0.4, -1.2), (-2.0, 0.2, 3.0)],
)
def test_euler_round_trips_through_a_quaternion(roll, pitch, yaw):
    angles = euler_from_quaternion(quaternion_from_euler(roll, pitch, yaw))

    np.testing.assert_allclose(angles, [roll, pitch, yaw], atol=1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_euler_pitch_saturates_at_the_gimbal_lock(sign):
    half = np.sqrt(0.5) * (1.0 + 1e-9)

    _, pitch, _ = euler_from_quaternion((0.0, sign * half, 0.0, half))

    assert pitch == sign * np.pi / 2


def test_pose3d_to_pose2d_keeps_position_and_yaw():
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -0.7)

    pose2d = pose3d_to_pose2d(
        Pose(position=Point(x=1.5, y=-4.0, z=9.0), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))
    )

    assert isinstance(pose2d, Pose2D)
    assert (pose2d.x, pose2d.y) == (1.5, -4.0)
    assert pose2d.theta == pytest.approx(-0.7)


def test_pose_with_covariance_uses_the_inner_pose():
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, 1.1)
    inner = Pose(position=Point(x=2.0, y=3.0), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))

    pose2d = pose_with_covariance_to_pose2d(PoseWithCovariance(pose=inner))

    assert (pose2d.x, pose2d.y, pose2d.theta) == (2.0, 3.0, pytest.approx(1.1))


def test_goal_distance_in_robot_frame_is_euclidean():
    rho, _ = get_goal_pose_in_robot_frame(Pose2D(x=4.0, y=6.0, theta=0.0), Pose2D(x=1.0, y=2.0, theta=1.0))

    assert rho == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("goal", "robot", "bearing"),
    [
        ((1.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        ((0.0, 2.0), (0.0, 0.0, 0.0), np.pi / 2),
        ((0.0, 2.0), (0.0, 0.0, np.pi), -np.pi / 2),
    ],
)
def test_goal_bearing_is_relative_to_the_robot_heading(goal, robot, bearing):
    _, theta = get_goal_pose_in_robot_frame(
        Pose2D(x=goal[0], y=goal[1]), Pose2D(x=robot[0], y=robot[1], theta=robot[2])
    )

    assert theta == pytest.approx(bearing)


def test_false_params_lists_falsy_keyword_names_in_order():
    assert false_params(a=0, b=1, c="", d=None, e=[0], f=[]) == ["a", "c", "d", "f"]
