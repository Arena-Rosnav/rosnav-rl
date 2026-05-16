from typing import Tuple

import numpy as np
from geometry_msgs.msg import Pose2D

DistanceToGoal = float
AngleToGoal = float
Pose2DType = np.dtype([("x", np.float32), ("y", np.float32), ("yaw", np.float32)])
TwistType = np.dtype(
    [("linear_x", np.float32), ("linear_y", np.float32), ("angular", np.float32)]
)


def euler_from_quaternion(quaternion):
    """
    Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)
    """
    x, y, z, w = quaternion

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def get_goal_pose_in_robot_frame(
    goal_pos: Pose2D, robot_pos: Pose2D
) -> Tuple[DistanceToGoal, AngleToGoal]:
    y_relative = goal_pos.y - robot_pos.y
    x_relative = goal_pos.x - robot_pos.x
    rho = (x_relative**2 + y_relative**2) ** 0.5
    theta = (np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi) % (
        2 * np.pi
    ) - np.pi
    return rho, theta


def pose3d_to_pose2d(pose3d) -> Pose2D:
    pose2d = Pose2D()
    pose2d.x = pose3d.position.x
    pose2d.y = pose3d.position.y
    quaternion = (
        pose3d.orientation.x,
        pose3d.orientation.y,
        pose3d.orientation.z,
        pose3d.orientation.w,
    )
    euler = euler_from_quaternion(quaternion)
    yaw = euler[2]
    pose2d.theta = yaw
    return pose2d


def pose_with_covariance_to_pose2d(pose_with_covariance) -> Pose2D:
    pose3d = pose_with_covariance.pose
    return pose3d_to_pose2d(pose3d)


def false_params(**kwargs):
    false_params = []
    for key, val in kwargs.items():
        if not val:
            false_params.append(key)
    return false_params
