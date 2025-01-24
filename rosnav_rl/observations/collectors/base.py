"""
OBSERVATIONS THAT ACTIVELY LISTEN TO TOPICS AND PREPROCESS MESSAGES
"""

from abc import ABC
from typing import ClassVar, Type, List

import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import sensor_msgs.msg as sensor_msgs

from ..utils.pose import Pose2DType, TwistType, pose3d_to_pose2d
from .base_collector import ObservationCollectorUnit

from rl_utils.utils.constants import Simulator

__all__ = [
    "LaserCollector",
    "FullRangeLaserCollector",
    "PoseCollector",
    "RobotPoseCollector",
    "GoalCollector",
    "SubgoalCollector",
    "LastActionCollector",
    "GlobalPlanCollector",
]


class LaserCollector(ObservationCollectorUnit[sensor_msgs.LaserScan, np.ndarray]):
    """
    A class that collects laser scan observations.

    Attributes:
        name (str): The name of the collector.
        topic (str): The topic to subscribe to for laser scan messages.
        msg_data_class (Type[sensor_msgs.LaserScan]): The message data class for laser scan messages.
        data_class (Type[np.ndarray]): The data class for storing the collected laser scan observations.
        up_to_date_required (bool): Specifies whether value should be kept up to date, i.e. a new message is required for every step.
    """

    name: ClassVar[str] = "laser_scan"
    topic: ClassVar[str] = "scan"
    up_to_date_required: ClassVar[bool] = True
    msg_data_class: ClassVar[Type[sensor_msgs.LaserScan]] = sensor_msgs.LaserScan
    applicable_simulators: List[Simulator] = [
        Simulator.FLATLAND,
        Simulator.UNITY,
        Simulator.GAZEBO,
    ]

    def preprocess(self, msg: sensor_msgs.LaserScan) -> np.ndarray:
        """
        Preprocesses the received laser scan message.

        Args:
            msg (sensor_msgs.LaserScan): The laser scan message to preprocess.

        Returns:
            np.ndarray: The preprocessed laser scan data as a NumPy array.
        """
        super().preprocess(msg)
        if len(msg.ranges) == 0:
            return np.array([])

        laser = np.array(msg.ranges, np.float32)
        laser[np.isnan(laser)] = msg.range_max
        return laser


class FullRangeLaserCollector(LaserCollector):
    """
    A class representing a collector for full range laser scans. (i.e., 360-degree laser scans.)

    Attributes:
        name (str): The name of the collector.
        topic (str): The topic to subscribe to for laser scan messages.
    """

    name: ClassVar[str] = "full_range_laser_scan"
    topic: ClassVar[str] = "full_scan"


class PoseCollector(
    ObservationCollectorUnit[geometry_msgs.PoseStamped, np.ndarray], ABC
):
    """
    PoseCollector is a class that collects pose observations in the form of geometry_msgs.PoseStamped
    and converts them into numpy arrays.

    Attributes:
        name (str): The name of the collector.
        topic (str): The ROS topic from which the pose messages are collected.
    """

    name: ClassVar[str]
    topic: ClassVar[str]
    msg_data_class: ClassVar[Type[geometry_msgs.PoseStamped]] = (
        geometry_msgs.PoseStamped
    )
    applicable_simulators: List[Simulator] = [
        Simulator.FLATLAND,
        Simulator.UNITY,
        Simulator.GAZEBO,
    ]


class RobotPoseCollector(
    PoseCollector,
    ObservationCollectorUnit[nav_msgs.Odometry, np.ndarray],
):
    """
    A class for collecting robot pose observations.

    This class inherits from the `PoseCollector` class and provides a specific implementation
    for collecting robot pose observations.

    Attributes:
        name (str): The name of the collector.
        topic (str): The ROS topic to subscribe to for pose messages.
        msg_data_class (Type[nav_msgs.Odometry]): The ROS message data class for pose messages.
        data_class (Type[np.ndarray]): The data class for storing the collected pose observations.
        up_to_date_required (bool): Specifies whether value should be kept up to date, i.e. a new message is required for every step.
    """

    name: ClassVar[str] = "robot_pose"
    topic: ClassVar[str] = "odom"
    up_to_date_required: ClassVar[bool] = True
    msg_data_class: ClassVar[Type[nav_msgs.Odometry]] = nav_msgs.Odometry

    def preprocess(self, msg: nav_msgs.Odometry) -> np.ndarray:
        """
        Preprocesses the received pose message.

        This method converts the received pose message into a 2D pose representation
        (x, y, theta) and returns it as a numpy array.

        Args:
            msg (nav_msgs.Odometry): The received pose message.

        Returns:
            np.ndarray: The preprocessed pose observation as a numpy array.
        """
        super().preprocess(msg)
        pose3d: geometry_msgs.PoseWithCovariance = (
            msg.pose.pose
        )  # Access the inner pose object
        pose2d: geometry_msgs.Pose2D = pose3d_to_pose2d(pose3d)
        return np.array((pose2d.x, pose2d.y, pose2d.theta), dtype=Pose2DType)


class GoalCollector(
    PoseCollector, ObservationCollectorUnit[nav_msgs.Odometry, np.ndarray]
):
    """
    A collector class for collecting goal observations.

    This collector collects goal observations from the "move_base_simple/goal" topic
    and preprocesses them into a numpy array representation.

    Attributes:
        name (str): The name of the collector.
        topic (str): The ROS topic to subscribe to for goal messages.
        msg_data_class (Type[geometry_msgs.PoseStamped]): The ROS message data class for goal messages.
        data_class (Type[np.ndarray]): The numpy array data class for the preprocessed goal observations.
    """

    name: ClassVar[str] = "goal"
    topic: ClassVar[str] = "move_base_simple/goal"

    def preprocess(self, msg: geometry_msgs.PoseStamped) -> np.ndarray:
        """
        Preprocesses the goal message into a numpy array representation.

        Args:
            msg (geometry_msgs.PoseStamped): The goal message to preprocess.

        Returns:
            np.ndarray: The preprocessed goal observation as a numpy array.
        """
        super().preprocess(msg)
        pose3d: geometry_msgs.Pose = msg.pose
        pose2d: geometry_msgs.Pose2D = pose3d_to_pose2d(pose3d)
        return np.array((pose2d.x, pose2d.y, pose2d.theta), dtype=Pose2DType)


class SubgoalCollector(GoalCollector):
    """
    A class for collecting subgoals in the environment. Subgoals are similar to goals but are intermediate goals
    that the robot must reach to achieve the main goal. They are published by the intermediate planner.

    Attributes:
        name (str): The name of the subgoal collector.
        topic (str): The topic to subscribe to for subgoal information.
    """

    name: ClassVar[str] = "subgoal"
    topic: ClassVar[str] = "subgoal"


class LastActionCollector(ObservationCollectorUnit[geometry_msgs.Twist, np.ndarray]):
    """
    Collects the last action taken by the agent as an observation.

    This collector subscribes to the "cmd_vel" topic and preprocesses the received
    `geometry_msgs.Twist` message to extract the linear and angular components of
    the action. The preprocessed action is returned as a NumPy array.

    Attributes:
        name (str): The name of the collector.
        topic (str): The ROS topic to subscribe to for action messages.
        msg_data_class (Type[geometry_msgs.Twist]): The ROS message data class for action messages.
        data_class (Type[np.ndarray]): The data class for the preprocessed action.
        up_to_date_required (bool): Specifies whether value should be kept up to date, i.e. a new message is required for every step.
    """

    name: ClassVar[str] = "last_action"
    topic: ClassVar[str] = "cmd_vel"
    up_to_date_required: bool = False
    msg_data_class: ClassVar[Type[geometry_msgs.Twist]] = geometry_msgs.Twist
    applicable_simulators: List[Simulator] = [
        Simulator.FLATLAND,
        Simulator.UNITY,
        Simulator.GAZEBO,
    ]

    def preprocess(self, msg: geometry_msgs.Twist) -> np.ndarray:
        """
        Preprocesses the received action message to extract the linear and angular components.

        Args:
            msg (geometry_msgs.Twist): The action message received from the "cmd_vel" topic.

        Returns:
            np.ndarray: The preprocessed action as a NumPy array.

        """
        super().preprocess(msg)
        return np.array((msg.linear.x, msg.linear.y, msg.angular.z))


class GlobalPlanCollector(ObservationCollectorUnit[nav_msgs.Path, np.ndarray]):
    """
    Collects the global plan as an observation.

    This collector extracts the global plan from the `global_plan` topic and preprocesses it into a numpy array.

    Attributes:
        name (str): The name of the collector.
        topic (str): The topic to subscribe to for the global plan.
        msg_data_class (Type[nav_msgs.Path]): The ROS message data class for the global plan.
        data_class (Type[np.ndarray]): The data class for the preprocessed global plan.

    Methods:
        preprocess(msg: nav_msgs.Path) -> np.ndarray:
            Preprocesses the global plan message into a numpy array.

    """

    name: ClassVar[str] = "global_plan"
    topic: ClassVar[str] = "global_plan"
    msg_data_class: ClassVar[Type[nav_msgs.Path]] = nav_msgs.Path
    applicable_simulators: List[Simulator] = [
        Simulator.FLATLAND,
        Simulator.UNITY,
        Simulator.GAZEBO,
    ]

    def preprocess(self, msg: nav_msgs.Path) -> np.ndarray:
        """
        Preprocesses the global plan message into a numpy array.

        Args:
            msg (nav_msgs.Path): The global plan message.

        Returns:
            np.ndarray: The preprocessed global plan as a numpy array.

        """
        super().preprocess(msg)
        return np.array(
            list(
                map(
                    lambda p: [p.pose.position.x, p.pose.position.y],
                    msg.poses,
                )
            )
        )
