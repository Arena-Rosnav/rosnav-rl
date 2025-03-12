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
    """A collector unit for processing laser scan messages from ROS topics.
    
    This class inherits from ObservationCollectorUnit and is responsible for 
    collecting and preprocessing laser scan data from ROS sensors. It handles 
    laser readings and converts them into a format suitable for use in 
    reinforcement learning environments.
    
    Attributes:
        name (ClassVar[str]): The name identifier for this collector ("laser_scan").
        topic (ClassVar[str]): The ROS topic to subscribe to for laser scan data ("scan").
        up_to_date_required (ClassVar[bool]): Flag indicating whether the data must be current.
        msg_data_class (ClassVar[Type]): The message type this collector processes (sensor_msgs.LaserScan).
        applicable_simulators (List[Simulator]): List of simulators where this collector can be used
            (Flatland, Unity, and Gazebo).
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
        """Preprocess raw LaserScan message into numpy array.
        
        This method takes a LaserScan message, filters out NaN values by replacing them with the maximum range,
        and returns a numpy array representation of the laser scan data.
        
        Args:
            msg (sensor_msgs.LaserScan): The raw LaserScan message from ROS.
        
        Returns:
            np.ndarray: Processed laser scan data as a numpy array. Empty array if input ranges are empty.
        """
        
        super().preprocess(msg)
        if len(msg.ranges) == 0:
            return np.array([])

        laser = np.array(msg.ranges, np.float32)
        laser[np.isnan(laser)] = msg.range_max
        return laser


class FullRangeLaserCollector(LaserCollector):
    """A collector for full range laser scan data.
    
    This collector is responsible for handling the processing of full range laser scan
    data received from the 'full_scan' topic.
    
    Attributes:
        name (ClassVar[str]): The name identifier for this collector.
        topic (ClassVar[str]): The ROS topic name from which to collect laser scan data.
    """
    name: ClassVar[str] = "full_range_laser_scan"
    topic: ClassVar[str] = "full_scan"


class PoseCollector(
    ObservationCollectorUnit[geometry_msgs.PoseStamped, np.ndarray], ABC
):
    """Base class for collecting pose observations from ROS topics.
    
    This abstract class serves as the foundation for observation collectors that process pose data
    from ROS messages. It inherits from ObservationCollectorUnit and specifies that it will convert
    geometry_msgs.PoseStamped messages to numpy arrays.
    
    Attributes:
        name (ClassVar[str]): The name identifier for the collector (to be defined in subclasses).
        topic (ClassVar[str]): The ROS topic to subscribe to (to be defined in subclasses).
        msg_data_class (ClassVar[Type[geometry_msgs.PoseStamped]]): The message type to expect (PoseStamped).
        applicable_simulators (List[Simulator]): List of simulators where this collector can be used.
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
    """Collector for robot pose information from the odometry topic.
    
    This collector subscribes to the robot's odometry messages and extracts 
    the 2D pose (x, y, theta) of the robot. It is designed to provide the robot's
    position and orientation as part of the observation space for reinforcement
    learning algorithms.
    
    Attributes:
        name (str): Name identifier for this collector, set to "robot_pose".
        topic (str): ROS topic to subscribe to, set to "odom".
        up_to_date_required (bool): Flag indicating whether the newest data is required, set to True.
        msg_data_class (Type): Message type class, set to nav_msgs.Odometry.
    
    Note:
        This collector inherits from both PoseCollector and ObservationCollectorUnit
        and specializes in processing odometry messages into 2D pose representations.
    """
    name: ClassVar[str] = "robot_pose"
    topic: ClassVar[str] = "odom"
    up_to_date_required: ClassVar[bool] = True
    msg_data_class: ClassVar[Type[nav_msgs.Odometry]] = nav_msgs.Odometry

    def preprocess(self, msg: nav_msgs.Odometry) -> np.ndarray:
        """Preprocess the Odometry message to extract 2D pose information.
        
        This method extracts the 2D pose (x, y, theta) from the 3D pose contained in
        the Odometry message.
        
        Args:
            msg (nav_msgs.Odometry): The Odometry message to preprocess.
        
        Returns:
            np.ndarray: A numpy array containing the 2D pose information (x, y, theta)
                       with dtype Pose2DType.
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
        """Preprocesses a ROS PoseStamped message by converting it to a 2D pose array.
        
        This method converts a 3D pose from a PoseStamped message to a 2D representation
        containing x, y positions and theta orientation.
        
        Args:
            msg (geometry_msgs.PoseStamped): The ROS PoseStamped message to preprocess.
        
        Returns:
            np.ndarray: A numpy array of type Pose2DType containing (x, y, theta).
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
