"""
This module contains all Collector classes.

Collectors are data sources that subscribe to external sources (e.g., ROS topics)
and preprocess the raw data into a clean, usable format.
"""

import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import sensor_msgs.msg as sensor_msgs
import nav2_msgs.msg as nav2_msgs
import people_msgs.msg as people_msgs
import arena_people_msgs.msg as arena_people_msgs

from ..utils.pose import Pose2DType, pose3d_to_pose2d
from .base import Collector
from ..utils.types import (
    Pose2D,
    LidarRanges,
    RobotVelocity,
    NavigationPath,
    ImageData,
    PedestrianDetections,
    ArenaPedestrianDetections,
    SafetyStatus,
)


class LaserScanCollector(Collector[sensor_msgs.LaserScan, LidarRanges]):
    """
    Collects and preprocesses laser scan data from a ROS topic.
    """

    def _preprocess(self, msg: sensor_msgs.LaserScan) -> LidarRanges:
        if len(msg.ranges) == 0:
            return np.array([])
        laser = np.array(msg.ranges, np.float32)
        laser[np.isnan(laser)] = msg.range_max
        return laser


class OdometryCollector(Collector[nav_msgs.Odometry, Pose2D]):
    """
    Collects and preprocesses robot pose data from the odometry topic.
    """

    def _preprocess(self, msg: nav_msgs.Odometry) -> Pose2D:
        pose3d: geometry_msgs.PoseWithCovariance = msg.pose.pose
        pose2d: geometry_msgs.Pose2D = pose3d_to_pose2d(pose3d)
        return np.array((pose2d.x, pose2d.y, pose2d.theta), dtype=Pose2DType)


class PoseStampedCollector(Collector[geometry_msgs.PoseStamped, Pose2D]):
    """
    Collects and preprocesses the goal pose.
    """

    def _preprocess(self, msg: geometry_msgs.PoseStamped) -> Pose2D:
        pose3d: geometry_msgs.Pose = msg.pose
        pose2d: geometry_msgs.Pose2D = pose3d_to_pose2d(pose3d)
        return np.array((pose2d.x, pose2d.y, pose2d.theta), dtype=Pose2DType)


class TwistCollector(Collector[geometry_msgs.Twist, RobotVelocity]):
    """
    Collects the last action sent to the robot.
    """

    def _preprocess(self, msg: geometry_msgs.Twist) -> RobotVelocity:
        return np.array((msg.linear.x, msg.linear.y, msg.angular.z))


class CollisionMonitorStateCollector(
    Collector[nav2_msgs.CollisionMonitorState, SafetyStatus]
):
    """
    Collects the collision monitor state.
    """

    def __init__(self, name: str, topic: str, state_key: str, **kwargs):
        self.state_key = state_key
        super().__init__(name, topic, **kwargs)

    def _preprocess(self, msg: nav2_msgs.CollisionMonitorState) -> SafetyStatus:
        # polygon_name is a list[str]; check membership rather than equality
        return self.state_key in msg.polygon_name


class PathCollector(Collector[nav_msgs.Path, NavigationPath]):
    """
    Collects the path from the navigation stack.
    """

    def _preprocess(self, msg: nav_msgs.Path) -> NavigationPath:
        if not msg.poses:
            return np.empty((0, 2), dtype=np.float32)

        plan = np.empty((len(msg.poses), 2), dtype=np.float32)
        for i, stamped_pose in enumerate(msg.poses):
            plan[i, 0] = stamped_pose.pose.position.x
            plan[i, 1] = stamped_pose.pose.position.y
        return plan


class ImageColorCollector(Collector[sensor_msgs.Image, ImageData]):
    """A class that collects color images as observations."""

    def _preprocess(self, msg: sensor_msgs.Image) -> ImageData:
        """
        Preprocesses the image message and returns the processed image.

        Args:
            msg (sensor_msgs.Image): The image message to preprocess.

        Returns:
            ImageData: The processed image as HWC or CHW format.
        """
        if not msg.data:
            return np.array([])

        # Convert image data to numpy array
        if msg.encoding in ["rgb8", "bgr8"]:
            dtype = np.uint8
            channels = 3
        elif msg.encoding in ["rgba8", "bgra8"]:
            dtype = np.uint8
            channels = 4
        elif msg.encoding == "mono8":
            dtype = np.uint8
            channels = 1
        elif msg.encoding in ["32FC1", "mono32"]:
            dtype = np.float32
            channels = 1
        elif msg.encoding in ["32FC3"]:
            dtype = np.float32
            channels = 3
        else:
            # Default fallback
            dtype = np.uint8
            channels = 3

        # Reshape image data
        image_array = np.frombuffer(msg.data, dtype=dtype)

        if channels > 1:
            image_array = image_array.reshape((msg.height, msg.width, channels))
            # Convert to CHW format (channels first)
            return image_array.transpose((2, 0, 1))
        else:
            return image_array.reshape((msg.height, msg.width))


class PeopleCollector(Collector[people_msgs.People, PedestrianDetections]):
    """A class that collects information about people in the environment."""

    def _preprocess(self, msg: people_msgs.People) -> people_msgs.People:
        return msg


class ArenaPedestrianCollector(
    Collector[arena_people_msgs.Pedestrians, ArenaPedestrianDetections]
):
    """A class that collects information about pedestrians from Arena simulator.

    Processes arena_people_msgs/Pedestrians messages which include:
    - Header with timestamp
    - Array of Pedestrian objects with:
      - name: Pedestrian identifier
      - id: Unique ID
      - pose: Full 3D pose (position + orientation)
      - twist: Linear and angular velocities
      - animation_state: Behavioral state (IDLE, WALKING, RUNNING, etc.)
    """

    def _preprocess(
        self, msg: arena_people_msgs.Pedestrians
    ) -> arena_people_msgs.Pedestrians:
        return msg
