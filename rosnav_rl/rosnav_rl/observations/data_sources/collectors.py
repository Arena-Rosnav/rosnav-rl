"""
This module contains all Collector classes.

Collectors are data sources that subscribe to external sources (e.g., ROS topics)
and preprocess the raw data into a clean, usable format.
"""

from typing import Any

import arena_people_msgs.msg as arena_people_msgs
import control_msgs.msg as control_msgs
import geometry_msgs.msg as geometry_msgs
import nav2_msgs.msg as nav2_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import people_msgs.msg as people_msgs
import rclpy
import ros_gz_interfaces.msg as ros_gz_interfaces
import sensor_msgs.msg as sensor_msgs
import trajectory_msgs.msg as trajectory_msgs

from ..utils.pose import Pose2DType, pose3d_to_pose2d
from ..utils.types import (
    ArenaPedestrianDetections,
    ArmActionTrajectory,
    ContactData,
    ContactPoint,
    GripperState,
    ImageData,
    JointStates,
    LidarRanges,
    NavigationPath,
    PedestrianDetections,
    Pose2D,
    RobotVelocity,
    SafetyStatus,
)
from .base import Collector


class LaserScanCollector(Collector[sensor_msgs.LaserScan, LidarRanges]):
    """
    Collects and preprocesses laser scan data from a ROS topic.
    """

    def _preprocess(self, msg: sensor_msgs.LaserScan) -> LidarRanges:
        if len(msg.ranges) == 0:
            return np.array([])
        laser = np.array(msg.ranges, np.float32)
        laser[np.isnan(laser)] = msg.range_max
        laser[np.isinf(laser)] = msg.range_max
        return laser


class OdometryCollector(Collector[nav_msgs.Odometry, Pose2D]):
    """
    Collects and preprocesses robot pose data from the odometry topic.
    """

    def _preprocess(self, msg: nav_msgs.Odometry) -> Pose2D:
        pose3d: geometry_msgs.Pose = msg.pose.pose
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


class OdometryTwistCollector(Collector[nav_msgs.Odometry, RobotVelocity]):
    """
    Collects the measured robot velocity in the base frame from the odometry topic.
    """

    def _preprocess(self, msg: nav_msgs.Odometry) -> RobotVelocity:
        twist = msg.twist.twist
        return np.array((twist.linear.x, twist.linear.y, twist.angular.z), dtype=np.float32)


class CollisionMonitorStateCollector(Collector[nav2_msgs.CollisionMonitorState, SafetyStatus]):
    """
    Collects the collision monitor state.
    """

    def __init__(self, name: str, topic: str, state_key: str, **kwargs: Any):
        self.state_key = state_key
        super().__init__(name, topic, **kwargs)

    def _preprocess(self, msg: nav2_msgs.CollisionMonitorState) -> SafetyStatus:
        return self.state_key == msg.polygon_name


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


class ArenaPedestrianCollector(Collector[arena_people_msgs.Pedestrians, ArenaPedestrianDetections]):
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

    def _preprocess(self, msg: arena_people_msgs.Pedestrians) -> arena_people_msgs.Pedestrians:
        return msg


class JointStateCollector(Collector[control_msgs.JointTrajectoryControllerState, JointStates]):
    """
    Collects and preprocesses joint state data from a ROS topic.
    """

    def _preprocess(self, msg: control_msgs.JointTrajectoryControllerState) -> JointStates:

        names = msg.joint_names
        positions = msg.feedback.positions

        velocities = msg.feedback.velocities if len(msg.feedback.velocities) == len(names) else [0.0] * len(names)

        efforts = msg.feedback.effort if len(msg.feedback.effort) == len(names) else [0.0] * len(names)

        return {
            name: {
                "position": float(pos),
                "velocity": float(vel),
                "effort": float(eff),
            }
            for name, pos, vel, eff in zip(names, positions, velocities, efforts, strict=False)
        }


class JointTrajectoryCollector(Collector[trajectory_msgs.JointTrajectory], ArmActionTrajectory):
    """
    Collects and preprocesses joint trajectory data from a ROS topic.
    """

    def _preprocess(self, msg: trajectory_msgs.JointTrajectory) -> ArmActionTrajectory:

        if not msg.points:
            return np.array([]), 0.0

        point = msg.points[0]
        positions = np.asarray(point.positions, dtype=float)
        tfs = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9

        return positions, tfs


class GripperStateCollector(Collector[control_msgs.GripperCommand.Feedback, GripperState]):
    """
    Collects gripper state from the gripper action server feedback topic.

    Subscribes to the action feedback topic (e.g. ``<gripper_controller>/gripper_cmd/_action/feedback``)
    and exposes position, effort, stalled, and reached_goal.
    """

    def _preprocess(self, msg: control_msgs.GripperCommand.Feedback) -> GripperState:
        return {
            "position": float(msg.position),
            "effort": float(msg.effort),
            "stalled": bool(msg.stalled),
            "reached_goal": bool(msg.reached_goal),
        }


class ContactCollector(Collector[ros_gz_interfaces.Contacts, ContactData]):
    """Collects contact state from a Gazebo contact sensor."""

    def __init__(
        self,
        name: str,
        topic: str,
        node: rclpy.node.Node | None = None,
        timeout: float = 0.5,
        up_to_date_required: bool = False,
        **kwargs: Any,
    ):
        if up_to_date_required:
            raise ValueError(
                "ContactCollector must not require up-to-date data: gz contact "
                "sensors stop publishing when the contact ends, so a freshness "
                "requirement would block forever."
            )

        super().__init__(name, topic, node, up_to_date_required=False, **kwargs)

        # gz-sim only publishes while a contact exists, so a reading must expire.
        self.timeout = timeout

    def _preprocess(self, msg: ros_gz_interfaces.Contacts) -> ContactData:
        contacts: list[ContactPoint] = []

        for contact in msg.contacts:
            other: str = contact.collision2.name

            n = len(contact.positions)
            has_normal = len(contact.normals) == n
            has_depth = len(contact.depths) == n
            has_wrench = len(contact.wrenches) == n

            for i, pos in enumerate(contact.positions):
                normal = np.zeros(3, dtype=np.float32)
                if has_normal:
                    raw = contact.normals[i]
                    normal = np.array([raw.x, raw.y, raw.z], dtype=np.float32)
                    length = float(np.linalg.norm(normal))
                    if length > 1e-6:
                        normal /= length

                depth = float(contact.depths[i]) if has_depth else 0.0

                wrench = None
                if has_wrench:
                    w = contact.wrenches[i].body_1_wrench
                    wrench = np.array(
                        [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
                        dtype=np.float32,
                    )

                contacts.append(
                    ContactPoint(
                        other=other,
                        position=np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                        normal=normal,
                        depth=depth,
                        wrench=wrench,
                    )
                )

        return ContactData(tuple(contacts))

    def get_observation(self) -> ContactData:
        """The sensor only publishes while a contact exists; expire stale readings."""
        if self._clock is not None and self.age > self.timeout:
            return ContactData()
        return self._value if self._value is not None else ContactData()
