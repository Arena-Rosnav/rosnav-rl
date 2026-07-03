"""
This module contains all Generator classes.

Generators are data sources that produce new, derived data by performing
calculations on data from other `DataSource`s.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from typing import TYPE_CHECKING
from warnings import warn

import arena_people_msgs.msg as arena_people_msgs
import numpy as np
import people_msgs.msg as people_msgs
import rclpy
import tf2_ros
from tf_transformations import euler_from_quaternion

if TYPE_CHECKING:
    from rosnav_rl.cfg.parameters import AgentParameters
from ..utils.pose import Pose2DType
from ..utils.semantic import get_relative_pos_to_robot, get_relative_vel_to_robot
from ..utils.types import (
    ArenaPedestrianDetections,
    ArenaPedestrianStates,
    DistanceAngleMetrics,
    GoalLocation,
    LidarRanges,
    PedestrianDetections,
    PedestrianDistances,
    PedestrianGraphNodes,
    PedestrianNodeMask,
    PedestrianRelativeLocations,
    PedestrianRelativeVelocities,
    PedestrianSocialStates,
    PedestrianTypeArray,
    PedestrianTypeMinDistances,
    PedestrianWorldLocations,
    Pose2D,
    RobotRelativePosition,
    SafetyStatus,
    SubgoalLocation,
)
from .base import Generator


class RobotPoseTFGenerator(Generator[Pose2D]):
    """Robot 2D Pose Generator (TF-based)

    Generates the robot's 2D pose (x, y, theta) by listening to the TF tree.
    Looks up the transform from source_frame to target_frame.

    Output Format: np.ndarray of shape (3,) [x, y, theta]
    """

    # This generator doesn't depend on other data sources - it gets data from TF
    requires = {}

    def __init__(
        self,
        name: str,
        node: rclpy.Node | None = None,
        source_frame: str = "",
        target_frame: str = "map",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._node = node
        if not self._node:
            raise ValueError("RobotPoseTFGenerator requires a ROS 2 node.")
        if not source_frame:
            raise ValueError("RobotPoseTFGenerator requires a non-empty source_frame.")

        self._tf_buffer = tf2_ros.Buffer(node=self._node)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        self._last_pose = np.array((0.0, 0.0, 0.0), dtype=Pose2DType)
        self._is_initialized = False

        self.SOURCE_FRAME: str = source_frame
        self.TARGET_FRAME: str = target_frame

    def _generate(self, simulation_state_container: AgentParameters, **kwargs) -> Pose2D:
        """Generates the robot's 2D pose (x, y, theta) from the TF tree.

        Args:
            simulation_state_container (AgentParameters): Simulation state for context
            **kwargs: Additional keyword arguments (unused)

        Returns:
            Pose2D: Robot pose as np.ndarray of shape (3,) [x, y, theta] in target frame

        Example:
            [1.2, 3.4, 0.78]
        """

        if not self._is_initialized:
            try:
                available = self._tf_buffer.can_transform(
                    self.TARGET_FRAME,
                    self.SOURCE_FRAME,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                if not available:
                    self._node.get_logger().warn(
                        f"Waiting for transform from '{self.SOURCE_FRAME}' to"
                        f" '{self.TARGET_FRAME}': frame not yet available"
                    )
                    return self._last_pose
                self._is_initialized = True
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self._node.get_logger().warn(
                    f"Waiting for transform from '{self.SOURCE_FRAME}' to '{self.TARGET_FRAME}': {e}"
                )
                return self._last_pose

        try:
            transform_stamped = self._tf_buffer.lookup_transform(
                self.TARGET_FRAME,
                self.SOURCE_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            trans = transform_stamped.transform.translation
            rot = transform_stamped.transform.rotation
            _, _, theta = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self._last_pose = np.array((trans.x, trans.y, theta), dtype=Pose2DType)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self._node.get_logger().warn(
                f"Could not get transform from '{self.SOURCE_FRAME}' to '{self.TARGET_FRAME}': {e}"
            )
        return self._last_pose


class GoalLocationInRobotFrameGenerator(Generator[RobotRelativePosition]):
    """Goal Location in Robot Frame Generator

    Transforms the global goal position into the robot's local coordinate frame for relative navigation planning.

    Technical Specifications:
    - Input: Global goal pose, robot pose
    - Output: Goal position in robot-centric frame

    Output Format: np.ndarray of shape (2,) [x, y] in robot frame

    Applications: Goal-directed navigation, local planning.
    """

    # Schema-based requirements with rich metadata (inspired by Albumentations)
    requires = {
        "robot_pose": Pose2D,
        "goal_pose": GoalLocation,
    }

    def _generate(
        self,
        robot_pose: Pose2D,
        goal_pose: GoalLocation,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> RobotRelativePosition:
        """Transforms the global goal position into the robot's local coordinate frame.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            goal_pose (GoalLocation): Goal pose in map frame [x, y]
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            RobotRelativePosition: Goal position in robot frame as np.ndarray (2,)

        Example:
            [2.5, -1.0]
        """
        return get_relative_pos_to_robot(robot_pose, np.array([[goal_pose["x"], goal_pose["y"], 1]])).squeeze(0)


class SubgoalLocationInRobotFrameGenerator(Generator[RobotRelativePosition]):
    """Subgoal Location in Robot Frame Generator

    Transforms subgoal location from global coordinates to robot's reference frame.
    Provides navigation-relevant subgoal position relative to the current robot pose.

    Technical Specifications:
    - Input: Global subgoal pose, robot pose
    - Output: Subgoal position in robot-centric frame

    Output Format: np.ndarray of shape (2,) [x, y] in robot frame

    Applications: Waypoint navigation, hierarchical planning.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "subgoal_pose": SubgoalLocation,
        "robot_pose": Pose2D,
    }

    def _generate(
        self,
        subgoal_pose: SubgoalLocation,
        robot_pose: Pose2D,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> RobotRelativePosition:
        """Transforms the global subgoal position into the robot's local coordinate frame.

        Args:
            subgoal_pose (SubgoalLocation): Subgoal pose in map frame [x, y]
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            RobotRelativePosition: Subgoal position in robot frame as np.ndarray (2,)

        Example:
            [1.0, 3.2]
        """
        return get_relative_pos_to_robot(robot_pose, np.array([[subgoal_pose["x"], subgoal_pose["y"], 1]])).squeeze(0)


class DistAngleToGoalGenerator(Generator[DistanceAngleMetrics]):
    """Distance and Angle to Goal Generator

    Computes the distance and angle to the goal from the robot's perspective.
    Provides navigation-relevant metrics for goal-directed behavior.

    Technical Specifications:
    - Input: Goal position in robot frame
    - Output: Distance and angle to goal

    Output Format: np.ndarray of shape (2,) [distance, angle]

    Applications: Reward shaping, navigation heuristics.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "goal_in_robot_frame": RobotRelativePosition,
    }

    def _generate(
        self,
        goal_in_robot_frame: RobotRelativePosition,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> DistanceAngleMetrics:
        """Computes the distance and angle to the goal from the robot's perspective.

        Args:
            goal_in_robot_frame (RobotRelativePosition): Goal position in robot frame [x, y]
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            DistanceAngleMetrics: np.ndarray (2,) [distance, angle]

        Example:
            [3.5, 0.78]
        """
        dist_to_goal = np.linalg.norm(goal_in_robot_frame)
        angle_to_goal = np.arctan2(goal_in_robot_frame[1], goal_in_robot_frame[0])
        return np.array((dist_to_goal, angle_to_goal))


class DistAngleToSubgoalGenerator(Generator[DistanceAngleMetrics]):
    """Distance and Angle to Subgoal Generator

    Computes the distance and angle to the subgoal from the robot's perspective.
    Similar to goal metrics but for intermediate waypoints.

    Technical Specifications:
    - Input: Subgoal position in robot frame
    - Output: Distance and angle to subgoal

    Output Format: np.ndarray of shape (2,) [distance, angle]

    Applications: Hierarchical navigation, subgoal-based planning.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "subgoal_in_robot_frame": RobotRelativePosition,
    }

    def _generate(
        self,
        subgoal_in_robot_frame: RobotRelativePosition,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> DistanceAngleMetrics:
        """Computes the distance and angle to the subgoal from the robot's perspective.

        Args:
            subgoal_in_robot_frame (RobotRelativePosition): Subgoal position in robot frame [x, y]
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            DistanceAngleMetrics: np.ndarray (2,) [distance, angle]

        Example:
            [2.1, -0.5]
        """
        dist_to_goal = np.linalg.norm(subgoal_in_robot_frame)
        angle_to_goal = np.arctan2(subgoal_in_robot_frame[1], subgoal_in_robot_frame[0])
        return np.array((dist_to_goal, angle_to_goal))


class LaserSafeDistanceGenerator(Generator[SafetyStatus]):
    """Laser Safe Distance Generator

    Computes the minimum safe distance from laser scan data.
    Provides distance-to-collision information for navigation safety.

    Technical Specifications:
    - Input: Laser scan ranges
    - Output: Boolean safety status (True if too close)

    Output Format: bool

    Applications: Collision avoidance, safety monitoring.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "front_laser": LidarRanges,
    }

    def _generate(
        self,
        front_laser: LidarRanges,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> SafetyStatus:
        """Computes the minimum safe distance from front laser scan data.

        Args:
            front_laser (LidarRanges): Laser scan ranges (np.ndarray)
            simulation_state_container (AgentParameters): Provides robot safety distance
            **kwargs: Additional keyword arguments (unused)

        Returns:
            SafetyStatus: True if minimum distance is below safety threshold, else False

        Example:
            True (if too close), False (if safe)
        """
        min_distance = np.min(front_laser) if len(front_laser) > 0 else float("inf")
        safety_distance = simulation_state_container.safety_distance
        return min_distance <= safety_distance


class PedestrianRelativeLocationGenerator(Generator[PedestrianRelativeLocations]):
    """Pedestrian Relative Location Generator (Optimized)

    Generates the relative locations of pedestrians with respect to the robot using pre-allocated buffers and vectorized extraction.

    Technical Specifications:
    - Input: Robot pose, detected pedestrian positions (global)
    - Output: Pedestrian positions in robot-centric frame
    - Buffer reuse for high-frequency efficiency

    Output Format: np.ndarray of shape (N, 2) [x, y] for N pedestrians

    Applications: Social navigation, crowd interaction, feature map encoding.
    """

    requires = {
        "robot_pose": Pose2D,
        "people_data": PedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._pose_buffer = None
        self._result_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        robot_pose: Pose2D,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeLocations:
        """Generates the relative locations of pedestrians with respect to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            people_data (people_msgs.People): List of detected pedestrians (global positions)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeLocations: np.ndarray of shape (N, 2) [x, y] in robot frame for N pedestrians

        Example:
            [[2.1, 1.0], [-1.5, 0.5]]
        """
        if not people_data or not people_data.people:
            return np.array([])

        num_peds = len(people_data.people)
        if self._pose_buffer is None or self._last_num_peds != num_peds:
            self._pose_buffer = np.ones((num_peds, 3), dtype=np.float32)
            self._result_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        # positions
        self._pose_buffer[:, :2] = np.array(
            [(p.position.x, p.position.y) for p in people_data.people], dtype=np.float32
        )

        # Use pre-allocated result buffer if get_relative_pos_to_robot supports it
        return get_relative_pos_to_robot(
            robot_pose=robot_pose,
            distant_poses=self._pose_buffer,
            output_buffer=self._result_buffer,
        )


class PedestrianLocationGenerator(Generator[PedestrianWorldLocations]):
    """Pedestrian World Location Generator

    Generates absolute locations of pedestrians in world coordinates.
    Simple extraction of pedestrian positions from detection data.

    Technical Specifications:
    - Input: Detected pedestrian positions (global)
    - Output: Pedestrian positions in world frame
    - Buffer reuse for efficiency

    Output Format: np.ndarray of shape (N, 2) [x, y] for N pedestrians

    Applications: Global crowd analysis, visualization, logging.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "people_data": PedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._locations_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianWorldLocations:
        """Generates absolute locations of pedestrians in world coordinates.

        Args:
            people_data (people_msgs.People): List of detected pedestrians (global positions)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianWorldLocations: np.ndarray of shape (N, 2) [x, y] in world frame for N pedestrians

        Example:
            [[1.0, 2.0], [3.5, -0.5]]
        """
        if not people_data or not people_data.people:
            return np.array([])

        num_peds = len(people_data.people)
        if self._locations_buffer is None or self._last_num_peds != num_peds:
            self._locations_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        self._locations_buffer[:, 0] = [p.position.x for p in people_data.people]
        self._locations_buffer[:, 1] = [p.position.y for p in people_data.people]
        return self._locations_buffer[:num_peds]


class PedestrianRelativeVelGenerator(Generator[PedestrianRelativeVelocities]):
    """Pedestrian Relative Velocity Generator

    Generates the full velocity vectors of pedestrians relative to the robot.
    Computes pedestrian velocities in the robot's reference frame for motion prediction and social navigation.

    Technical Specifications:
    - Input: Robot pose, detected pedestrian velocities (global)
    - Output: Pedestrian velocities in robot-centric frame
    - Buffer reuse for efficiency

    Output Format: np.ndarray of shape (N, 2) [vx, vy] for N pedestrians

    Applications: Social navigation, motion prediction, feature map encoding.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "robot_pose": Pose2D,
        "people_data": PedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._vel_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        robot_pose: Pose2D,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Generates the full velocity vectors of pedestrians relative to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            people_data (people_msgs.People): List of detected pedestrians (global velocities)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeVelocities: np.ndarray of shape (N, 2) [vx, vy] in robot frame for N pedestrians

        Example:
            [[0.5, 1.2], [-0.8, 0.3]]
        """
        if not people_data or not people_data.people:
            return np.array([])

        num_peds = len(people_data.people)
        if self._vel_buffer is None or self._last_num_peds != num_peds:
            self._vel_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        self._vel_buffer[:, 0] = [p.velocity.x for p in people_data.people]
        self._vel_buffer[:, 1] = [p.velocity.y for p in people_data.people]

        return get_relative_vel_to_robot(
            robot_pose=robot_pose,
            pedestrian_vel_vector=self._vel_buffer[:num_peds],
        )


class PedestrianRelativeVelXGenerator(Generator[PedestrianRelativeVelocities]):
    """Pedestrian Relative Velocity X-Component Generator

    Extracts the X component of pedestrian velocities in the robot's frame.
    Provides the forward/backward velocity component for each pedestrian.

    Technical Specifications:
    - Input: Pedestrian velocities in robot frame
    - Output: X component (forward/backward) for each pedestrian

    Output Format: np.ndarray of shape (N,) for N pedestrians

    Applications: Flow analysis, feature map encoding, crowd movement prediction.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "pedestrian_relative_velocities": PedestrianRelativeVelocities,
    }

    def _generate(
        self,
        pedestrian_relative_velocities: PedestrianRelativeVelocities,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Extracts the X component of pedestrian velocities in the robot's frame.

        Args:
            pedestrian_relative_velocities (PedestrianRelativeVelocities): np.ndarray (N, 2) of velocities
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeVelocities: np.ndarray (N,) X component for N pedestrians

        Example:
            [0.5, -0.8]
        """
        if len(pedestrian_relative_velocities) == 0:
            return np.array([])
        return pedestrian_relative_velocities[:, 0]


class PedestrianRelativeVelYGenerator(Generator[PedestrianRelativeVelocities]):
    """Pedestrian Relative Velocity Y-Component Generator

    Generates the y-component of pedestrian velocities relative to the robot.
    Extracts left/right motion component in robot's local y-axis.

    Technical Specifications:
    - Input: Pedestrian velocities in robot frame
    - Output: Y component (left/right) for each pedestrian

    Output Format: np.ndarray of shape (N,) for N pedestrians

    Applications: Lateral flow analysis, feature map encoding, group movement detection.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "pedestrian_relative_velocities": PedestrianRelativeVelocities,
    }

    def _generate(
        self,
        pedestrian_relative_velocities: PedestrianRelativeVelocities,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Extracts the Y component of pedestrian velocities in the robot's frame.

        Args:
            pedestrian_relative_velocities (PedestrianRelativeVelocities): np.ndarray (N, 2) of velocities
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeVelocities: np.ndarray (N,) Y component for N pedestrians

        Example:
            [1.2, 0.3]
        """
        return (
            pedestrian_relative_velocities[:, 1]
            if len(pedestrian_relative_velocities) > 0
            else pedestrian_relative_velocities
        )


class PedestrianDistanceGenerator(Generator[PedestrianTypeMinDistances]):
    """Pedestrian Type Minimum Distance Generator

    Calculates the minimum distance to pedestrians of each type/group.
    Uses pedestrian relative locations and group IDs to compute the minimum distance to each unique type.

    Technical Specifications:
    - Input: Pedestrian relative locations, group IDs
    - Output: Dictionary mapping group ID to minimum distance

    Output Format: dict {group_id: min_distance}

    Applications: Social group analysis, safety metrics, reward shaping.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "pedestrian_relative_locations": PedestrianRelativeLocations,  # Output from PedestrianRelativeLocationGenerator
        "people_data": PedestrianDetections,
    }

    def _generate(
        self,
        pedestrian_relative_locations: PedestrianRelativeLocations,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianTypeMinDistances:
        """Calculates the minimum distance to pedestrians of each type/group.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): np.ndarray (N, 2) of positions in robot frame
            people_data (people_msgs.People): List of detected pedestrians (tags for group_id)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianTypeMinDistances: dict {group_id: min_distance}

        Example:
            {"1": 2.3, "2": 1.1}
        """
        if len(pedestrian_relative_locations) == 0 or not people_data.people:
            return {}

        if len(pedestrian_relative_locations) != len(people_data.people):
            warn("Number of pedestrian locations and people do not match!")
            return {}

        # Vectorized distance calculation for efficiency
        distances = np.linalg.norm(pedestrian_relative_locations, axis=1)
        min_distances = defaultdict(lambda: float("inf"))

        for i, person in enumerate(people_data.people):
            try:
                group_id = person.tags[person.tagnames.index("group_id")]
                distance = distances[i]

                if distance < min_distances[group_id]:
                    min_distances[group_id] = distance
            except (ValueError, IndexError):
                # Skip person if 'group_id' tag is missing or malformed
                continue

        return dict(min_distances)


class PedestrianTypeGenerator(Generator[PedestrianTypeArray]):
    """Pedestrian Type Array Generator

    Generates an array of pedestrian group IDs/types (in order).
    Extracts group ID tags from pedestrian data for social navigation.

    Technical Specifications:
    - Input: Detected pedestrian data (tags)
    - Output: Array of group/type IDs for each pedestrian
    - Uses group_id tag as unique identifier

    Output Format: np.ndarray of shape (N,) for N pedestrians

    Applications: Semantic crowd analysis, group-based navigation, feature map encoding.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "people_data": PedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._ped_ids = []
        self._ped_types_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianTypeArray:
        """Generates an array of pedestrian group/type IDs (in order).

        Args:
            people_data (people_msgs.People): List of detected pedestrians (tags for group_id)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianTypeArray: np.ndarray (N,) of group/type IDs for N pedestrians

        Example:
            [1, 2, 1, 3]
        """
        if not people_data or not people_data.people:
            self._ped_ids = []
            self._ped_types_buffer = None
            self._last_num_peds = 0
            return np.array([])

        # Use group_id as the unique identifier for each pedestrian
        try:
            group_id_idx = people_data.people[0].tagnames.index("group_id")
            current_ped_ids = [p.tags[group_id_idx] for p in people_data.people]
        except (ValueError, AttributeError, IndexError):
            warn("Pedestrian group ID not found in the data. Returning empty array.")
            self._ped_ids = []
            self._ped_types_buffer = np.array([])
            self._last_num_peds = 0
            return self._ped_types_buffer

        num_peds = len(people_data.people)
        if self._ped_types_buffer is None or self._last_num_peds != num_peds or current_ped_ids != self._ped_ids:
            try:
                self._ped_types_buffer = np.empty(num_peds, dtype=int)
                for i, data in enumerate(people_data.people):
                    self._ped_types_buffer[i] = int(data.tags[group_id_idx])
            except (ValueError, AttributeError, IndexError):
                warn("Pedestrian group ID not found in the data. Returning empty array.")
                self._ped_types_buffer = np.array([])
            self._ped_ids = current_ped_ids
            self._last_num_peds = num_peds
        return self._ped_types_buffer[:num_peds] if self._ped_types_buffer is not None else np.array([])


class PedestrianSocialStateGenerator(Generator[PedestrianSocialStates]):
    """Pedestrian Social State Array Generator

    Generates an array of pedestrian social/behavior states (in order).
    Extracts behavior tags from pedestrian data for social-aware navigation.

    Technical Specifications:
    - Input: Detected pedestrian data (tags)
    - Output: Array of social state codes for each pedestrian

    Output Format: np.ndarray of shape (N,) for N pedestrians

    Applications: Social behavior recognition, group interaction modeling, feature map encoding.
    """

    # Schema-based requirements with rich metadata
    requires = {
        "people_data": PedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._ped_ids = []
        self._ped_social_states_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianSocialStates:
        """Generates an array of pedestrian social/behavior states (in order).

        Args:
            people_data (people_msgs.People): List of detected pedestrians (tags for behavior)
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianSocialStates: np.ndarray (N,) of social state codes for N pedestrians

        Example:
            [0, 2, 1, 3]
        """
        if not people_data or not people_data.people:
            self._ped_ids = []
            self._ped_social_states_buffer = None
            self._last_num_peds = 0
            return np.array([])

        current_ped_ids = [p.name for p in people_data.people]
        num_peds = len(people_data.people)
        if (
            self._ped_social_states_buffer is None
            or self._last_num_peds != num_peds
            or current_ped_ids != self._ped_ids
        ):
            try:
                behavior_idx = people_data.people[0].tagnames.index("behavior")
                self._ped_social_states_buffer = np.empty(num_peds, dtype=int)
                for i, data in enumerate(people_data.people):
                    self._ped_social_states_buffer[i] = int(data.tags[behavior_idx])
            except (ValueError, AttributeError, IndexError):
                warn("Pedestrian social state not found in the data. Returning empty array.")
                self._ped_social_states_buffer = np.array([])
            self._ped_ids = current_ped_ids
            self._last_num_peds = num_peds
        return self._ped_social_states_buffer[:num_peds] if self._ped_social_states_buffer is not None else np.array([])


class PedestrianGraphNodeGenerator(Generator[PedestrianGraphNodes]):
    """Fixed-size pedestrian node-set for the Social-RSSM GAT (C1, M1.1).

    Produces a padded, robot-frame node tensor of shape ``(max_peds, F + 1)`` where each row is
    ``[dx, dy, vx, vy, (social_state,) valid]``. The nearest ``max_peds`` pedestrians by distance
    are kept; the rest are dropped. Remaining rows are zero-padded and flagged invalid via the
    trailing validity column, so the GAT and the mask space can mask padding unambiguously.

    Ordering is deterministic: rows are sorted by ascending robot-frame distance with a stable
    pedestrian-id tie-break (CRC32 of the name), so the node ordering does not flicker between
    frames when two pedestrians are equidistant.

    The output width is the single source of truth tied to ``SocialCfg.node_feat_dim``:
    ``F = 4`` (dx, dy, vx, vy) or ``F = 5`` when ``include_social_state`` is True.
    """

    requires = {
        "robot_pose": Pose2D,
        "people_data": PedestrianDetections,
    }

    def __init__(
        self,
        name: str,
        max_peds: int = 8,
        include_social_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self._max_peds = int(max_peds)
        self._include_social_state = bool(include_social_state)
        # F feature columns (+1 trailing validity column).
        self._num_features = 4 + (1 if self._include_social_state else 0)
        self._width = self._num_features + 1

        # Pre-allocate reusable buffers — avoids per-step numpy allocation at 20 Hz.
        # Sized for max possible peds (max_peds); resized lazily if scene has more.
        self._out_buf: np.ndarray = np.zeros((self._max_peds, self._width), dtype=np.float32)
        self._poses_h_buf: np.ndarray = np.ones((self._max_peds, 3), dtype=np.float32)
        self._vel_buf: np.ndarray = np.zeros((self._max_peds, 2), dtype=np.float32)
        self._social_buf: np.ndarray = np.zeros(self._max_peds, dtype=np.float32)
        self._ids_buf: np.ndarray = np.zeros(self._max_peds, dtype=np.int64)
        # CRC32 cache: ped name → stable integer id. Names are stable across steps.
        self._id_cache: dict = {}

    def _ensure_buffers(self, num: int) -> None:
        """Grow all working buffers if the scene has more peds than max_peds."""
        if num <= self._poses_h_buf.shape[0]:
            return
        self._poses_h_buf = np.ones((num, 3), dtype=np.float32)
        self._vel_buf = np.zeros((num, 2), dtype=np.float32)
        self._social_buf = np.zeros(num, dtype=np.float32)
        self._ids_buf = np.zeros(num, dtype=np.int64)

    def _stable_id(self, name: str) -> int:
        """Return a deterministic integer id for a pedestrian name (cached CRC32)."""
        cached = self._id_cache.get(name)
        if cached is None:
            cached = zlib.crc32(str(name).encode("utf-8"))
            self._id_cache[name] = cached
        return cached

    def _extract_social_states(self, people: list, out: np.ndarray) -> None:
        """Fill ``out`` with per-ped behavior codes from people_msgs tags (0 when absent)."""
        out[:len(people)] = 0.0
        try:
            behavior_idx = people[0].tagnames.index("behavior")
            for i, person in enumerate(people):
                out[i] = float(int(person.tags[behavior_idx]))
        except (ValueError, AttributeError, IndexError):
            # No behavior tag in this simulator configuration: fall back to zeros (M1.5).
            pass

    def _generate(
        self,
        robot_pose: Pose2D,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianGraphNodes:
        """Generate the padded, robot-frame pedestrian node-set.

        Args:
            robot_pose (Pose2D): Robot pose in the map frame (keys x, y, yaw).
            people_data (people_msgs.People): Detected pedestrians in the map frame.
            simulation_state_container (AgentParameters): Simulation state (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            PedestrianGraphNodes: ``(max_peds, F + 1)`` float32 array, nearest-N padded.
        """
        self._out_buf.fill(0.0)
        if not people_data or not people_data.people:
            return self._out_buf

        people = people_data.people
        num = len(people)
        self._ensure_buffers(num)

        # Extract world-frame positions and velocities into pre-allocated buffers.
        ph = self._poses_h_buf
        vb = self._vel_buf
        ph[:num, 2] = 1.0  # homogeneous coordinate
        for i, p in enumerate(people):
            ph[i, 0] = p.position.x
            ph[i, 1] = p.position.y
            vb[i, 0] = p.velocity.x
            vb[i, 1] = p.velocity.y

        rel_pos = np.asarray(
            get_relative_pos_to_robot(robot_pose=robot_pose, distant_poses=ph[:num])
        )
        rel_vel = np.asarray(get_relative_vel_to_robot(robot_pose, vb[:num]))

        # Deterministic nearest-N: primary key = distance, tie-break = cached CRC32 id.
        distances = np.linalg.norm(rel_pos, axis=1)
        ids = self._ids_buf
        for i, p in enumerate(people):
            ids[i] = self._stable_id(p.name)
        order = np.lexsort((ids[:num], distances))
        keep = order[: self._max_peds]
        n = len(keep)

        self._out_buf[:n, 0:2] = rel_pos[keep]
        self._out_buf[:n, 2:4] = rel_vel[keep]
        col = 4
        if self._include_social_state:
            self._extract_social_states(people, self._social_buf)
            self._out_buf[:n, col] = self._social_buf[keep]
            col += 1
        self._out_buf[:n, col] = 1.0  # validity flag (trailing column)
        return self._out_buf


class PedestrianTrajectoryBufferGenerator(Generator[PedestrianGraphNodes]):
    """Per-ped K-step trajectory ring buffer for DALI inference (M1.3).

    Maintains a ``deque(maxlen=K)`` of node rows keyed by pedestrian name so
    DALI's GRU can seed from the real K=40-step window during acting.

    **Training does NOT use this generator.** Training trajectories come from
    the replay-buffer ``batch_length`` sequence that the RSSM already processes,
    so no separate buffer is needed there.  This generator is active at inference
    only (``social.dali.enabled=True``).

    Output shape: ``(max_peds, K, F)`` float32.  A ped that has fewer than K
    history frames is left-padded with zeros.  If a ped disappears and
    reappears its history is reset (ring-buffer entry deleted).
    """

    requires = {
        "robot_pose": Pose2D,
        "people_data": PedestrianDetections,
    }

    def __init__(
        self,
        name: str,
        max_peds: int = 8,
        k_steps: int = 40,
        node_feat_dim: int = 5,
        **kwargs,
    ) -> None:
        from collections import deque as _deque

        super().__init__(name, **kwargs)
        self._max_peds = int(max_peds)
        self._k_steps = int(k_steps)
        self._node_feat_dim = int(node_feat_dim)
        self._deque_cls = _deque
        # Per-ped history: name → (K, F) circular buffer + write pointer.
        # Using (K, F) ring arrays instead of deques of (F,) rows avoids
        # per-step list() conversion and heap allocation at 20 Hz.
        self._history: dict = {}   # name → (K, F) float32 array
        self._history_len: dict = {}  # name → int: valid rows from the right
        # Shared pre-allocated row buffer (reused each step).
        self._row_buf: np.ndarray = np.zeros(self._node_feat_dim, dtype=np.float32)
        # Pre-allocated output and working buffers.
        self._out_buf: np.ndarray = np.zeros(
            (self._max_peds, self._k_steps, self._node_feat_dim), dtype=np.float32
        )
        self._poses_h_buf: np.ndarray = np.ones((self._max_peds, 3), dtype=np.float32)
        self._vel_buf: np.ndarray = np.zeros((self._max_peds, 2), dtype=np.float32)
        self._ids_buf: np.ndarray = np.zeros(self._max_peds, dtype=np.int64)
        self._id_cache: dict = {}

    def _stable_id(self, name: str) -> int:
        cached = self._id_cache.get(name)
        if cached is None:
            cached = zlib.crc32(str(name).encode("utf-8"))
            self._id_cache[name] = cached
        return cached

    def reset(self) -> None:
        """Clear history at episode boundaries."""
        self._history.clear()
        self._history_len.clear()
        self._id_cache.clear()

    def _push_row(self, name: str, row: np.ndarray) -> None:
        """Append a feature row to the named ped's ring buffer (shift-left, write at end)."""
        if name not in self._history:
            self._history[name] = np.zeros(
                (self._k_steps, self._node_feat_dim), dtype=np.float32
            )
            self._history_len[name] = 0
        buf = self._history[name]
        h = self._history_len[name]
        if h < self._k_steps:
            buf[h] = row
            self._history_len[name] = h + 1
        else:
            # Ring: shift left by one, write at end.
            buf[:-1] = buf[1:]
            buf[-1] = row

    def _generate(
        self,
        robot_pose: Pose2D,
        people_data: people_msgs.People,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> np.ndarray:
        """Build (max_peds, K, F) trajectory tensor from ring buffers.

        Peds are ordered by ascending robot-frame distance (same ordering as
        ``PedestrianGraphNodeGenerator``) so slot indices align between generators.
        Missing history is left-padded with zeros.

        Returns:
            np.ndarray: ``(max_peds, K, F)`` float32.
        """
        F = self._node_feat_dim
        self._out_buf.fill(0.0)
        if not people_data or not people_data.people:
            self._history.clear()
            self._history_len.clear()
            return self._out_buf

        people = people_data.people
        num = len(people)

        # Grow working buffers lazily.
        if num > self._poses_h_buf.shape[0]:
            self._poses_h_buf = np.ones((num, 3), dtype=np.float32)
            self._vel_buf = np.zeros((num, 2), dtype=np.float32)
            self._ids_buf = np.zeros(num, dtype=np.int64)

        ph = self._poses_h_buf
        vb = self._vel_buf
        ph[:num, 2] = 1.0
        for i, p in enumerate(people):
            ph[i, 0] = p.position.x
            ph[i, 1] = p.position.y
            vb[i, 0] = p.velocity.x
            vb[i, 1] = p.velocity.y

        rel_pos = np.asarray(
            get_relative_pos_to_robot(robot_pose=robot_pose, distant_poses=ph[:num])
        )
        rel_vel = np.asarray(get_relative_vel_to_robot(robot_pose, vb[:num]))

        distances = np.linalg.norm(rel_pos, axis=1)
        ids = self._ids_buf
        for i, p in enumerate(people):
            ids[i] = self._stable_id(p.name)
        order = np.lexsort((ids[:num], distances))
        keep = order[: self._max_peds]

        seen_names: set = set()
        row = self._row_buf
        for slot, idx in enumerate(keep):
            p = people[idx]
            seen_names.add(p.name)
            row[:] = 0.0
            row[0:2] = rel_pos[idx]
            if F > 2:
                row[2:4] = rel_vel[idx]
            self._push_row(p.name, row)
            buf = self._history[p.name]
            h = self._history_len[p.name]
            # Write the h valid rows into the right-aligned output slot (left-pad zeros).
            self._out_buf[slot, self._k_steps - h :] = buf[:h]

        # Drop history for peds that have left detection range.
        stale = [n for n in self._history if n not in seen_names]
        for name in stale:
            del self._history[name]
            del self._history_len[name]

        return self._out_buf


# =============================================================================
# ARENA PEDESTRIAN GENERATORS
# =============================================================================
# These generators process arena_people_msgs/Pedestrians messages which include
# header, full pose/twist data, and animation states.


class ArenaPedestrianRelativeLocationGenerator(Generator[PedestrianRelativeLocations]):
    """Arena Pedestrian Relative Location Generator

    Generates pedestrian positions relative to the robot's frame using Arena pedestrian data.
    Transforms pedestrian positions from global to robot-centric coordinates.

    Technical Specifications:
    - Input: Robot pose, Arena pedestrian detections (with full pose data)
    - Output: Pedestrian positions in robot-centric frame
    - Handles arena_people_msgs/Pedestrians message type

    Output Format: np.ndarray of shape (N, 2) [x, y] for N pedestrians

    Applications: Social navigation with Arena simulator data.
    """

    requires = {
        "robot_pose": Pose2D,
        "arena_people_data": ArenaPedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._pose_buffer = None
        self._result_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        robot_pose: Pose2D,
        arena_people_data: arena_people_msgs.Pedestrians,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeLocations:
        """Generates relative locations of Arena pedestrians with respect to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeLocations: np.ndarray of shape (N, 2) [x, y] in robot frame
        """
        if not arena_people_data or not arena_people_data.pedestrians:
            return np.array([], dtype=np.float32)

        num_peds = len(arena_people_data.pedestrians)
        if self._pose_buffer is None or self._last_num_peds != num_peds:
            self._pose_buffer = np.ones((num_peds, 3), dtype=np.float32)
            self._result_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        # Extract positions directly into buffer using vectorized operations
        for i, p in enumerate(arena_people_data.pedestrians):
            self._pose_buffer[i, 0] = p.pose.position.x
            self._pose_buffer[i, 1] = p.pose.position.y

        return get_relative_pos_to_robot(
            robot_pose=robot_pose,
            distant_poses=self._pose_buffer,
            output_buffer=self._result_buffer,
        )


class ArenaPedestrianLocationGenerator(Generator[PedestrianWorldLocations]):
    """Arena Pedestrian World Location Generator

    Generates absolute locations of Arena pedestrians in world coordinates.
    Simple extraction of pedestrian positions from Arena detection data.

    Technical Specifications:
    - Input: Arena pedestrian detections (global)
    - Output: Pedestrian positions in world frame

    Output Format: np.ndarray of shape (N, 2) [x, y] for N pedestrians

    Applications: Global crowd analysis with Arena simulator data.
    """

    requires = {
        "arena_people_data": ArenaPedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._location_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        arena_people_data: arena_people_msgs.Pedestrians,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianWorldLocations:
        """Generates world locations of Arena pedestrians.

        Args:
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianWorldLocations: np.ndarray of shape (N, 2) [x, y] in world frame
        """
        if not arena_people_data or not arena_people_data.pedestrians:
            return np.array([], dtype=np.float32)

        num_peds = len(arena_people_data.pedestrians)
        if self._location_buffer is None or self._last_num_peds != num_peds:
            self._location_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        # Direct buffer updates - more cache-friendly
        for i, p in enumerate(arena_people_data.pedestrians):
            self._location_buffer[i, 0] = p.pose.position.x
            self._location_buffer[i, 1] = p.pose.position.y

        return self._location_buffer


class ArenaPedestrianRelativeVelGenerator(Generator[PedestrianRelativeVelocities]):
    """Arena Pedestrian Relative Velocity Generator

    Generates relative velocities of Arena pedestrians with respect to the robot.
    Transforms pedestrian velocities from global to robot-centric frame.

    Technical Specifications:
    - Input: Robot pose, Arena pedestrian detections (with twist data)
    - Output: Pedestrian velocities in robot-centric frame

    Output Format: np.ndarray of shape (N, 2) [vx, vy] for N pedestrians

    Applications: Motion prediction, collision avoidance with Arena simulator data.
    """

    requires = {
        "robot_pose": Pose2D,
        "arena_people_data": ArenaPedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._vel_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        robot_pose: Pose2D,
        arena_people_data: arena_people_msgs.Pedestrians,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Generates relative velocities of Arena pedestrians.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (AgentParameters): Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianRelativeVelocities: np.ndarray of shape (N, 2) [vx, vy] in robot frame
        """
        if not arena_people_data or not arena_people_data.pedestrians:
            return np.array([], dtype=np.float32)

        num_peds = len(arena_people_data.pedestrians)
        if self._vel_buffer is None or self._last_num_peds != num_peds:
            self._vel_buffer = np.empty((num_peds, 2), dtype=np.float32)
            self._last_num_peds = num_peds

        # Direct velocity extraction into buffer
        for i, p in enumerate(arena_people_data.pedestrians):
            self._vel_buffer[i, 0] = p.twist.linear.x
            self._vel_buffer[i, 1] = p.twist.linear.y

        return get_relative_vel_to_robot(
            robot_pose=robot_pose,
            pedestrian_vel_vector=self._vel_buffer,
        )


class ArenaPedestrianDistanceGenerator(Generator[PedestrianDistances]):
    """Arena Pedestrian Distance Generator

    Computes minimum distances from robot to Arena pedestrians grouped by ID.

    Technical Specifications:
    - Input: Arena pedestrian relative locations
    - Output: Dictionary mapping pedestrian ID to minimum distance

    Output Format: Dict[int, float] {id: distance}

    Applications: Safety monitoring, social distance compliance with Arena data.
    """

    requires = {
        "arena_pedestrian_relative_locations": PedestrianRelativeLocations,
        "arena_people_data": ArenaPedestrianDetections,
    }

    def _generate(
        self,
        arena_pedestrian_relative_locations: PedestrianRelativeLocations,
        arena_people_data: arena_people_msgs.Pedestrians,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> PedestrianDistances:
        """Computes minimum distances to Arena pedestrians by ID.

        Args:
            arena_pedestrian_relative_locations: Relative positions of pedestrians
            arena_people_data: Arena pedestrian detections
            simulation_state_container: Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            PedestrianDistances: Dict mapping pedestrian ID to distance
        """
        if not arena_people_data or not arena_people_data.pedestrians or len(arena_pedestrian_relative_locations) == 0:
            return {}

        # Vectorized distance computation (much faster)
        distances_array = np.linalg.norm(arena_pedestrian_relative_locations, axis=1)

        # Build result dict efficiently
        result = {}
        for i, ped in enumerate(arena_people_data.pedestrians):
            dist = distances_array[i]
            ped_id = ped.id
            if ped_id not in result or dist < result[ped_id]:
                result[ped_id] = float(dist)

        return result


class ArenaPedestrianStateGenerator(Generator[ArenaPedestrianStates]):
    """Arena Pedestrian Animation State Generator

    Extracts animation/behavior states from Arena pedestrian detections.
    States include: IDLE, WALKING, RUNNING, PANIC, SURPRISED, CURIOUS, THREATENING.

    Technical Specifications:
    - Input: Arena pedestrian detections
    - Output: Array of animation state codes (uint8)

    Output Format: np.ndarray of shape (N,) with state codes [0-6]

    Applications: Behavior-aware navigation, pedestrian intention prediction.
    """

    requires = {
        "arena_people_data": ArenaPedestrianDetections,
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._animation_states_buffer = None
        self._last_num_peds = 0

    def _generate(
        self,
        arena_people_data: arena_people_msgs.Pedestrians,
        simulation_state_container: AgentParameters,
        **kwargs,
    ) -> ArenaPedestrianStates:
        """Extracts animation states from Arena pedestrian data.

        Args:
            arena_people_data: Arena pedestrian detections
            simulation_state_container: Simulation state (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            ArenaPedestrianStates: np.ndarray of animation state codes
        """
        if not arena_people_data or not arena_people_data.pedestrians:
            return np.array([], dtype=np.uint8)

        num_peds = len(arena_people_data.pedestrians)
        if self._animation_states_buffer is None or self._last_num_peds != num_peds:
            self._animation_states_buffer = np.empty(num_peds, dtype=np.uint8)
            self._last_num_peds = num_peds

        # Direct buffer population - avoids intermediate list
        for i, p in enumerate(arena_people_data.pedestrians):
            self._animation_states_buffer[i] = p.animation_state

        return self._animation_states_buffer


class PedestrianNodeMaskGenerator(Generator[PedestrianNodeMask]):
    """Extracts the validity mask column from the padded pedestrian node-set.

    The ``PedestrianGraphNodeGenerator`` stores the validity flag in the last
    column of the ``(max_peds, F+1)`` node tensor.  This generator slices that
    column out so ``PedestrianMaskSpace`` can consume it independently from the
    main node features consumed by ``PedestrianNodeSetSpace``.
    """

    requires = {"peds_nodes": PedestrianGraphNodes}

    def _generate(self, peds_nodes: PedestrianGraphNodes, **_: object) -> PedestrianNodeMask:
        return peds_nodes[:, -1].astype(np.float32)
