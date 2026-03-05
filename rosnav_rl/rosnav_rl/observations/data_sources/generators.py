"""
This module contains all Generator classes.

Generators are data sources that produce new, derived data by performing
calculations on data from other `DataSource`s.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Union
from warnings import warn

import arena_people_msgs.msg as arena_people_msgs
import numpy as np
import people_msgs.msg as people_msgs
import rclpy
from rosnav_rl.utils.rostopic import Namespace
import tf2_ros
from tf_transformations import euler_from_quaternion

from ...states import SimulationStateContainer
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
    Looks up the transform from a global frame (e.g., 'map') to the robot's base frame.

    Technical Specifications:
    - Source: ROS 2 TF tree
    - Output: 2D pose (x, y, theta) in map frame
    - Handles initialization and frame configuration

    Output Format: np.ndarray of shape (3,) [x, y, theta]

    Applications: Localization, navigation, and robot-centric transformations.
    """

    # This generator doesn't depend on other data sources - it gets data from TF
    requires = {}

    def __init__(
        self,
        name: str,
        node: rclpy.Node | None = None,
        ns: Union[str, Namespace] | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._node = node
        if not self._node:
            raise ValueError("RobotPoseTFGenerator requires a ROS 2 node.")

        self._tf_buffer = tf2_ros.Buffer(node=self._node)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        self._last_pose = np.array((0.0, 0.0, 0.0), dtype=Pose2DType)
        self._is_initialized = False
        self._namespace = Namespace(ns) if ns else None

        # Set default frame names, will be updated when simulation_state_container is available
        self.TARGET_FRAME: str = "map"  # Reference/parent frame
        self.SOURCE_FRAME: str = (
            f"{self._namespace.simulation_ns.without_slashes()}_{self._namespace.robot_ns.without_slashes()}/base_link"
            if self._namespace
            else "jackal/base_link"  # Robot frame
        )

    def _generate(self, simulation_state_container: SimulationStateContainer, **kwargs) -> Pose2D:
        """Generates the robot's 2D pose (x, y, theta) from the TF tree.

        Args:
            simulation_state_container (SimulationStateContainer):
                - Simulation state for context (may provide robot config)
                - Used to update frame configuration if needed
            **kwargs: Additional keyword arguments (unused)

        Returns:
            Pose2D: Robot pose as np.ndarray of shape (3,) [x, y, theta] in map frame

        Example:
            [1.2, 3.4, 0.78]
        """

        # Update frame configuration from simulation state if needed
        if simulation_state_container and not self._is_initialized:
            # Use simulation state to get proper robot configuration
            # robot_params = arena_simulation_setup.entities.robot.Robot(
            #     "jackal"
            # ).model_params
            # self.TARGET_FRAME = f"jackal/{robot_params.base_frame}"
            pass

        if not self._is_initialized:
            try:
                self._tf_buffer.can_transform(
                    self.TARGET_FRAME,  # "map" (was SOURCE_FRAME)
                    self.SOURCE_FRAME,  # "jackal/base_link" (was TARGET_FRAME)
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> RobotRelativePosition:
        """Transforms the global goal position into the robot's local coordinate frame.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            goal_pose (GoalLocation): Goal pose in map frame [x, y]
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> RobotRelativePosition:
        """Transforms the global subgoal position into the robot's local coordinate frame.

        Args:
            subgoal_pose (SubgoalLocation): Subgoal pose in map frame [x, y]
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> DistanceAngleMetrics:
        """Computes the distance and angle to the goal from the robot's perspective.

        Args:
            goal_in_robot_frame (RobotRelativePosition): Goal position in robot frame [x, y]
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> DistanceAngleMetrics:
        """Computes the distance and angle to the subgoal from the robot's perspective.

        Args:
            subgoal_in_robot_frame (RobotRelativePosition): Subgoal position in robot frame [x, y]
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> SafetyStatus:
        """Computes the minimum safe distance from front laser scan data.

        Args:
            front_laser (LidarRanges): Laser scan ranges (np.ndarray)
            simulation_state_container (SimulationStateContainer): Provides robot safety distance
            **kwargs: Additional keyword arguments (unused)

        Returns:
            SafetyStatus: True if minimum distance is below safety threshold, else False

        Example:
            True (if too close), False (if safe)
        """
        min_distance = np.min(front_laser) if len(front_laser) > 0 else float("inf")
        safety_distance = simulation_state_container.robot.safety_distance
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeLocations:
        """Generates the relative locations of pedestrians with respect to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            people_data (people_msgs.People): List of detected pedestrians (global positions)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianWorldLocations:
        """Generates absolute locations of pedestrians in world coordinates.

        Args:
            people_data (people_msgs.People): List of detected pedestrians (global positions)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Generates the full velocity vectors of pedestrians relative to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            people_data (people_msgs.People): List of detected pedestrians (global velocities)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Extracts the X component of pedestrian velocities in the robot's frame.

        Args:
            pedestrian_relative_velocities (PedestrianRelativeVelocities): np.ndarray (N, 2) of velocities
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Extracts the Y component of pedestrian velocities in the robot's frame.

        Args:
            pedestrian_relative_velocities (PedestrianRelativeVelocities): np.ndarray (N, 2) of velocities
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianTypeMinDistances:
        """Calculates the minimum distance to pedestrians of each type/group.

        Args:
            pedestrian_relative_locations (PedestrianRelativeLocations): np.ndarray (N, 2) of positions in robot frame
            people_data (people_msgs.People): List of detected pedestrians (tags for group_id)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianTypeArray:
        """Generates an array of pedestrian group/type IDs (in order).

        Args:
            people_data (people_msgs.People): List of detected pedestrians (tags for group_id)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianSocialStates:
        """Generates an array of pedestrian social/behavior states (in order).

        Args:
            people_data (people_msgs.People): List of detected pedestrians (tags for behavior)
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeLocations:
        """Generates relative locations of Arena pedestrians with respect to the robot.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianWorldLocations:
        """Generates world locations of Arena pedestrians.

        Args:
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
        **kwargs,
    ) -> PedestrianRelativeVelocities:
        """Generates relative velocities of Arena pedestrians.

        Args:
            robot_pose (Pose2D): Robot pose in map frame [x, y, theta]
            arena_people_data (arena_people_msgs.Pedestrians): Arena pedestrian detections
            simulation_state_container (SimulationStateContainer): Simulation state (unused)
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
        simulation_state_container: SimulationStateContainer,
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
        simulation_state_container: SimulationStateContainer,
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
