"""
ADVANCED OBSERVATIONS: OBSERVATIONS THAT ARE NOT DIRECTLY DERIVED FROM TOPICS
"""

from __future__ import annotations

__all__ = [
    "GoalLocationInRobotFrameGenerator",
    "DistAngleToGoalGenerator",
    "SubgoalLocationInRobotFrameGenerator",
    "DistAngleToSubgoalGenerator",
    "LaserSafeDistanceGenerator",
]


from typing import TYPE_CHECKING, List, Tuple, TypeVar

import numpy as np

# import rospy

from rosnav_rl.states import SimulationStateContainer

if TYPE_CHECKING:
    from rosnav_rl.utils.type_aliases import ObservationDict

from ..collectors import (
    BaseUnit,
    FullRangeLaserCollector,
    GoalCollector,
    LaserCollector,
    RobotPoseCollector,
    SubgoalCollector,
)
from ..utils.semantic import get_relative_pos_to_robot
from .base_generator import ObservationGeneratorUnit


class GoalLocationInRobotFrameGenerator(ObservationGeneratorUnit[np.ndarray]):
    """Observation generator that provides the goal location in the robot's frame of reference.

    This generator transforms the global goal coordinates into the robot's local coordinate frame,
    which is useful for navigation tasks where the robot needs to know the relative position
    of its goal.

    Attributes:
        name (str): The name identifier for this generator.
        requires (List[BaseUnit]): The required collector units (GoalCollector and RobotPoseCollector).
        data_class (type): The output data type (numpy array).
    """

    name: str = "goal_location_in_robot_frame"
    requires: List[BaseUnit] = [GoalCollector, RobotPoseCollector]
    data_class = np.ndarray

    def generate(self, obs_dict: "ObservationDict", *args, **kwargs) -> np.ndarray:
        """Generate an observation based on the robot and goal poses.

        This method computes the relative position of the goal with respect to the robot's frame.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data with at least:
                                        - goal pose from GoalCollector
                                        - robot pose from RobotPoseCollector
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: A numpy array representing the goal's position relative to the robot's frame.
                        The array contains [x, y] coordinates after transformation.
        """

        goal_pose: GoalCollector.data_class = obs_dict[GoalCollector.name]
        robot_pose: RobotPoseCollector.data_class = obs_dict[RobotPoseCollector.name]

        return get_relative_pos_to_robot(
            robot_pose,
            np.array([[goal_pose["x"], goal_pose["y"], 1]]),
        ).squeeze(0)


class SubgoalLocationInRobotFrameGenerator(ObservationGeneratorUnit[np.ndarray]):
    """
    Generator unit that computes the relative position vector from the robot to the subgoal in the robot's frame.

    This unit transforms the global coordinates of the subgoal into a vector in the robot's local
    coordinate frame. The resulting vector represents how the robot should move to reach the subgoal
    from its current position.

    Returns:
        np.ndarray: A 2D vector [x, y] representing the subgoal's position relative to the robot's
                    current position and orientation.

    Requires:
        - SubgoalCollector: Provides the global coordinates of the current subgoal
        - RobotPoseCollector: Provides the current robot pose
    """

    name: str = "subgoal_location_in_robot_frame"
    requires: List[BaseUnit] = [SubgoalCollector, RobotPoseCollector]
    data_class = np.ndarray

    def generate(self, obs_dict: "ObservationDict", *args, **kwargs) -> np.ndarray:
        """
        Generates an observation vector containing the relative position of the goal with respect to the robot.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data, including goal and robot poses
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: A numpy array representing the relative position of the goal with respect to the robot frame
                       (usually [x, y] coordinates in the robot's coordinate system)

        Note:
            This method extracts goal and robot poses from the observation dictionary and calculates
            their relative positions using the get_relative_pos_to_robot utility function.
        """
        goal_pose: GoalCollector.data_class = obs_dict[SubgoalCollector.name]
        robot_pose: RobotPoseCollector.data_class = obs_dict[RobotPoseCollector.name]

        return get_relative_pos_to_robot(
            robot_pose,
            np.array([[goal_pose["x"], goal_pose["y"], 1]]),
        ).squeeze(0)


DistToGoal = TypeVar("DistToGoal", float, float)
AngleToGoal = TypeVar("AngleToGoal", float, float)


class DistAngleToGoalGenerator(
    ObservationGeneratorUnit[Tuple[DistToGoal, AngleToGoal]]
):
    """
    Observation generator that computes the distance and angle to the goal.

    This class combines two pieces of information:
    1. The euclidean distance from the robot to the goal
    2. The angle between the robot's forward direction and the goal

    Attributes:
        name (str): The name of this observation generator
        requires (List[BaseUnit]): Dependencies required by this generator
        data_class: The type of data returned by this generator

    Returns:
        Tuple[DistToGoal, AngleToGoal]: A numpy array containing:
            - The euclidean distance to the goal
            - The angle to the goal in radians, where 0 means the goal is directly ahead,
              positive values mean the goal is to the left, and negative values mean
              the goal is to the right
    """

    name: str = "dist_angle_to_goal"
    requires: List[BaseUnit] = [GoalLocationInRobotFrameGenerator]
    data_class = Tuple[DistToGoal, AngleToGoal]

    def generate(
        self, obs_dict: "ObservationDict", *args, **kwargs
    ) -> Tuple[DistToGoal, AngleToGoal]:
        """
        Generate distance and angle to goal based on observation dictionary.

        This method computes the Euclidean distance to the goal and the angle to the goal
        in the robot's frame of reference.

        Args:
            obs_dict: Observation dictionary containing goal location in robot frame.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple containing:
                - dist_to_goal: Euclidean distance to the goal.
                - angle_to_goal: Angle to the goal in radians.
        """
        goal_in_robot_frame: GoalLocationInRobotFrameGenerator.data_class = obs_dict[
            GoalLocationInRobotFrameGenerator.name
        ]

        dist_to_goal = np.linalg.norm(goal_in_robot_frame)
        angle_to_goal = np.arctan2(goal_in_robot_frame[1], goal_in_robot_frame[0])

        return np.array((dist_to_goal, angle_to_goal))


class DistAngleToSubgoalGenerator(
    ObservationGeneratorUnit[Tuple[DistToGoal, AngleToGoal]]
):
    name: str = "dist_angle_to_subgoal"
    requires: List[BaseUnit] = [SubgoalLocationInRobotFrameGenerator]
    data_class = Tuple[DistToGoal, AngleToGoal]

    def generate(
        self, obs_dict: "ObservationDict", *args, **kwargs
    ) -> Tuple[DistToGoal, AngleToGoal]:
        goal_in_robot_frame: SubgoalLocationInRobotFrameGenerator.data_class = obs_dict[
            SubgoalLocationInRobotFrameGenerator.name
        ]

        dist_to_goal = np.linalg.norm(goal_in_robot_frame)
        angle_to_goal = np.arctan2(goal_in_robot_frame[1], goal_in_robot_frame[0])

        return np.array((dist_to_goal, angle_to_goal))


class LaserSafeDistanceGenerator(ObservationGeneratorUnit[bool]):
    """An ObservationGeneratorUnit that determines whether any laser scan point violates the robot's safety distance.

    This class checks if any of the laser scan readings are closer than the robot's defined safety distance,
    indicating a potential safety violation.

    Attributes:
        name: The name identifier for this generator unit.
        requires: List of required collector units (LaserCollector and FullRangeLaserCollector).
        data_class: The type of data this generator produces (boolean).

    Returns:
        bool: True if any laser scan point is closer than or equal to the robot's safety distance,
              False if all points are beyond the safety distance or if there are no laser readings.
    """

    name: str = "laser_safe_distance_violation"
    requires: List[BaseUnit] = [LaserCollector]  # , FullRangeLaserCollector]
    data_class = bool

    def generate(
        self,
        obs_dict: "ObservationDict",
        simulation_state_container: SimulationStateContainer,
        *args,
        **kwargs,
    ) -> bool:
        """
        Generate information about whether the robot's safety distance is violated.

        This method checks if any laser scan value is below the robot's safety distance threshold, indicating a potential collision.

        Args:
            obs_dict (ObservationDict): Dictionary containing observation data
            simulation_state_container (SimulationStateContainer): Container holding simulation state information
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            bool: True if any laser scan reading is less than or equal to the safety distance (indicating danger),
                  False if all readings are greater than the safety distance or if no laser data is available
        """
        # if not isinstance(simulation_state_container, SimulationStateContainer):
        #     rospy.logwarn_throttle(
        #         60,
        #         f"Can't calculate '{self.name}'-Generator! SimulationStateContainer not provided.",
        #     )
        laser_data = (
            obs_dict[LaserCollector.name]
            if FullRangeLaserCollector.name not in obs_dict
            else obs_dict[FullRangeLaserCollector.name]
        )
        return (
            False
            if len(laser_data) == 0
            else laser_data.min() <= simulation_state_container.robot.safety_distance
        )
