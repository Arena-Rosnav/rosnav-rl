"""Legacy Navigation Spaces - Migrated from Base

Original goal and subgoal navigation spaces integrated into hierarchical architecture.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import DistAngleToGoalGenerator, DistAngleToSubgoalGenerator
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)


@SpaceFactory.register("dist_angle_to_goal", SpaceCategory.NAVIGATION)
class DistAngleToGoalSpace(BaseObservationSpace):
    """Original observation space for distance and angle to goal.

    This class defines an observation space that represents the distance and angle
    to the goal for a robot. It uses the DistAngleToGoalGenerator to provide the
    required observation data.

    A 2-dimensional observation space where:
        - First dimension: distance to goal [0, goal_max_dist]
        - Second dimension: angle to goal [-π, π]
    """

    name = "DIST_ANGLE_TO_GOAL"
    required_observation_units = [DistAngleToGoalGenerator]

    def __init__(self, goal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = goal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the goal observation.

        Returns:
            spaces.Space: The Gym space for the goal observation.
        """
        return spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([self._max_dist, np.pi]),
            dtype=np.float32,
            shape=(2,),
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Encodes the goal observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded goal observation.
        """
        return observation[DistAngleToGoalGenerator.name]


@SpaceFactory.register("dist_angle_to_subgoal", SpaceCategory.NAVIGATION)
class DistAngleToSubgoalSpace(BaseObservationSpace):
    """A space for representing the distance and angle to a subgoal.

    This observation space encodes the distance and angle to a subgoal as a 2D vector.
    The first component represents the distance to the subgoal (normalized from 0 to max_dist),
    and the second component represents the angle to the subgoal (from -π to π).
    """

    name = "DIST_ANGLE_TO_SUBGOAL"
    required_observation_units = [DistAngleToSubgoalGenerator]

    def __init__(self, subgoal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = subgoal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym space for the subgoal observation.

        Returns:
            spaces.Space: The Gym space for the subgoal observation.
        """
        return spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([self._max_dist, np.pi]),
            dtype=np.float32,
            shape=(2,),
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Encodes the subgoal observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded subgoal observation.
        """
        return observation[DistAngleToSubgoalGenerator.name]
