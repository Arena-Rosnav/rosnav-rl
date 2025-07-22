from typing import Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import DistAngleToSubgoalGenerator
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("dist_angle_to_subgoal")
class DistAngleToSubgoalSpace(BaseObservationSpace):
    """A space for representing the distance and angle to a subgoal.

    This observation space encodes the distance and angle to a subgoal as a 2D vector.
    The first component represents the distance to the subgoal (normalized from 0 to max_dist),
    and the second component represents the angle to the subgoal (from -π to π).

    Attributes:
        name (str): The name of the observation space.
        required_observation_units (list): List of required observation generators.
        subgoal_max_dist (float, optional): The maximum distance to a subgoal. Defaults to 30.
        *args: Variable length argument list passed to the parent class.
        **kwargs: Arbitrary keyword arguments passed to the parent class.
    """

    name = "DIST_ANGLE_TO_SUBGOAL"
    required_observation_units = [DistAngleToSubgoalGenerator]

    def __init__(self, subgoal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = subgoal_max_dist
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
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> Any:  # Intended: DistAngleToSubgoalGenerator.data_class
        """
        Encodes the goal observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            DistAngleToSubgoalGenerator.data_class: The encoded goal observation.

        """
        return observation[DistAngleToSubgoalGenerator.name]
