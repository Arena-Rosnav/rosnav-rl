import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import DistAngleToGoalGenerator
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("dist_angle_to_goal")
class DistAngleToGoalSpace(BaseObservationSpace):
    """Observation space for distance and angle to goal.

    This class defines an observation space that represents the distance and angle
    to the goal for a robot. It uses the DistAngleToGoalGenerator to provide the
    required observation data.

    Attributes:
        name (str): Name identifier for the observation space.
        required_observation_units (list): List of required generator units.

    Parameters:
        goal_max_dist (float, optional): Maximum distance to the goal in meters. Defaults to 30.
        *args: Variable length argument list passed to parent class.
        **kwargs: Arbitrary keyword arguments passed to parent class.

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
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> DistAngleToGoalGenerator.data_class:
        """
        Encodes the goal observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded goal observation.

        """
        return observation[DistAngleToGoalGenerator.name]
