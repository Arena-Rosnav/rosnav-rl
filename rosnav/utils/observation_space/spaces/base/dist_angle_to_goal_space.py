import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from ...utils import stack_spaces


from rl_utils.utils.observation_collector import DistAngleToGoal, ObservationDict


@SpaceFactory.register("dist_angle_to_goal")
class DistAngleToGoalSpace(BaseObservationSpace):
    """
    Represents the observation space for the goal in a navigation task.

    Args:
        goal_max_dist (float): The maximum distance to the goal. Defaults to 30.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _max_dist (float): The maximum distance to the goal.

    Methods:
        get_gym_space: Returns the Gym space for the goal observation.
        encode_observation: Encodes the goal observation.

    """

    name = "DIST_ANGLE_TO_GOAL"
    required_observations = [DistAngleToGoal]

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
            low=np.array([[0, -np.pi]]),
            high=np.array([[self._max_dist, np.pi]]),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> DistAngleToGoal.data_class:
        """
        Encodes the goal observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded goal observation.

        """
        return observation[DistAngleToGoal.name][np.newaxis, :]
