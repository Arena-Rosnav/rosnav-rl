from typing import Tuple

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from ...observation_space_factory import SpaceFactory
from ...utils import stack_spaces
from ..base_observation_space import BaseObservationSpace

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


@SpaceFactory.register("last_action")
class LastActionSpace(BaseObservationSpace):
    """
    Observation space representing the last action taken by the agent.

    Args:
        min_linear_vel (float): The minimum linear velocity.
        max_linear_vel (float): The maximum linear velocity.
        min_angular_vel (float): The minimum angular velocity.
        max_angular_vel (float): The maximum angular velocity.
        min_translational_vel (float, optional): The minimum translational velocity. Defaults to 0.0.
        max_translational_vel (float, optional): The maximum translational velocity. Defaults to 0.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _min_linear_vel (float): The minimum linear velocity.
        _max_linear_vel (float): The maximum linear velocity.
        _min_translational_vel (float): The minimum translational velocity.
        _max_translational_vel (float): The maximum translational velocity.
        _min_angular_vel (float): The minimum angular velocity.
        _max_angular_vel (float): The maximum angular velocity.

    Methods:
        get_gym_space(): Returns the Gym observation space.
        encode_observation(observation, *args, **kwargs): Encodes the observation.

    """

    def __init__(
        self,
        min_linear_vel: float,
        max_linear_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        min_translational_vel: float = 0.0,
        max_translational_vel: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        self._min_linear_vel = min_linear_vel
        self._max_linear_vel = max_linear_vel
        self._min_translational_vel = min_translational_vel
        self._max_translational_vel = max_translational_vel
        self._min_angular_vel = min_angular_vel
        self._max_angular_vel = max_angular_vel
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym spaces for the last action space.

        Returns:
            A tuple of gym spaces representing the last action space.
        """
        _spaces = (
            spaces.Box(
                low=self._min_linear_vel,
                high=self._max_linear_vel,
                shape=(1,),
                dtype=np.float32,
            ),
            spaces.Box(
                low=self._min_translational_vel,
                high=self._max_translational_vel,
                shape=(1,),
                dtype=np.float32,
            ),
            spaces.Box(
                low=self._min_angular_vel,
                high=self._max_angular_vel,
                shape=(1,),
                dtype=np.float32,
            ),
        )
        return stack_spaces(*_spaces)

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        """
        Encodes the observation by extracting the last action from the observation dictionary.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded observation representing the last action.
        """
        return observation[OBS_DICT_KEYS.LAST_ACTION]
