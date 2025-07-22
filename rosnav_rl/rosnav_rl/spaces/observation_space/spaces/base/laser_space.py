from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import LaserCollector
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("laser")
class LaserScanSpace(BaseObservationSpace):
    """
    Represents the observation space for laser scan data.

    Args:
        laser_num_beams (int): The number of laser beams.
        laser_max_range (float): The maximum range of the laser.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _num_beams (int): The number of laser beams.
        _max_range (float): The maximum range of the laser.
    """

    name = "LASER"
    required_observation_units = [LaserCollector]

    def __init__(
        self, laser_num_beams: int, laser_max_range: float, *args, **kwargs
    ) -> None:
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for laser scan data.

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=0,
            high=self._max_range,
            shape=(self._num_beams,),
            dtype=np.float32,
        )

    @staticmethod
    def apply_limit(laserbeams: np.ndarray, max_range: float) -> np.ndarray:
        """
        Applies a limit to the laser beams, setting values greater than max_range to max_range.

        Args
            laserbeams (np.ndarray): The array of laser beams.
            max_range (float): The maximum range value to apply.

        Returns:
            np.ndarray: The modified laser beams with values capped at max_range.
        """
        laserbeams[laserbeams > max_range] = max_range
        return laserbeams

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> Any:  # Intended: LaserCollector.data_class
        """
        Extracts laser scan data from the observation dictionary.

        Args:
            observation (ObservationDict): A dictionary containing observation data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            LaserCollector.data_class: The laser scan data extracted from the observation dictionary.
        """
        return LaserScanSpace.apply_limit(
            observation[LaserCollector.name], self._max_range
        )
