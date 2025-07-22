from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import LaserCollector
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .laser_space import LaserScanSpace


@SpaceFactory.register("reduce_laser")
class ReducedLaserScanSpace(BaseObservationSpace):
    """A class representing a reduced laser scan observation space.

    This observation space reduces the dimensionality of laser scan data by
    selecting a subset of the original laser beams at evenly spaced intervals.
    This can be useful for reducing computational complexity while still
    maintaining sufficient environmental awareness for navigation tasks.

    Attributes:
        name (str): Identifier for this observation space type, set to "REDUCED_LASER".
        required_observation_units (list): List of required collectors, only LaserCollector needed.
        _num_beams (int): The number of beams in the original laser scan.
        _max_range (float): The maximum range value for laser measurements.
        _reduced_num_beams (int): The target number of beams after reduction.
    """

    name = "REDUCED_LASER"
    required_observation_units = [LaserCollector]

    def __init__(
        self,
        laser_num_beams: int,
        laser_max_range: float,
        reduced_num_beams: int,
        *args,
        **kwargs
    ) -> None:
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        self._reduced_num_beams = reduced_num_beams
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
            shape=(self._reduced_num_beams,),
            dtype=np.float32,
        )

    @staticmethod
    def reduce_laserbeams(laserbeams: np.ndarray, x: int) -> np.ndarray:
        """
        Reduces the number of laser beams in the given laser scan.

        Args:
            laserbeams (np.ndarray): The laser scan.
            x (int): The number of reduced laser beams.

        Returns:
            np.ndarray: The reduced laser scan.

        """
        if x >= len(laserbeams):
            return np.asarray(laserbeams)

        indices = np.linspace(0, len(laserbeams) - 1, x, dtype=int)
        return np.asarray(laserbeams)[indices]

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> Any:  # Intended: ReducedLaserGenerator.data_class
        """
        Encodes the laser scan observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            ndarray: The encoded laser scan observation.
        """
        return ReducedLaserScanSpace.reduce_laserbeams(
            LaserScanSpace.apply_limit(
                observation[LaserCollector.name], self._max_range
            ),
            self._reduced_num_beams,
        )
