import numpy as np
from gymnasium import spaces
from rl_utils.utils.observation_collector import LaserCollector, ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("laser")
class ReducedLaserScanSpace(BaseObservationSpace):
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

    name = "REDUCED_LASER"
    required_observations = [LaserCollector]

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
            shape=(1, self._reduced_num_beams),
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
            return laserbeams
        indices = np.round(np.linspace(0, len(laserbeams) - 1, x)).astype(int)[:x]
        if type(laserbeams) == tuple:
            laserbeams = np.array(laserbeams)
        return laserbeams[indices]

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> LaserCollector.data_class:
        """
        Encodes the laser scan observation.

        Args:
            observation (ObservationDict): The observation dictionary.

        Returns:
            ndarray: The encoded laser scan observation.
        """
        return ReducedLaserScanSpace.reduce_laserbeams(
            observation[LaserCollector.name], self._reduced_num_beams
        )[np.newaxis, :]
