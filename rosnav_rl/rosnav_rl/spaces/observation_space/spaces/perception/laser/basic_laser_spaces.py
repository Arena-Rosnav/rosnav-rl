"""Basic Laser Perception Spaces

Standard laser-based perception spaces integrated into hierarchical architecture.
"""

from typing import Any
import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import LaserCollector
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from ...base_observation_space import BaseObservationSpace


@SpaceFactory.register("laser", SpaceCategory.PERCEPTION)
class LaserScanSpace(BaseObservationSpace):
    """
    Original laser scan observation space.

    Represents the observation space for laser scan data with basic range limiting.

    Args:
        laser_num_beams (int): The number of laser beams.
        laser_max_range (float): The maximum range of the laser.
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
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Extracts laser scan data from the observation dictionary.

        Args:
            observation (ObservationDict): A dictionary containing observation data.

        Returns:
            Laser scan data with applied range limits.
        """
        return LaserScanSpace.apply_limit(
            observation[LaserCollector.name], self._max_range
        )


@SpaceFactory.register("reduce_laser", SpaceCategory.PERCEPTION)
class ReducedLaserScanSpace(BaseObservationSpace):
    """A class representing a reduced laser scan observation space.

    This observation space reduces the dimensionality of laser scan data by
    selecting a subset of the original laser beams at evenly spaced intervals.
    This can be useful for reducing computational complexity while still
    maintaining sufficient environmental awareness for navigation tasks.
    """

    name = "REDUCED_LASER"
    required_observation_units = [LaserCollector]

    def __init__(
        self,
        laser_num_beams: int,
        laser_max_range: float,
        reduced_num_beams: int,
        *args,
        **kwargs,
    ) -> None:
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range
        self._reduced_num_beams = reduced_num_beams
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the Gym observation space for reduced laser scan data.

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=0,
            high=self._max_range,
            shape=(self._reduced_num_beams,),
            dtype=np.float32,
        )

    def get_indices(self) -> np.ndarray:
        """
        Calculates the indices for selecting laser beams for reduction.

        This method determines which indices from the original laser scan
        should be used to create the reduced representation. The indices
        are evenly distributed across the full range of laser beams.

        Returns:
            np.ndarray: Array of indices to select from the original laser scan.

        Raises:
            ValueError: If reduced_num_beams is greater than the original num_beams.
        """
        if self._reduced_num_beams > self._num_beams:
            raise ValueError(
                f"Cannot reduce {self._num_beams} beams to {self._reduced_num_beams} beams"
            )

        # Calculate evenly spaced indices
        step = self._num_beams / self._reduced_num_beams
        indices = np.array([int(i * step) for i in range(self._reduced_num_beams)])

        # Ensure we don't exceed array bounds
        indices = np.clip(indices, 0, self._num_beams - 1)

        return indices

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: ObservationDict, *args, **kwargs) -> Any:
        """
        Encodes a reduced version of the laser scan observation.

        Args:
            observation (ObservationDict): A dictionary containing observation data.

        Returns:
            Reduced laser scan data with applied range limits.
        """
        # Get full laser scan and apply range limits
        full_laser = LaserScanSpace.apply_limit(
            observation[LaserCollector.name], self._max_range
        )

        # Get indices for reduction
        indices = self.get_indices()

        # Return reduced laser scan
        return full_laser[indices]
