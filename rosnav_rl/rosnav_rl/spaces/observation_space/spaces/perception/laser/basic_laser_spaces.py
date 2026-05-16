"""Basic Laser Perception Spaces

Standard laser-based perception spaces integrated into hierarchical architecture.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from ...base_observation_space import BaseObservationSpace
from rosnav_rl.observations.utils.types import (
    LidarRanges,
)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class LaserScanSpace(BaseObservationSpace):
    """Basic laser scan observation space for front-facing lidar perception.

    Provides a production-ready, normalized representation of raw laser scan data from a front-facing lidar sensor.
    Applies range limiting and outputs a fixed-length vector for robust perception and navigation.

    Technical Specifications:
    - Laser Range Limiting: Clamps all values to a configurable maximum range
    - Full Resolution: Uses all available beams from the sensor

    Configuration:
    - laser_num_beams: Number of beams in the laser scan
    - laser_max_range: Maximum valid range for each beam (meters)

    Output Format: 1D numpy array of length laser_num_beams, with all values ∈ [0, laser_max_range].

    Applications: Obstacle avoidance, mapping, and raw sensor fusion.
    """

    name = "LaserScanSpace"
    requires = {
        "front_laser": LidarRanges,  # Front-facing laser scanner range measurements
    }

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

        Returns a new array to avoid mutating the shared collector buffer.

        Args
            laserbeams (np.ndarray): The array of laser beams.
            max_range (float): The maximum range value to apply.

        Returns:
            np.ndarray: A new array with values capped at max_range.
        """
        return np.minimum(laserbeams, max_range)

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> LidarRanges:
        """Encode full-resolution laser scan with range limiting for robust perception.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges from front-facing lidar
                - Shape: (laser_num_beams,) - Full resolution
                - Dtype: np.float32
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, laser_max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (array of distance measurements)

        Returns:
            LidarRanges: Laser scan data with applied range limits.
                - Shape: (laser_num_beams,) - Full resolution laser scan
                - Dtype: np.float32
                - Range: [0.0, laser_max_range] - Distances clamped to max range
                - Units: meters
                - Example: [0.5, 1.2, 3.4, 2.8, 1.9] for 5-beam laser
        """
        return LaserScanSpace.apply_limit(front_laser, self._max_range)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class ReducedLaserScanSpace(BaseObservationSpace):
    """Reduced laser scan observation space for efficient perception.

    Provides a subsampled, range-limited representation of laser scan data by selecting a subset of beams
    at evenly spaced intervals. Reduces computational complexity while maintaining environmental awareness.

    Technical Specifications:
    - Laser Range Limiting: Clamps all values to a configurable maximum range
    - Subsampling: Evenly selects reduced_num_beams from the full scan

    Configuration:
    - laser_num_beams: Number of beams in the original laser scan
    - laser_max_range: Maximum valid range for each beam (meters)
    - reduced_num_beams: Number of beams in the reduced scan

    Output Format: 1D numpy array of length reduced_num_beams, with all values ∈ [0, laser_max_range].

    Applications: Lightweight navigation, embedded systems, and fast obstacle avoidance.
    """

    name = "ReducedLaserScanSpace"
    requires = {
        "front_laser": LidarRanges,  # Front-facing laser scanner range measurements (reduced)
    }

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
        self._cached_indices = None  # Lazily cached
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
        Results are cached after first computation.

        Returns:
            np.ndarray: Array of indices to select from the original laser scan.

        Raises:
            ValueError: If reduced_num_beams is greater than the original num_beams.
        """
        if self._cached_indices is not None:
            return self._cached_indices

        if self._reduced_num_beams > self._num_beams:
            raise ValueError(
                f"Cannot reduce {self._num_beams} beams to {self._reduced_num_beams} beams"
            )

        # Calculate evenly spaced indices
        step = self._num_beams / self._reduced_num_beams
        indices = np.array([int(i * step) for i in range(self._reduced_num_beams)])

        # Ensure we don't exceed array bounds
        indices = np.clip(indices, 0, self._num_beams - 1)

        self._cached_indices = indices
        return indices

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> LidarRanges:
        """Encode reduced laser scan with range limiting and subsampling for efficient perception.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges (full resolution before reduction)
                - Shape: (laser_num_beams,) - Full resolution
                - Dtype: np.float32
                - Units: meters
                - Source: lidar sensor
                - Constraints: ranges ∈ [0, laser_max_range], NaN replaced with max_range
                - Example: [0.5, 1.2, 3.4, ..., 2.1] (will be subsampled to reduced_num_beams)

        Returns:
            LidarRanges: Reduced laser scan data with applied range limits.
                - Shape: (reduced_num_beams,) - Subsampled laser scan
                - Dtype: np.float32
                - Range: [0.0, laser_max_range] - Distances clamped to max range
                - Units: meters
                - Sampling: Evenly spaced indices from full laser scan
                - Example: [0.5, 3.4, 1.9] for 3-beam reduction from 360-beam laser
        """
        # Get full laser scan and apply range limits
        full_laser = LaserScanSpace.apply_limit(front_laser, self._max_range)

        # Get indices for reduction
        indices = self.get_indices()

        # Return reduced laser scan
        return full_laser[indices]
