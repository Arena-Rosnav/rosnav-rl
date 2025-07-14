from collections import deque

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations import (
    DoneObservation,
    LaserCollector,
)
from rosnav_rl.utils.type_aliases import ObservationDict

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from .base_feature_map_space import BaseFeatureMapSpace

from ..base.laser_space import LaserScanSpace


@SpaceFactory.register("stacked_laser_map")
class StackedLaserMapSpace(BaseFeatureMapSpace):
    """A feature map space that stacks laser scan data to create a 2D representation of the environment.

    This class processes laser scan data by maintaining a queue of consecutive scans and
    transforming them into a feature map representation. The resulting map provides
    spatial information about the robot's surroundings based on laser readings.

    Attributes:
        name (str): The identifier for this observation space ("STACKED_LASER_MAP").
        required_observation_units (list): List of required collectors, specifically LaserCollector.
        laser_stack_size (int): Number of consecutive laser scans to stack.
        feature_map_size (int): Size of the output feature map (both width and height).
        roi_in_m (float): Region of interest in meters, also used as the upper bound for the gym space.
        flatten (bool, optional): Whether to flatten the output feature map. Defaults to True.
        *args: Additional positional arguments for the parent class.
        **kwargs: Additional keyword arguments for the parent class.

    Notes:
        - The feature map has shape (1, feature_map_size, feature_map_size)
        - Laser scans are processed to extract both minimum and average values
        - The map represents spatial information around the robot's position
    """

    name = "STACKED_LASER_MAP"
    required_observation_units = [LaserCollector]

    def __init__(
        self,
        laser_stack_size: int,
        feature_map_size: int,
        roi_in_m: float,
        laser_max_range: float,
        flatten: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self._laser_queue = deque()
        self._laser_stack_size = laser_stack_size
        self._laser_max_range = laser_max_range
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        """
        Resets the laser stack with zeros.

        Args:
            laser_scan (np.ndarray): The laser scan.

        """
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _process_laser_scan(
        self, laser_scan: LaserCollector.data_class, done: DoneObservation.data_class
    ) -> np.ndarray:
        """Process laser scan data and build a stacked laser map.

        This method handles the processing of laser scan data to maintain a queue of scans
        and build a map representation. If no laser scan is available or the environment is reset,
        appropriate actions are taken.

        Args:
            laser_scan (LaserCollector.data_class): The laser scan data, expected as a numpy array
            done (DoneObservation.data_class): Flag indicating if the episode is done

        Returns:
            np.ndarray: The processed laser map or zeros array if no valid scan is available

        Note:
            - If laser_scan is not a numpy array, returns a zero-filled array
            - If queue is empty or done flag is True, resets the laser stack
            - Maintains a queue of laser scans for stacking
        """
        if type(laser_scan) is not np.ndarray:
            return np.zeros(self.get_gym_space().shape)

        laser_scan = LaserScanSpace.apply_limit(laser_scan, self._laser_max_range)

        if len(self._laser_queue) == 0 or done:
            self._reset_laser_stack(laser_scan)

        self._laser_queue.pop()
        self._laser_queue.appendleft(laser_scan)

        laser_map = self._build_laser_map(self._laser_queue)

        return laser_map

    def _build_laser_map(self, laser_queue: deque) -> np.ndarray:
        """Builds a laser map from a queue of laser scans.

        This method processes laser scan data stored in a deque structure and transforms it into
        a feature map representation. The processing includes reshaping the data, calculating
        minimum and average values, and formatting the result as a map.

        Args:
            laser_queue (deque): A queue containing laser scan data frames.
                                 Expected to contain `_laser_stack_size` frames, each with a number of points
                                 divisible by `_feature_map_size`.

        Returns:
            np.ndarray: A 3D array of shape (1, `_feature_map_size`, `_feature_map_size`)
                        representing the processed laser map.
                       Even rows contain minimum values of laser readings,
                       odd rows contain average values of laser readings.
                       If processing fails, returns an empty map of the same shape.
        Raises:
            No exceptions are raised as they are caught internally and logged as warnings.
        """

        try:
            laser_scans_array = np.array(laser_queue, dtype=np.float32)

            # Reshape to group laser points for feature extraction.
            # e.g., (10, 720) -> (10, 80, 9)
            grouped_scans = laser_scans_array.reshape(
                self._laser_stack_size, self._feature_map_size, -1
            )

            # Calculate min and mean for each group of points.
            min_features = grouped_scans.min(axis=2)
            mean_features = grouped_scans.mean(axis=2)

            # Interleave min and mean features more concisely using np.stack and reshape.
            # This creates a (20, 80) array where even rows are min and odd rows are mean.
            interleaved_features = np.stack(
                (min_features, mean_features), axis=1
            ).reshape(self._laser_stack_size * 2, self._feature_map_size)

            # Tile the features vertically to create a square map.
            # e.g., tile (20, 80) array 4 times to get (80, 80).
            num_tiles = self._feature_map_size // interleaved_features.shape[0]
            tiled_features = np.tile(interleaved_features, (num_tiles, 1))

            # Reshape to the final (1, size, size) feature map.
            feature_map = tiled_features.reshape(
                1, self._feature_map_size, self._feature_map_size
            )

        except Exception as e:
            print(
                f"[{StackedLaserMapSpace.__name__}]: {e} \n Cannot build laser map. Instead return empty map."
            )
            return np.zeros(self.get_gym_space().shape)

        return feature_map

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym space for the feature map.

        Returns:
            spaces.Space: The gym space.

        """
        return spaces.Box(
            low=0,
            high=self._roi_in_m,
            shape=(1, self._feature_map_size, self._feature_map_size),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Encodes the observation into a feature map.

        Args:
            observation (ObservationDict): The observation dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ndarray: The encoded feature map.

        """
        return self._process_laser_scan(
            observation[LaserCollector.name],
            observation.get(DoneObservation.name, False),
        )
