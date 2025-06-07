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
        flatten: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self._laser_queue = deque()
        self._laser_stack_size = laser_stack_size
        self._default_reward_info = {}
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
                                 Expected to contain 10 frames with 80×9 points each.

        Returns:
            np.ndarray: A 3D array of shape (1, 80, 80) representing the processed laser map.
                       Even rows contain minimum values of laser readings,
                       odd rows contain average values of laser readings.
                       If processing fails, returns an empty map of the same shape.
        Raises:
            No exceptions are raised as they are caught internally and logged as warnings.
        """

        try:
            temp = np.array(laser_queue, dtype=np.float32).flatten()

            # Single reshape for all operations
            reshaped = temp.reshape(10, 80, 9)

            # Pre-allocate output with matching dtype
            scan_avg = np.zeros((20, 80), dtype=np.float32)

            # Vectorized calculations using axis reduction
            scan_avg[::2] = reshaped.min(axis=2)  # Even rows: minima
            scan_avg[1::2] = reshaped.mean(axis=2)  # Odd rows: averages

            # Final transformations
            scan_avg = scan_avg.reshape(1600)
            scan_avg_map = np.tile(scan_avg, (4, 1)).reshape(1, 80, 80)
        except Exception as e:
            print(
                f"[{StackedLaserMapSpace.__name__}]: {e} \n Cannot build laser map. Instead return empty map."
            )
            return np.zeros(self.get_gym_space().shape)

        return scan_avg_map

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
