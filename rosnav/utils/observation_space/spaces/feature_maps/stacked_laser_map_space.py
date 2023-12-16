from collections import deque

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("stacked_laser_map")
class StackedLaserMapSpace(BaseFeatureMapSpace):
    def __init__(
        self,
        laser_stack_size: int,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._laser_queue = deque()
        self._laser_stack_size = laser_stack_size
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _process_laser_scan(self, laser_scan: np.ndarray, done: bool) -> np.ndarray:
        if len(self._laser_queue) == 0:
            self._reset_laser_stack(laser_scan)

        self._laser_queue.pop()
        self._laser_queue.appendleft(laser_scan)

        laser_map = self._build_laser_map(self._laser_queue)

        if done:
            self._reset_laser_stack(laser_scan)

        return laser_map

    def _build_laser_map(self, laser_queue: deque) -> np.ndarray:
        laser_array = np.array(laser_queue)
        # laserstack list of 10 np.arrays of shape (720,)
        scan_avg = np.zeros((20, self._feature_map_size))
        # horizontal stacking of the pooling operations
        # min pooling over every 9th entry
        scan_avg[::2, :] = np.min(
            laser_array.reshape(10, self._feature_map_size, 9), axis=2
        )
        # avg pooling over every 9th entry
        scan_avg[1::2, :] = np.mean(
            laser_array.reshape(10, self._feature_map_size, 9), axis=2
        )

        scan_avg_map = np.tile(scan_avg.ravel(), 4).reshape(
            (self._feature_map_size, self._feature_map_size)
        )

        return scan_avg_map

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=self._roi_in_m,
            shape=(self._feature_map_size * self._feature_map_size),
            dtype=np.float32,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        return self._process_laser_scan(
            observation[OBS_DICT_KEYS.LASER],
            observation.get(OBS_DICT_KEYS.DONE, False),
        )
