from collections import deque
from typing import List, Tuple

import numpy as np
from pedsim_msgs.msg import SemanticDatum
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import (
    OBS_SPACE_TO_OBS_DICT_KEY,
    SPACE_FACTORY_KEYS,
)
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory


@BaseSpaceEncoderFactory.register("SemanticResNetSpaceEncoder")
class SemanticResNetSpaceEncoder(DefaultEncoder):
    def __init__(
        self,
        radius: float,
        is_holonomic: bool,
        actions: dict,
        is_action_space_discrete: bool,
        feature_map_size: int = 80,
        roi_in_m: int = 20,
        laser_stack_size: int = 10,
        stacked: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            radius=radius,
            is_holonomic=is_holonomic,
            actions=actions,
            is_action_space_discrete=is_action_space_discrete,
            stacked=stacked,
            **kwargs
        )
        self._laser_queue = deque()
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._laser_stack_size = laser_stack_size

        self._observation_space_manager = ObservationSpaceManager(
            [
                SPACE_FACTORY_KEYS.STACKED_LASER_MAP.name,
                SPACE_FACTORY_KEYS.PEDESTRIAN_LOCATION.name,
                SPACE_FACTORY_KEYS.PEDESTRIAN_TYPE.name,
                SPACE_FACTORY_KEYS.GOAL.name,
            ],
            enable_frame_stacking=self._stacked,
            space_kwargs={
                "roi_in_m": self._roi_in_m,
                "feature_map_size": self._feature_map_size,
                "goal_max_dist": 20,
                **kwargs,
            },
        )

    def get_observation_space(self):
        return self._observation_space_manager.unified_observation_space

    def encode_observation(self, observation: dict, structure: list) -> np.ndarray:
        laser_map = self._process_laser_scan(
            observation[OBS_DICT_KEYS.LASER], observation.get("is_done", False)
        )

        observation_maps = [
            self._get_semantic_map(
                observation[OBS_SPACE_TO_OBS_DICT_KEY[space]],
                observation[OBS_DICT_KEYS.ROBOT_POSE],
            )
            for space in self._observation_space_manager.space_list
        ]

        return np.concatenate(
            [laser_map.flatten()]
            + [obs_map.flatten() for obs_map in observation_maps]
            + [observation[name] for name in structure if name not in ["last_action"]],
            axis=0,
        )

    @staticmethod
    def get_relative_pos(reference_frame, distant_frame) -> tuple:
        return (
            distant_frame.x - reference_frame.x,
            distant_frame.y - reference_frame.y,
            distant_frame.z - reference_frame.z,
        )

    def _get_map_index(self, position: tuple) -> tuple:
        x, y, *_ = position
        x = int((x / self._roi_in_m) * self._feature_map_size) + (
            self._feature_map_size // 2 - 1
        )
        y = int((y / self._roi_in_m) * self._feature_map_size) + (
            self._feature_map_size // 2 - 1
        )
        x = min(max(x, 0), self._feature_map_size - 1)
        y = min(max(y, 0), self._feature_map_size - 1)
        return x, y

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _build_laser_map(self, laser_queue) -> np.ndarray:
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
        # scan avg map shape (80, 80)
        return scan_avg_map

    def _process_laser_scan(self, laser_scan: np.ndarray, done: bool) -> np.ndarray:
        if len(self._laser_queue) == 0:
            self._reset_laser_stack(laser_scan)

        self._laser_queue.pop()
        self._laser_queue.appendleft(laser_scan)

        laser_map = self._build_laser_map(self._laser_queue)

        if done:
            self._reset_laser_stack(laser_scan)

        return laser_map

    def _get_semantic_map(
        self, semantic_data: List[SemanticDatum], robot_pose
    ) -> np.ndarray:
        pos_map = np.zeros((self._feature_map_size, self._feature_map_size))
        map_size = pos_map.shape[0]

        for data in semantic_data:
            relative_pos = SemanticResNetSpaceEncoder.get_relative_pos(
                data.location, robot_pose
            )
            index = self._get_map_index(relative_pos)
            if 0 <= index[0] < map_size and 0 <= index[1] < map_size:
                pos_map[index] = data.evidence

        return pos_map
