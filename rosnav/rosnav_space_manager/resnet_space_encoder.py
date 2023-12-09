from collections import deque
from typing import List, Tuple

import numpy as np
from gymnasium import spaces
from pedsim_msgs.msg import SemanticDatum

from ..utils.utils import stack_spaces
from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory

from pedsim_agents.utils import SemanticAttribute


@BaseSpaceEncoderFactory.register("ResNetEncoder")
class ResNetSpaceEncoder(DefaultEncoder):
    feature_map_observation_space = {
        SemanticAttribute.IS_PEDESTRIAN: lambda map_size: spaces.Box(
            low=0,
            high=1,
            shape=(map_size * map_size,),
            dtype=int,
        ),
        SemanticAttribute.PEDESTRIAN_TYPE: lambda map_size: spaces.Box(
            low=0,
            high=5,
            shape=(map_size * map_size,),
            dtype=int,
        ),
        SemanticAttribute.PEDESTRIAN_VEL_X: lambda map_size: spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(map_size * map_size,),
            dtype=np.float32,
        ),
        SemanticAttribute.PEDESTRIAN_VEL_Y: lambda map_size: spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(map_size * map_size,),
            dtype=np.float32,
        ),
    }

    def __init__(
        self,
        laser_num_beams,
        laser_max_range,
        radius,
        is_holonomic,
        actions,
        is_action_space_discrete,
        feature_map_size: int = 80,
        roi_in_m: int = 20,
        laser_stack_size: int = 10,
    ):
        super().__init__(
            laser_num_beams,
            laser_max_range,
            radius,
            is_holonomic,
            actions,
            is_action_space_discrete,
        )
        self._laser_queue = deque()
        self._feature_map_size = feature_map_size
        self._roi_in_m = roi_in_m
        self._grid_center = self._feature_map_size // 2
        self._laser_stack_size = laser_stack_size

        self._semantic_info = [
            SemanticAttribute.IS_PEDESTRIAN,
            SemanticAttribute.PEDESTRIAN_TYPE,
            SemanticAttribute.PEDESTRIAN_VEL_X,
            SemanticAttribute.PEDESTRIAN_VEL_Y,
        ]

    def get_observation_space(self):
        return stack_spaces(
            spaces.Box(
                low=0,
                high=self._roi_in_m,
                shape=(self._feature_map_size * self._feature_map_size,),
                dtype=int,
            ),
            # semantic info
            *(
                ResNetSpaceEncoder.feature_map_observation_space[
                    SemanticAttribute[obs_space.name]
                ](self._feature_map_size)
                for obs_space in self._semantic_info
            ),
            # goal distance
            spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
            # goal angle
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            # linear vel
            spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(2,),
                dtype=np.float32,
            ),
            # angular vel
            spaces.Box(
                low=-4.0,
                high=4.0,
                shape=(1,),
                dtype=np.float32,
            ),
        )

    def encode_observation(self, observation: dict, structure: list) -> np.ndarray:
        laser_map = self._process_laser_scan(
            observation["laser_scan"], observation.get("is_done", False)
        )

        observation_maps = [
            self._get_semantic_map(observation[sem_info.value], observation["odom"])
            for sem_info in self._semantic_info
        ]

        return np.concatenate(
            [laser_map.flatten()]
            + [obs_map.flatten() for obs_map in observation_maps]
            + [observation[name] for name in structure if name != "laser_scan"],
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
        x = int((x / self._roi_in_m + 0.5) * self._feature_map_size)
        y = int((y / self._roi_in_m + 0.5) * self._feature_map_size)
        x = min(max(x, 0), self._feature_map_size - 1)
        y = min(max(y, 0), self._feature_map_size - 1)
        return x, y

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _build_laser_map(self, laser_queue) -> np.ndarray:
        laser_array = np.array(laser_queue)
        # laserstack list of 10 np.arrays of shape (720,)
        scan_avg = np.zeros((20, 80))
        # horizontal stacking of the pooling operations
        # min pooling over every 9th entry
        scan_avg[::2, :] = np.min(laser_array.reshape(10, 80, 9), axis=2)
        # avg pooling over every 9th entry
        scan_avg[1::2, :] = np.mean(laser_array.reshape(10, 80, 9), axis=2)

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
            relative_pos = ResNetSpaceEncoder.get_relative_pos(
                data.location, robot_pose
            )
            index = self._get_map_index(np.array(relative_pos))
            if 0 <= index[0] < map_size and 0 <= index[1] < map_size:
                pos_map[tuple(index)] = data.evidence

        return pos_map
