import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from .base_feature_map_space import BaseFeatureMapSpace

from pedsim_agents.utils import SemanticAttribute

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


@SpaceFactory.register("ped_vel_x")
class PedestrianVelXSpace(BaseFeatureMapSpace):
    def __init__(
        self,
        min_speed_x: float,
        max_speed_x: float,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._map_size = feature_map_size
        self._min_speed = min_speed_x
        self._max_speed = max_speed_x
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=self._min_speed,
            high=self._max_speed,
            shape=(self._map_size * self._map_size,),
            dtype=float,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        return self._get_semantic_map(
            observation[SemanticAttribute.PEDESTRIAN_VEL_X.value],
            observation[OBS_DICT_KEYS.ROBOT_POSE],
        )
