import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from .base_feature_map_space import BaseFeatureMapSpace

from pedsim_agents.utils import SemanticAttribute

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


@SpaceFactory.register("ped_type")
class PedestrianTypeSpace(BaseFeatureMapSpace):
    def __init__(
        self,
        num_ped_types: int,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs
    ) -> None:
        self._num_ped_types = num_ped_types
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs
        )

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=self._num_ped_types,
            shape=(self._feature_map_size * self._feature_map_size,),
            dtype=int,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        return self._get_semantic_map(
            observation[SemanticAttribute.PEDESTRIAN_TYPE.value],
            observation[OBS_DICT_KEYS.ROBOT_POSE],
        )
