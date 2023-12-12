import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("ped_type")
class PedestrianTypeSpace(BaseObservationSpace):
    def __init__(self, feature_map_size: int, num_types: int, *args, **kwargs) -> None:
        self._map_size = feature_map_size
        self._num_types = num_types

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=self._num_types - 1,
            shape=(self._map_size * self._map_size,),
            dtype=int,
        )
