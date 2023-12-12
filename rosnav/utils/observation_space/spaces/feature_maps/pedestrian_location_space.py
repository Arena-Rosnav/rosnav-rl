import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("ped_location")
class PedestrianLocationSpace(BaseObservationSpace):
    def __init__(self, feature_map_size: int, *args, **kwargs) -> None:
        self._map_size = feature_map_size

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self._map_size * self._map_size,),
            dtype=int,
        )
