import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("stacked_laser_map")
class StackedLaserMapSpace(BaseObservationSpace):
    def __init__(self, roi_in_m: float, feature_map_size: int, *args, **kwargs) -> None:
        self._map_size = feature_map_size
        self._roi_in_m = roi_in_m

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=self._roi_in_m,
            shape=(self._map_size * self._map_size,),
            dtype=np.float32,
        )
