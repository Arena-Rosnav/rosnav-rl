import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("laser")
class LaserScanSpace(BaseObservationSpace):
    def __init__(
        self, laser_num_beams: int, laser_max_range: float, *args, **kwargs
    ) -> None:
        self._num_beams = laser_num_beams
        self._max_range = laser_max_range

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=self._max_range,
            shape=(self._num_beams,),
            dtype=np.float32,
        )