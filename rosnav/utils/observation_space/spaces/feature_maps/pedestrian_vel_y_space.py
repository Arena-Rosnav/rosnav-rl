from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("ped_vel_y")
class PedestrianVelYSpace(BaseObservationSpace):
    def __init__(
        self,
        feature_map_size: int,
        min_speed_y: float,
        max_speed_y: float,
        *args,
        **kwargs
    ) -> None:
        self._map_size = feature_map_size
        self._min_speed = min_speed_y
        self._max_speed = max_speed_y

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=self._min_speed,
            high=self._max_speed,
            shape=(self._map_size * self._map_size,),
            dtype=float,
        )
