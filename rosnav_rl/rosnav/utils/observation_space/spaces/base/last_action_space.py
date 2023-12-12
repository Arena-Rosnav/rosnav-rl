from typing import Tuple

import numpy as np
from gymnasium import spaces

from ...observation_space_factory import SpaceFactory
from ...utils import stack_spaces
from ..base_observation_space import BaseObservationSpace


@SpaceFactory.register("last_action")
class LastActionSpace(BaseObservationSpace):
    def __init__(
        self,
        min_linear_vel: float,
        max_linear_vel: float,
        min_angular_vel: float,
        max_angular_vel: float,
        min_translational_vel: float = 0.0,
        max_translational_vel: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        self._min_linear_vel = min_linear_vel
        self._max_linear_vel = max_linear_vel
        self._min_translational_vel = min_translational_vel
        self._max_translational_vel = max_translational_vel
        self._min_angular_vel = min_angular_vel
        self._max_angular_vel = max_angular_vel

    def get_gym_space(self) -> Tuple[spaces.Space, ...]:
        _spaces = (
            spaces.Box(
                low=self._min_linear_vel,
                high=self._max_linear_vel,
                shape=(1,),
                dtype=np.float32,
            ),
            spaces.Box(
                low=self._min_translational_vel,
                high=self._max_translational_vel,
                shape=(1,),
                dtype=np.float32,
            ),
            spaces.Box(
                low=self._min_angular_vel,
                high=self._max_angular_vel,
                shape=(1,),
                dtype=np.float32,
            ),
        )
        return stack_spaces(*_spaces)
