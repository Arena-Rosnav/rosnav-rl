from typing import Tuple

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace
from ...utils import stack_spaces


@SpaceFactory.register("goal")
class GoalSpace(BaseObservationSpace):
    def __init__(self, goal_max_dist: float = 30, *args, **kwargs) -> None:
        self._max_dist = goal_max_dist
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        _spaces = (
            spaces.Box(low=0, high=self._max_dist, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        )
        return stack_spaces(*_spaces)

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        return observation[OBS_DICT_KEYS.GOAL]
