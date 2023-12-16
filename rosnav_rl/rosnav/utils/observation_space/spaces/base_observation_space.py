from abc import ABC, abstractmethod
from typing import List
from gymnasium import spaces

import numpy as np


class BaseObservationSpace(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._space = self.get_gym_space()

    @property
    def space(self) -> spaces.Space:
        return self._space

    @property
    def shape(self):
        return self._space.shape

    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()
