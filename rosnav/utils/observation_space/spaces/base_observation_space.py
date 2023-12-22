from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gym import spaces
from gymnasium import spaces


class BaseObservationSpace(ABC):
    """
    Base class for defining observation spaces in reinforcement learning environments.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._space = self.get_gym_space()

    @property
    def space(self) -> spaces.Space:
        """
        Get the gym.Space object representing the observation space.
        """
        return self._space

    @property
    def shape(self):
        """
        Get the shape of the observation space.
        """
        return self._space.shape

    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        """
        Abstract method to define and return the gym.Space object representing the observation space.
        """
        raise NotImplementedError()

    @abstractmethod
    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        """
        Abstract method to encode the observation into a numpy array.
        """
        raise NotImplementedError()
