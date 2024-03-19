from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gym import spaces
from gymnasium import spaces

from ..normalization import *


class BaseObservationSpace(ABC):
    """
    Base class for defining observation spaces in reinforcement learning environments.
    """

    def __init__(
        self,
        normalize: bool = False,
        norm_func: str = "max_abs_scaling",
        *args,
        **kwargs,
    ) -> None:
        self._space = self.get_gym_space()
        self._normalize = normalize
        self._setup_normaliization(normalize, norm_func)

    def _setup_normaliization(self, normalize: bool, norm_func: str):
        try:
            self._norm_func = (
                eval(norm_func) if normalize and norm_func else lambda x: x
            )
        except ImportError:
            print(
                f"Error: Failed to import normalization function '{norm_func}'. Identity function will be used."
            )
            self._norm_func = lambda x, *args, **kwargs: x

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

    @staticmethod
    def apply_normalization(func):
        def wrapper(
            self: BaseObservationSpace, observation: dict, *args, **kwargs
        ) -> np.ndarray:
            """
            Apply max absolute scaling to the observation array.
            """
            observation_arr = func(self, observation, **kwargs)
            if self._normalize:
                return self._norm_func(observation_arr, self.space.low, self.space.high)
            return observation_arr

        return wrapper
