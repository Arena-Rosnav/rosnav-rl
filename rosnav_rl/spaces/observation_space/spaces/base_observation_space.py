import inspect
from abc import ABC, abstractmethod
from typing import List, Set, Type, TypeVar, Union

import numpy as np
from gym import spaces
from gymnasium import spaces
from rl_utils.utils.observation_collector import *
from rosnav_rl.spaces.observation_space.normalization import *

ObservationCollector = TypeVar("ObservationCollector", bound=ObservationCollectorUnit)
ObservationGenerator = TypeVar("ObservationGenerator", bound=ObservationGeneratorUnit)

from warnings import warn


class BaseObservationSpace(ABC):
    """
    Base class for defining observation spaces in reinforcement learning environments.
    """

    name: str = "BASE_OBSERVATION_SPACE"
    required_observations: List[Union[ObservationCollector, ObservationGenerator]] = []

    def __init__(
        self,
        normalize: bool = False,
        norm_func: str = "max_abs_scaling",
        *args,
        **kwargs,
    ) -> None:
        self._space = self.get_gym_space()
        self._normalize = normalize
        self._setup_normalization(normalize, norm_func)

        self.__params__ = {
            "normalize": normalize,
            "norm_func": norm_func,
            "args": args,
            **kwargs,
        }

    def __repr__(self):
        return f"{self.name}"

    def _setup_normalization(self, normalize: bool, norm_func: str):
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
    def config(self):
        """
        Returns the configuration parameters.

        Returns:
            dict: The configuration parameters stored in the __params__ attribute.
        """
        return self.__params__

    @property
    def space(self) -> spaces.Space:
        """
        Get the gym.Space object representing the observation space.
        """
        return self._space

    @property
    def shape(self) -> dict:
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
    def encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> np.ndarray:
        """
        Abstract method to encode the observation into a numpy array.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_normalization(func):
        def wrapper(
            self: BaseObservationSpace, observation: ObservationDict, *args, **kwargs
        ) -> np.ndarray:
            """
            Apply max absolute scaling to the observation array.
            """
            observation_arr = func(self, observation, **kwargs)
            if self._normalize:
                return self._norm_func(observation_arr, self.space.low, self.space.high)
            return observation_arr

        return wrapper

    @staticmethod
    def check_dtype(func):
        def wrapper(
            self: BaseObservationSpace, observation: ObservationDict, *args, **kwargs
        ) -> np.ndarray:
            """
            Apply max absolute scaling to the observation array.
            """
            observation_arr = func(self, observation, **kwargs)

            if (
                not np.isfinite(observation_arr).all()
                or not np.isreal(observation_arr).all()
            ):
                warn(f"[{self.name}] Invalid observation array: {observation_arr}")
                observation_arr = np.zeros_like(observation_arr, dtype=np.float32)

            # Check if the observation array is of the correct dtype
            # if not np.issubdtype(observation_arr.dtype, np.float32):
            #     observation_arr = observation_arr.astype(np.float32)
            return observation_arr

        return wrapper