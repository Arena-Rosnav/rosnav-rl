from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List, Union
from warnings import warn

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.type_aliases import (
    ObservationCollector,
    ObservationDict,
    ObservationGenerator,
)

from ..normalization import get_normalizer, Normalizer


class BaseObservationSpace(ABC):
    """An abstract base class for observation spaces in reinforcement learning environments.

    This class defines the interface for observation spaces and provides common
    functionality for normalization and validation of observations. It is designed to be
    extended by concrete observation space implementations.

    Attributes:
        name (ClassVar[str]): The name of the observation space.
        required_observation_units (ClassVar[List[Union[ObservationCollector, ObservationGenerator]]]):
            List of observation collectors or generators required by this observation space.

    Parameters:
        normalize (bool, optional): Whether to normalize observations. Defaults to False.
        normalizer (str, optional): Name of the normalizer to use. Defaults to "max_abs".
            Available normalizers: "max_abs", "min_max", "standard", "identity", "none"
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments (passed to normalizer if applicable).

    Methods:
        get_gym_space(): Define and return the gym.Space object representing the observation space.
        encode_observation(observation, *args, **kwargs): Encode the observation into a numpy array.

    Properties:
        config: Returns the configuration parameters.
        space: Get the gym.Space object representing the observation space.
        shape: Get the shape of the observation space.

    Decorators:
        apply_normalization: Decorator to apply normalization to observation arrays.
        check_dtype: Decorator to validate observation array data types.
    """

    name: ClassVar[str]
    required_observation_units: ClassVar[
        List[Union[ObservationCollector, ObservationGenerator]]
    ] = []

    def __init__(
        self,
        normalize: bool = False,
        normalizer: str = "max_abs",
        *args,
        **kwargs,
    ) -> None:
        self._space = self.get_gym_space()
        self._normalize = normalize
        self._normalizer = self._setup_normalizer(normalize, normalizer, **kwargs)

        self.__params__ = {
            "normalize": normalize,
            "normalizer": normalizer,
            "args": args,
            **kwargs,
        }

    def __repr__(self):
        return f"{self.name}"

    def _setup_normalizer(
        self, normalize: bool, normalizer_name: str, **kwargs
    ) -> Normalizer:
        """Set up the normalizer instance."""
        if not normalize:
            return get_normalizer("identity")

        try:
            return get_normalizer(normalizer_name, **kwargs)
        except ValueError as e:
            warn(
                f"Error setting up normalizer '{normalizer_name}': {e}. Using identity normalizer."
            )
            return get_normalizer("identity")

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

    def _apply_normalization(self, observation_arr: np.ndarray) -> np.ndarray:
        """Apply normalization to observation array if enabled."""
        if self._normalize:
            return self._normalizer.normalize(
                observation_arr, self.space.low, self.space.high
            )
        return observation_arr

    def _validate_observation(self, observation_arr: np.ndarray) -> np.ndarray:
        """Validate and fix observation array issues."""
        if (
            not np.isfinite(observation_arr).all()
            or not np.isreal(observation_arr).all()
        ):
            warn(f"[{self.name}] Invalid observation array: {observation_arr}")
            observation_arr = np.zeros_like(observation_arr, dtype=np.float32)
        return observation_arr

    @staticmethod
    def apply_normalization(func):
        """Decorator to apply normalization after observation encoding."""

        def wrapper(
            self: BaseObservationSpace, observation: ObservationDict, *args, **kwargs
        ) -> np.ndarray:
            observation_arr = func(self, observation, **kwargs)
            return self._apply_normalization(observation_arr)

        return wrapper

    @staticmethod
    def check_dtype(func):
        """Decorator to validate observation array data types."""

        def wrapper(
            self: BaseObservationSpace, observation: ObservationDict, *args, **kwargs
        ) -> np.ndarray:
            observation_arr = func(self, observation, **kwargs)
            return self._validate_observation(observation_arr)

        return wrapper
