from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces

from ..base_space_encoder import BaseSpaceEncoder
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)


class BaseEncoderWrapper(ABC):
    def __init__(self, encoder: BaseSpaceEncoder) -> None:
        self._encoder = encoder

    @property
    def observation_space(self) -> spaces.Box:
        """
        Get the observation space.

        Returns:
            spaces.Box: The observation space.
        """
        return self._encoder.observation_space

    @property
    def action_space(self) -> spaces.Box:
        """
        Get the action space.

        Returns:
            spaces.Box: The action space.
        """
        return self._encoder.action_space

    @property
    def observation_space_manager(self) -> ObservationSpaceManager:
        """
        Get the observation space manager.

        Returns:
            ObservationSpaceManager: The observation space manager.
        """
        return self._encoder.observation_space_manager

    @property
    def observation_list(self):
        """
        Gets the list of observation spaces.

        Returns:
            List[SPACE_INDEX]: The list of observation spaces.
        """
        return self._encoder._observation_list

    @property
    def observation_kwargs(self):
        """
        Gets the keyword arguments for configuring the observation space manager.

        Returns:
            dict: The keyword arguments for configuring the observation space manager.
        """
        return self._encoder._observation_kwargs

    def encode_observation(self, observation: dict, *args, **kwargs) -> np.ndarray:
        return self._encoder.encode_observation(observation, *args, **kwargs)

    def decode_action(self, action) -> np.ndarray:
        return self._encoder.decode_action(action)
