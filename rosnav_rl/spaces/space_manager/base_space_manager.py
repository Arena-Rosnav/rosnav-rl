from abc import ABC
from typing import Any, Dict, List, Union

import numpy as np
from rosnav_rl.spaces import (
    ActionSpaceManager,
    EncodedObservationDict,
    ObservationSpaceManager,
)
from rosnav_rl.utils.type import ObservationDict
from rosnav_rl.spaces import BaseObservationSpace, BaseFeatureMapSpace

from gym import spaces


class BaseSpaceManager(ABC):
    """
    BaseSpaceManager is an abstract base class that defines the interface for managing action and
    observation spaces in a reinforcement learning environment.

    Attributes:
        _action_space_manager (ActionSpaceManager): Manages the action space.
        _observation_space_manager (ObservationSpaceManager): Manages the observation space.

    Properties:
        action_space_manager (ActionSpaceManager): Gets the action space manager.
        observation_space_manager (ObservationSpaceManager): Gets the observation space manager.
        observation_space (object): Gets the observation space.
        action_space (object): Gets the action space.

    Methods:
        get_observation_space(): Abstract method to get the observation space.
        get_action_space(): Abstract method to get the action space.
        encode_observation(obs_dict: ObservationDict, *args, **kwargs) -> EncodedObservationDict: Abstract method to encode an observation.
        decode_action(action: np.ndarray) -> np.ndarray: Decodes an action.
    """

    _action_space_manager: ActionSpaceManager
    _observation_space_manager: ObservationSpaceManager

    def __init__(
        self,
        action_space_kwargs: Dict[str, Any],
        observation_space_list: List[Union[BaseObservationSpace, BaseFeatureMapSpace]],
        observation_space_kwargs: Dict[str, Any],
    ):
        self._init_action_space_manager(action_space_kwargs)
        self._init_observation_space_manager(
            observation_space_list, observation_space_kwargs
        )

    @property
    def action_space_manager(self) -> ActionSpaceManager:
        """
        Gets the action space manager.
        Returns:
            object: The action space manager.
        """
        return self._action_space_manager

    @property
    def observation_space_manager(self) -> ObservationSpaceManager:
        """
        Gets the observation space manager.
        Returns:
            object: The observation space manager.
        """
        return self._observation_space_manager

    @property
    def observation_space(self) -> spaces.Dict:
        """
        Gets the observation space.
        Returns:
            object: The observation space.
        """
        return self._observation_space_manager.observation_space

    @property
    def observation_space_list(
        self,
    ) -> List[Union[BaseObservationSpace, BaseFeatureMapSpace]]:
        """
        Gets the observation space list.
        Returns:
            object: The observation space list.
        """
        return self._observation_space_manager.space_list

    @property
    def action_space(self) -> Union[spaces.Dict, spaces.Box]:
        """
        Gets the action space.
        Returns:
            object: The action space.
        """
        return self._action_space_manager.action_space

    @property
    def config(self):
        return {
            "observation": self._observation_space_manager.config,
            "action": self._action_space_manager.config,
        }

    def _init_action_space_manager(self, action_space_kwargs: Dict[str, Any]):
        """
        Initializes the ActionSpaceManager.

        Args:
            action_space_kwargs (Dict[str, Any]): Additional keyword arguments for the action spaces.
        """
        self._action_space_manager = ActionSpaceManager(**action_space_kwargs)

    def _init_observation_space_manager(
        self,
        observation_space_list: List[Union[BaseObservationSpace, BaseFeatureMapSpace]],
        observation_space_kwargs: Dict[str, Any],
    ):
        """
        Initializes the ObservationSpaceManager.

        Args:
            observation_spaces (List[Union[BaseObservationSpace, BaseFeatureMapSpace]]): The list of observation spaces.
            observation_space_kwargs (Dict[str, Any]): Additional keyword arguments for the observation spaces.
        """
        self._observation_space_manager = ObservationSpaceManager(
            space_list=observation_space_list,
            space_kwargs=observation_space_kwargs,
        )

    def encode_observation(
        self, obs_dict: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """
        Encodes the given observation using the observation space manager.

        Args:
            observation: The observation to be encoded.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments to be passed to the observation space manager.

        Returns:
            EncodedObservationDict: The encoded observation.
        """
        return self._observation_space_manager.encode_observation(obs_dict, **kwargs)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        Decodes the given action using the action space manager.

        Args:
            action (np.ndarray): The action to be decoded.

        Returns:
            The decoded action as determined by the action space manager.
        """
        return self._action_space_manager.decode_action(action)
