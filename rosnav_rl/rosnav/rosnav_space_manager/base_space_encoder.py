from typing import List

import numpy as np
from gymnasium import spaces

from ..utils.action_space.action_space_manager import ActionSpaceManager
from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import SPACE_INDEX


class BaseSpaceEncoder:
    """
    Base class for space encoders used in ROS navigation.
    """

    def __init__(
        self,
        action_space_kwargs: dict,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
        stacked_observation: bool = False,
        *args,
        **kwargs
    ):
        """
        Initializes a new instance of the DefaultEncoder class.

        Args:
            action_space_kwargs (dict): Keyword arguments for configuring the action space manager.
            observation_list (List[SPACE_INDEX], optional): List of observation spaces to include. Defaults to None.
            observation_kwargs (dict, optional): Keyword arguments for configuring the observation space manager. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._observation_list = observation_list
        self._observation_kwargs = observation_kwargs
        self._stacked_observation = stacked_observation

        self.setup_action_space(action_space_kwargs)
        self.setup_observation_space(
            stacked_observation, observation_list, observation_kwargs
        )

    @property
    def observation_space(self) -> spaces.Space:
        """
        Gets the observation space.

        Returns:
            spaces.Space: The observation space.
        """
        return self._observation_space_manager.observation_space

    @property
    def action_space(self) -> spaces.Space:
        """
        Gets the action space.

        Returns:
            spaces.Space: The action space.
        """
        return self._action_space_manager.action_space

    @property
    def action_space_manager(self):
        """
        Gets the action space manager.

        Returns:
            ActionSpaceManager: The action space manager.
        """
        return self._action_space_manager

    @property
    def observation_space_manager(self):
        """
        Gets the observation space manager.

        Returns:
            ObservationSpaceManager: The observation space manager.
        """
        return self._observation_space_manager

    @property
    def observation_list(self):
        """
        Gets the list of observation spaces.

        Returns:
            List[SPACE_INDEX]: The list of observation spaces.
        """
        return self._observation_list

    @property
    def observation_kwargs(self):
        """
        Gets the keyword arguments for configuring the observation space manager.

        Returns:
            dict: The keyword arguments for configuring the observation space manager.
        """
        return self._observation_kwargs

    def setup_action_space(self, action_space_kwargs: dict):
        """
        Sets up the action space manager.

        Args:
            action_space_kwargs (dict): Keyword arguments for configuring the action space manager.
        """
        self._action_space_manager = ActionSpaceManager(**action_space_kwargs)

    def setup_observation_space(
        self,
        stacked_observation: bool,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
    ):
        """
        Sets up the observation space manager.

        Args:
            observation_list (List[SPACE_INDEX], optional): List of observation spaces to include. Defaults to None.
            observation_kwargs (dict, optional): Keyword arguments for configuring the observation space manager. Defaults to None.
        """
        self._observation_space_manager = ObservationSpaceManager(
            observation_list,
            space_kwargs=observation_kwargs,
            frame_stacking=stacked_observation,
        )

    def decode_action(self, action) -> np.ndarray:
        """
        Decodes the action.

        Args:
            action: The action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        return self._action_space_manager.decode_action(action)

    def encode_observation(self, observation, *args, **kwargs) -> np.ndarray:
        """
        Encodes the observation.

        Args:
            observation: The observation to encode.

        Returns:
            np.ndarray: The encoded observation.
        """
        return self._observation_space_manager.encode_observation(observation, **kwargs)
