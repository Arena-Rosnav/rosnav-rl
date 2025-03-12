from dataclasses import asdict
from typing import Any, Dict, List, Union

import numpy as np
from gym import spaces

from ...spaces import (
    ActionSpaceManager,
    ObservationSpaceManager,
)
from ...states import AgentStateContainer
from ...utils.type_aliases import (
    EncodedObservationDict,
    ObservationDict,
    ObservationSpaceList,
    ObservationSpaceUnit,
)


class BaseSpaceManager:
    """
    BaseSpaceManager is an abstract base class that manages the agent state, action space, and observation space.
    Translates observations to encoded observations for the model input and decodes actions for the environment.

    Attributes:
        _agent_state_container (AgentStateContainer): Container for the agent's state.
        _action_space_manager (ActionSpaceManager): Manager for the action space.
        _observation_space_manager (ObservationSpaceManager): Manager for the observation space.

    Methods:
        __init__(agent_state_container, action_space_kwargs, observation_space_list, observation_space_kwargs):
            Initializes the BaseSpaceManager with the given agent state container, action space arguments, and observation space arguments.

        agent_state_container:
            Returns the agent state container.

        action_space_manager:
            Returns the action space manager.

        observation_space_manager:
            Returns the observation space manager.

        observation_space:
            Returns the observation space as a dictionary.

        observation_space_list:
            Returns the list of observation spaces.

        action_space:
            Returns the action space, which can be either a dictionary or a box.

        config:
            Returns the configuration of the observation space manager, action space manager, and agent state container.

        _init_action_space_manager(action_space_kwargs):
            Initializes the ActionSpaceManager with the given action space arguments.

        _init_observation_space_manager(observation_space_list, observation_space_kwargs):
            Initializes the ObservationSpaceManager with the given observation space list and arguments.
    """

    _agent_state_container: AgentStateContainer
    _action_space_manager: ActionSpaceManager
    _observation_space_manager: ObservationSpaceManager

    def __init__(
        self,
        agent_state_container: AgentStateContainer,
        action_space_kwargs: Dict[str, Any],
        observation_space_list: ObservationSpaceList,
        observation_space_kwargs: Dict[str, Any],
    ):
        """
        Initializes the BaseSpaceManager.

        Args:
            agent_state_container (AgentStateContainer): The container holding the state of the agent.
            action_space_kwargs (Dict[str, Any]): Keyword arguments for initializing the action space manager.
            observation_space_list (ObservationSpaceList): List of observation spaces to be managed.
            observation_space_kwargs (Dict[str, Any]): Keyword arguments for initializing the observation space manager.
        """
        self._agent_state_container = agent_state_container
        self._init_action_space_manager(action_space_kwargs)
        self._init_observation_space_manager(
            observation_space_list, observation_space_kwargs
        )

    def _init_action_space_manager(self, action_space_kwargs: Dict[str, Any]):
        """
        Initializes the ActionSpaceManager.

        Args:
            action_space_kwargs (Dict[str, Any]): Additional keyword arguments for the action spaces.
        """
        action_space_kwargs.update(asdict(self._agent_state_container.action_space))
        self._action_space_manager = ActionSpaceManager(**action_space_kwargs)

    def _init_observation_space_manager(
        self,
        observation_space_list: ObservationSpaceList,
        observation_space_kwargs: Dict[str, Any],
    ):
        """Initialize the observation space manager.
        
        This method creates an ObservationSpaceManager instance by combining
        the provided observation space list and keyword arguments with
        the agent state container's observation space settings.
        
        Args:
            observation_space_list (ObservationSpaceList): The list of observation spaces to include
            observation_space_kwargs (Dict[str, Any]): Additional keyword arguments for observation space configuration
        
        Returns:
            None
        """
        observation_space_kwargs.update(
            asdict(self._agent_state_container.observation_space)
        )
        self._observation_space_manager = ObservationSpaceManager(
            space_list=observation_space_list,
            space_kwargs=observation_space_kwargs,
        )

    def encode_observation(
        self, obs_dict: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """Encode observation dictionary using the observation space manager.
        
        This method takes an observation dictionary and forwards it to the observation space manager
        for encoding.
        
        Args:
            obs_dict (ObservationDict): The raw observation dictionary to encode.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments passed to the observation space manager.
        
        Returns:
            EncodedObservationDict: The encoded observation dictionary.
        """
        return self._observation_space_manager.encode_observation(obs_dict, **kwargs)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        Decodes an action from the agent's action space to the robot's action space.
        
        Args:
            action (np.ndarray): The action to decode, represented as a numpy array in the agent's action space.
        
        Returns:
            np.ndarray: The decoded action in the robot's action space.
        """
        
        return self._action_space_manager.decode_action(action)

    @property
    def agent_state_container(self) -> AgentStateContainer:
        return self._agent_state_container

    @property
    def action_space_manager(self) -> ActionSpaceManager:
        return self._action_space_manager

    @property
    def observation_space_manager(self) -> ObservationSpaceManager:

        return self._observation_space_manager

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space_manager.observation_space

    @property
    def observation_space_list(self) -> List[ObservationSpaceUnit]:
        return self._observation_space_manager.space_list

    @property
    def action_space(self) -> Union[spaces.Dict, spaces.Box]:
        return self._action_space_manager.action_space

    @property
    def config(self):
        return {
            "observation": self._observation_space_manager.config,
            "action": self._action_space_manager.config,
            "agent_state_container": self._agent_state_container,
        }
