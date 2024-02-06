from abc import ABC, abstractmethod
from typing import List, Type

from rosnav.rosnav_space_manager.base_space_encoder import BaseSpaceEncoder
from rosnav.utils.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from .constants import BASE_AGENT_ATTR, PolicyType
from rosnav.rosnav_space_manager.default_encoder import DefaultEncoder


class BaseAgent(ABC):
    """
    Base class for defining an agent in a reinforcement learning environment.

    Attributes:
        space_encoder_class (Type[BaseSpaceEncoder]): The class of the space encoder used by the agent.
        observation_spaces (List[BaseObservationSpace]): List of observation space indices.
        observation_space_kwargs (dict): Additional keyword arguments for the observation space.
        type (PolicyType): The type of policy used by the agent.
        features_extractor_class (Type[BaseFeaturesExtractor]): The class of the features extractor used by the agent.
        features_extractor_kwargs (dict): Additional keyword arguments for the features extractor.
        net_arch (List[dict]): List of dictionaries specifying the architecture of the neural network.
        activation_fn (Type[Module]): The activation function used in the neural network.

    Methods:
        get_kwargs(): Get the keyword arguments for the agent.

    """

    @property
    def space_encoder_class(self) -> Type[BaseSpaceEncoder]:
        """
        Get the class of the space encoder used by the agent.

        Returns:
            Type[BaseSpaceEncoder]: The class of the space encoder.
        """
        return DefaultEncoder

    @property
    @abstractmethod
    def observation_spaces(self) -> List[BaseObservationSpace]:
        """
        Get the list of observation spaces.

        Returns:
            List[BaseObservationSpace]: List of observation spaces.
        """
        return None

    @property
    def observation_space_kwargs(self) -> dict:
        """
        Get additional keyword arguments for the observation space.

        Returns:
            dict: Additional keyword arguments for the observation space.
        """
        return {}

    @property
    @abstractmethod
    def type(self) -> PolicyType:
        """
        Get the type of policy used by the agent.

        Returns:
            PolicyType: The type of policy used by the agent.
        """
        pass

    @property
    @abstractmethod
    def features_extractor_class(self) -> Type[BaseFeaturesExtractor]:
        """
        Get the class of the features extractor used by the agent.

        Returns:
            Type[BaseFeaturesExtractor]: The class of the features extractor.
        """
        pass

    @property
    @abstractmethod
    def features_extractor_kwargs(self) -> dict:
        """
        Get additional keyword arguments for the features extractor.

        Returns:
            dict: Additional keyword arguments for the features extractor.
        """
        pass

    @property
    @abstractmethod
    def net_arch(self) -> List[dict]:
        """
        Get the architecture of the neural network.

        Returns:
            List[dict]: List of dictionaries specifying the architecture of the neural network.
        """
        pass

    @property
    @abstractmethod
    def activation_fn(self) -> Type[Module]:
        """
        Get the activation function used in the neural network.

        Returns:
            Type[Module]: The activation function used in the neural network.
        """
        pass

    def get_kwargs(self):
        """
        Get the keyword arguments for the agent.

        Returns:
            dict: Keyword arguments for the agent.
        """
        kwargs = {}
        for key in self.__dir__():
            if key in BASE_AGENT_ATTR:
                val = getattr(self, key)
                if val is not None:
                    kwargs[key] = val
        return kwargs
