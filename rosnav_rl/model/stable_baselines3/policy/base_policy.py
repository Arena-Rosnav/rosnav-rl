from abc import ABC, abstractmethod
from typing import List, Type

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from rosnav_rl.utils.type_aliases import ObservationSpaceList, ObservationSpaceKwargs

from .constants import BASE_AGENT_ATTR, POLICY_TYPE


class StableBaselinesPolicyDescription(ABC):
    """
    StableBaselinesPolicyDescription is an abstract base class that defines the interface for policies used in the Stable Baselines framework.

    Properties:
        algorithm_class (Type[BaseAlgorithm]): Abstract property to get the algorithm class.
        observation_spaces (ObservationSpaceList): Abstract property to get the list of observation spaces.
        observation_space_kwargs (ObservationSpaceKwargs): Property to get additional keyword arguments for the observation space.
        features_extractor_class (Type[BaseFeaturesExtractor]): Abstract property to get the class of the features extractor used by the agent.
        features_extractor_kwargs (dict): Abstract property to get additional keyword arguments for the features extractor.
        net_arch (List[dict]): Abstract property to get the architecture of the neural network.
        activation_fn (Type[Module]): Abstract property to get the activation function used in the neural network.
        stack_size (int): Property to get the stack size, defaults to 1.

    Methods:
        get_kwargs() -> dict: Generates the keyword arguments dict to be used in the agent.
    """

    @property
    @abstractmethod
    def algorithm_class(self) -> Type[BaseAlgorithm]:
        """
        Get the algorithm class.

        Returns:
            Type[BaseAlgorithm]: The algorithm class.
        """
        pass

    @property
    @abstractmethod
    def observation_spaces(self) -> ObservationSpaceList:
        """
        Get the list of observation spaces.

        Returns:
            List[BaseObservationSpace]: List of observation spaces.
        """
        return None

    @property
    def observation_space_kwargs(self) -> ObservationSpaceKwargs:
        """
        Get additional keyword arguments for the observation space.

        Returns:
            dict: Additional keyword arguments for the observation space.
        """
        return {}

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

    @property
    def stack_size(self) -> int:
        return 1

    def get_kwargs(self) -> dict:
        """
        Get the keyword arguments for the agent.

        Args:
            observation_space_manager (ObservationSpaceManager): The observation space manager.
            stack_size (int, optional): The stack size. Defaults to 1.

        Returns:
            dict: Keyword arguments for the agent.
        """
        kwargs = {}
        for key in self.__dir__():
            if key in BASE_AGENT_ATTR:
                val = getattr(self, key)
                if val is not None:
                    kwargs[key] = val

                if key == "features_extractor_kwargs":
                    kwargs[key].update({"stack_size": self.stack_size})

        return kwargs
