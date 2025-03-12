from abc import ABC, abstractmethod
from typing import List, Type

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from rosnav_rl.utils.type_aliases import ObservationSpaceKwargs, ObservationSpaceList

from .constants import BASE_AGENT_ATTR, POLICY_TYPE


class StableBaselinesPolicyDescription(ABC):
    """Abstract base class for describing a policy in Stable Baselines 3.

    This class provides an interface for defining the components and parameters
    needed to create a Stable Baselines 3 policy. It abstracts away the common
    elements required for any RL algorithm implementation, such as the observation
    spaces, feature extractors, network architecture, and activation functions.

    Each concrete implementation of this class should define the specific properties
    required for a particular algorithm/policy configuration.

    Attributes:
        BASE_AGENT_ATTR (list): List of attribute names that are common to all agents
                               (defined elsewhere in the codebase)
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
