from abc import ABC
from typing import TYPE_CHECKING, Any

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from .constants import BASE_AGENT_ATTR

if TYPE_CHECKING:
    from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
        BaseObservationSpace,
    )


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

    algorithm_class: type[BaseAlgorithm]
    observation_spaces: list[type["BaseObservationSpace"]]
    observation_space_kwargs: dict[str, Any] = {}
    features_extractor_class: type[BaseFeaturesExtractor]
    features_extractor_kwargs: dict[str, Any]
    net_arch: list[int] | dict[str, list[int]]
    activation_fn: type[Module]
    stack_size: int = 1

    def get_kwargs(self) -> dict[str, Any]:
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
