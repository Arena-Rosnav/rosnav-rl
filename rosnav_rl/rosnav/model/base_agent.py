from abc import ABC, abstractmethod
from typing import List, Type

from rosnav.rosnav_space_manager.base_space_encoder import BaseSpaceEncoder
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from .constants import BASE_AGENT_ATTR, PolicyType
from rosnav.rosnav_space_manager.default_encoder import DefaultEncoder


class BaseAgent(ABC):
    """Base class for models loaded on runtime from
    the Stable-Baselines3 policy registry during PPO instantiation.
    The architecture of the eventual policy is determined by the
    'policy_kwargs' of the SB3 RL algorithm.
    """

    @property
    def space_encoder_class(self) -> Type[BaseSpaceEncoder]:
        return DefaultEncoder

    @property
    def observation_spaces(self) -> List[SPACE_INDEX]:
        return None

    @property
    def observation_space_kwargs(self) -> dict:
        return {}

    @property
    @abstractmethod
    def type(self) -> PolicyType:
        pass

    @property
    @abstractmethod
    def features_extractor_class(self) -> Type[BaseFeaturesExtractor]:
        pass

    @property
    @abstractmethod
    def features_extractor_kwargs(self) -> dict:
        pass

    @property
    @abstractmethod
    def net_arch(self) -> List[dict]:
        pass

    @property
    @abstractmethod
    def activation_fn(self) -> Type[Module]:
        pass

    def get_kwargs(self):
        kwargs = {}
        for key in self.__dir__():
            if key in BASE_AGENT_ATTR:
                val = getattr(self, key)
                if val is not None:
                    kwargs[key] = val
        return kwargs
