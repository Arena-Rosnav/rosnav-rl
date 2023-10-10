from abc import ABC, abstractmethod
from typing import List, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.modules.module import Module

from .constants import BASE_AGENT_ATTR, PolicyType


class BaseAgent(ABC):
    """Base class for models loaded on runtime from
    the Stable-Baselines3 policy registry during PPO instantiation.
    The architecture of the eventual policy is determined by the
    'policy_kwargs' of the SB3 RL algorithm.
    """

    def __init__(self):
        pass

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
