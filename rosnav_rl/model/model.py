from abc import ABC, abstractmethod

from rosnav_rl.spaces import EncodedObservationDict, BaseObservationSpace
from typing import List, Dict, Any


class RL_Model(ABC):
    model = None

    @abstractmethod
    @property
    def observation_space_list(self) -> List[BaseObservationSpace]:
        raise NotImplementedError()

    @abstractmethod
    @property
    def observation_space_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    @property
    def config(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    def get_action(self, observation: EncodedObservationDict, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()
