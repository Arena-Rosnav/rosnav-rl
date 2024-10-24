from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseObservationSpace, EncodedObservationDict


class RL_Model(ABC):
    model = None
    algorithm_cfg: BaseModel = None

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def get_action(self, observation: "EncodedObservationDict", *args, **kwargs):
        pass

    @property
    def observation_space_list(self) -> List["BaseObservationSpace"]:
        raise NotImplementedError()

    @property
    def observation_space_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def stack_size(self) -> int:
        return 1

    @property
    def parameter_number(self) -> int:
        raise NotImplementedError()

    @property
    def config(self) -> Dict[str, Any]:
        return {}
