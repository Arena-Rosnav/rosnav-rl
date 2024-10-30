from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseObservationSpace, EncodedObservationDict


class RL_Model(ABC):
    _model = None
    _model_cfg: BaseModel = None
    _algorithm_cfg: BaseModel = None

    def __init__(
        self, model_cfg: BaseModel, algorithm_cfg: BaseModel, *args, **kwargs
    ) -> None:
        self._model_cfg = model_cfg
        self._algorithm_cfg = algorithm_cfg

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
    def is_model_initialized(self):
        return self._model is not None

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model not initialized. Call 'initialize' first.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def model_cfg(self):
        return self._model_cfg

    @property
    def algorithm_cfg(self):
        return self._algorithm_cfg

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
