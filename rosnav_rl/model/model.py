from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from rosnav_rl.spaces import BaseObservationSpace, EncodedObservationDict

import rosnav_rl.cfg.sb3_cfg as sb3_cfg
from rosnav_rl.utils.type_aliases import _SupportedRosnavRLModels


class RL_Model(ABC):
    _model = None
    _algorithm_cfg: BaseModel = None

    def __init__(self, algorithm_cfg: BaseModel, *args, **kwargs) -> None:
        self._algorithm_cfg = algorithm_cfg

    @abstractmethod
    def setup_model(self, *args, **kwargs):
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

    def transfer_weights(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def is_model_initialized(self):
        return self._model is not None

    @property
    def model(self) -> _SupportedRosnavRLModels:
        if self._model is None:
            raise ValueError("Model not initialized. Call 'initialize' first.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def algorithm_cfg(self) -> "sb3_cfg.BaseAlgorithmCfg":
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
