from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

import rosnav_rl.cfg.sb3_cfg as sb3_cfg
from rosnav_rl.utils.type_aliases import (
    EncodedObservationDict,
    _SupportedRosnavRLModels,
)

if TYPE_CHECKING:
    from rosnav_rl.rl_agent import RL_Agent
    from rosnav_rl.spaces import BaseObservationSpace


class RL_Model(ABC):
    """
    Abstract base class for reinforcement learning models.

    Attributes:
        _model: The underlying model used for reinforcement learning.
        _algorithm_cfg: Configuration for the reinforcement learning algorithm.
        _rl_agent: The reinforcement learning agent.

    Methods:
        __init__(rl_agent, algorithm_cfg, *args, **kwargs):
            Initializes the RL_Model with the given agent and algorithm configuration.

        setup_model(*args, **kwargs):
            Abstract method to set up the model. Must be implemented by subclasses.

        train(*args, **kwargs):
            Abstract method to train the model. Must be implemented by subclasses.

        save(*args, **kwargs):
            Abstract method to save the model. Must be implemented by subclasses.

        load(*args, **kwargs):
            Abstract method to load the model. Must be implemented by subclasses.

        get_action(observation, *args, **kwargs):
            Gets the action for a given observation.

        transfer_weights(*args, **kwargs):
            Abstract method to transfer weights. Must be implemented by subclasses.

        is_model_initialized:
            Checks if the model is initialized.

        model:
            Gets or sets the underlying model.

        algorithm_cfg:
            Gets the algorithm configuration.

        observation_space_list:
            Abstract property to get the list of observation spaces. Must be implemented by subclasses.

        observation_space_kwargs:
            Abstract property to get the keyword arguments for observation spaces. Must be implemented by subclasses.

        stack_size:
            Gets the stack size. Default is 1.

        parameter_number:
            Abstract property to get the number of parameters. Must be implemented by subclasses.

        config:
            Gets the configuration dictionary. Default is an empty dictionary.
    """

    _model: ...
    _algorithm_cfg: BaseModel
    _rl_agent: "RL_Agent"

    def __init__(
        self, rl_agent: "RL_Agent", algorithm_cfg: BaseModel, *args, **kwargs
    ) -> None:
        self._rl_agent = rl_agent
        self._algorithm_cfg = algorithm_cfg

    @abstractmethod
    def setup_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
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
    def algorithm_cfg(self) -> "sb3_cfg.SBAlgorithmCfg":
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
