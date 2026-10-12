from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel

from ..utils.type_aliases import (
    ObservationDict,
    _SupportedRosnavRLModels,
)

if TYPE_CHECKING:
    from ..cfg.framework import FrameworkCfg
    from ..rl_agent import RL_Agent
    from ..spaces.observation_space.spaces.base_observation_space import (
        BaseObservationSpace,
    )


class RL_Model(ABC):
    """Abstract base class for Reinforcement Learning models.

    This class serves as a blueprint for different reinforcement learning model implementations
    within the rosnav framework. It defines the common interface that all RL models must implement.

    Attributes:
        _model: The actual machine learning model instance.
        _algorithm_cfg (BaseModel): Configuration parameters for the RL algorithm.
        _rl_agent (RL_Agent): Reference to the RL agent controlling this model.

    Methods:
        setup_model: Initialize and configure the model.
        train: Train the model with provided data.
        save: Save the model to disk.
        load: Load the model from disk.
        get_action: Get an action from the model based on the current observation.
        transfer_weights: Transfer weights from another model (optional implementation).
        is_model_initialized: Check if the model has been initialized.
        model: Property to access the underlying model with validation.
        algorithm_cfg: Property to access algorithm configuration.
        observation_space_list: Property to get the list of observation spaces.
        observation_space_kwargs: Property to get keyword arguments for observation spaces.
        stack_size: Property that returns the size of observation stacks (default: 1).
        parameter_number: Property that returns the number of model parameters.
        config: Property that returns the model configuration (default: empty dict).
    """

    _model: _SupportedRosnavRLModels | None
    _algorithm_cfg: BaseModel
    _rl_agent: "RL_Agent"

    def __init__(self, rl_agent: "RL_Agent", algorithm_cfg: BaseModel, *args: Any, **kwargs: Any) -> None:
        self._rl_agent = rl_agent
        self._algorithm_cfg = algorithm_cfg

    @classmethod
    def from_framework_cfg(cls, rl_agent: "RL_Agent", framework_cfg: "FrameworkCfg", *args: Any, **kwargs: Any) -> Self:
        """Construct a model from a :class:`FrameworkCfg` envelope.

        Subclasses may override this to extract the relevant portion of the
        config (e.g. ``framework_cfg.algorithm`` for SB3).  The default
        implementation passes the entire ``framework_cfg`` as the algorithm
        config.
        """
        return cls(*args, rl_agent=rl_agent, algorithm_cfg=framework_cfg, **kwargs)

    @abstractmethod
    def setup_model(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    def save(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get_action(self, observation: ObservationDict, *args: Any, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def transfer_weights(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @property
    def is_model_initialized(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> _SupportedRosnavRLModels:
        if self._model is None:
            raise ValueError("Model not initialized. Call 'initialize' first.")
        return self._model

    @model.setter
    def model(self, model: _SupportedRosnavRLModels):
        self._model = model

    @property
    def algorithm_cfg(self) -> BaseModel:
        return self._algorithm_cfg

    @property
    def observation_space_list(self) -> list[type["BaseObservationSpace"]]:
        raise NotImplementedError()

    @property
    def observation_space_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError()

    @property
    def stack_size(self) -> int:
        return 1

    @property
    def parameter_number(self) -> int:
        raise NotImplementedError()

    @property
    def config(self) -> dict[str, Any]:
        return {}
