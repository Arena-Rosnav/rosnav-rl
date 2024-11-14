from typing import Optional, Union

import torch as th
from pydantic import BaseModel, model_validator

from .lr_schedule import LearningRateSchedulerCfg


class BaseAlgorithmParameters(BaseModel):
    total_batch_size: int = 2048
    n_steps: Optional[int] = None
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: Union[float, callable, LearningRateSchedulerCfg] = 0.0005
    stats_window_size: int = 100
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    seed: Optional[int] = None
    device: Union[th.device, str] = "auto"
    _init_setup_model: bool = True

    @model_validator(mode="after")
    def load_learning_rate_scheduler(self):
        if isinstance(self.learning_rate, dict):
            self.learning_rate = LearningRateSchedulerCfg(**self.learning_rate).callable
        elif isinstance(self.learning_rate, LearningRateSchedulerCfg):
            self.learning_rate = self.learning_rate.callable
        return self

    class Config:
        arbitrary_types_allowed = True


class BaseAlgorithmCfg(BaseModel):
    architecture_name: str
    checkpoint: Optional[str] = "last_model"
    parameters: BaseAlgorithmParameters
