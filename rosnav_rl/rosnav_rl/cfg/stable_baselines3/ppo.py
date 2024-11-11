import os
from typing import Optional, Union

import rospkg
import torch as th
from pydantic import BaseModel, model_validator

from .lr_schedule import LearningRateSchedulerCfg


class PPO_Algorithm_Cfg(BaseModel):
    total_batch_size: int = 2048
    n_steps: Optional[int] = None
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: Union[float, callable, LearningRateSchedulerCfg] = 0.0005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, callable] = 0.2
    clip_range_vf: Union[None, float, callable] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
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


class ResumeCfg(BaseModel):
    checkpoint: Optional[str] = "last_model"  # checkpoint name to load


class PPO_Policy_Cfg(BaseModel):
    # architecture name of the policy for the agentfactory
    architecture_name: Optional[str] = None
    # agent directory to resume training from
    resume: Optional[ResumeCfg] = None

    @model_validator(mode="after")
    def check_validity(self):
        if self.architecture_name is None and self.resume is None:
            raise ValueError(
                "Either architecture_name or resume must be provided for 'PPO_Policy_Cfg'"
            )
        return self
