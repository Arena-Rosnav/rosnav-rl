from typing import Optional, Union

import torch as th
from pydantic import BaseModel

from .lr_schedule import LearningRateSchedulerCfg


class CustomDiscreteActionSpaceCfg(BaseModel):
    """
    CustomDiscreteActionSpaceCfg is a configuration class for defining the discrete action space in a reinforcement learning environment.

    Attributes:
        buckets_linear_vel (int): The number of discrete buckets for linear velocity.
        buckets_angular_vel (int): The number of discrete buckets for angular velocity.
    """

    buckets_linear_vel: int
    buckets_angular_vel: int


class ActionSpaceCfg(BaseModel):
    is_discrete: Optional[bool] = False
    custom_discretization: Optional[CustomDiscreteActionSpaceCfg] = None


class PPO_Cfg(BaseModel):
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


class PPO_Policy_Cfg(BaseModel):
    architecture_name: str
    resume: Optional[str] = None
    checkpoint: Optional[str] = "last_model"
