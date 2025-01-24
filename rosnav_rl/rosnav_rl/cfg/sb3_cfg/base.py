from typing import Literal, Optional, Union, ClassVar

from abc import ABC
import torch as th
from pydantic import BaseModel

from .lr_schedule import LearningRateSchedulerCfg
from .normalization import NormalizationCfg
from .transfer import TransferWeightsCfg


class SBAlgorithmParameters(BaseModel, ABC):
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

    class Config:
        arbitrary_types_allowed = True


class SBAlgorithmCfg(BaseModel, ABC):
    __algorithm_name__: ClassVar[str] = None
    architecture_name: str
    checkpoint: Optional[str] = "last_model"
    transfer_weights: Optional[TransferWeightsCfg] = None
    parameters: SBAlgorithmParameters
    normalization: Optional[NormalizationCfg] = None
