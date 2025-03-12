from typing import Optional, Union, ClassVar

from abc import ABC
import torch as th
from pydantic import BaseModel, Field

from .lr_schedule import LearningRateSchedulerCfg
from .normalization import NormalizationCfg
from .transfer import TransferWeightsCfg
from .callbacks import CallbacksCfg

class SBAlgorithmParameters(BaseModel, ABC):
    """
    Base configuration class for Stable Baselines 3 algorithms.

    This class defines common parameters used across different RL algorithms
    in the Stable Baselines 3 library.

    Attributes:
        total_batch_size (int): Total number of environment steps to collect for each training update.
            Default is 2048.
        n_steps (Optional[int]): Number of steps to run for each environment per update.
            If None, will be determined based on algorithm requirements.
        batch_size (int): Minibatch size for each gradient update.
            Default is 256.
        n_epochs (int): Number of epochs to optimize on the collected data.
            Default is 5.
        learning_rate (Union[float, callable, LearningRateSchedulerCfg]): Learning rate for optimizer.
            Can be a constant float, a callable function, or a scheduler configuration.
            Default is 0.0005.
        stats_window_size (int): Window size for computing rolling statistics.
            Default is 100.
        tensorboard_log (Optional[str]): Path to save tensorboard logs.
            If None, no tensorboard logging will be done.
        verbose (int): Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
            Default is 0.
        seed (Optional[int]): Random seed for reproducibility.
            If None, a random seed will be used.
        device (Union[th.device, str]): Device to run the model on ('cpu', 'cuda', 'auto').
            Default is "auto" which selects the best available device.
        _init_setup_model (bool): Whether to build the network at the creation of the instance.
            Default is True.
    """

    algorithm_name: ClassVar[str] = None
    total_batch_size: int = 2048
    total_timesteps: int = Field(10_000_000, ge=1)
    n_steps: Optional[int] = None
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: Union[float, callable, LearningRateSchedulerCfg] = 0.0005
    stats_window_size: int = 100
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    seed: Optional[int] = None
    device: Union[th.device, str] = "auto"
    show_progress_bar: bool = False
    _init_setup_model: bool = True

    class Config:
        arbitrary_types_allowed = True


class SBAlgorithmCfg(BaseModel):
    """
    Configuration class for Stable Baselines 3 (SB3) algorithms.

    This abstract base class defines the common configuration parameters needed for
    all SB3 reinforcement learning algorithms.

    Attributes:
        architecture_name (str): Name of the neural network architecture to use.
        checkpoint (Optional[str]): Name or path of the checkpoint to load.
            Defaults to "last_model".
        transfer_weights (Optional[TransferWeightsCfg]): Configuration for weight
            transfer between models. Defaults to None.
        parameters (SBAlgorithmParameters): Algorithm-specific parameters.
        normalization (Optional[NormalizationCfg]): Configuration for observation
            normalization. Defaults to None.
    """

    architecture_name: str
    checkpoint: Optional[str] = "last_model"
    transfer_weights: Optional[TransferWeightsCfg] = None
    parameters: SBAlgorithmParameters
    normalization: Optional[NormalizationCfg] = None
    callbacks: CallbacksCfg = CallbacksCfg()
