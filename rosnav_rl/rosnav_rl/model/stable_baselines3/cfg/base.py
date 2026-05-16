from typing import ClassVar, Literal, Optional, Union

from abc import ABC

import torch as th
from pydantic import BaseModel, Field

from .callbacks import CallbacksCfg
from .lr_schedule import LearningRateSchedulerCfg
from .normalization import NormalizationCfg
from .transfer import TransferWeightsCfg


class SBAlgorithmParameters(BaseModel, ABC):
    """Base configuration class for all Stable Baselines 3 algorithms.

    Contains only parameters that are truly shared across *every* SB3 algorithm
    (on-policy **and** off-policy).  Algorithm-family-specific parameters live in
    :class:`OnPolicyParameters` and :class:`OffPolicyParameters`.

    Attributes:
        algorithm_name: Identifier used for discriminated-union dispatch.
        total_timesteps: Total environment steps for training.
        learning_rate: Optimizer learning rate (constant, callable, or scheduler cfg).
        batch_size: Minibatch size for gradient updates.
        n_steps: Steps to collect per environment per update (computed automatically
            for on-policy algorithms from ``total_batch_size``).
        gamma: Discount factor.
        stats_window_size: Window for rolling statistics.
        tensorboard_log: Optional TensorBoard log directory.
        verbose: Verbosity level (0 = silent, 1 = info, 2 = debug).
        seed: Random seed for reproducibility.
        device: Torch device ('cpu', 'cuda', 'auto').
        show_progress_bar: Show a progress bar during training.
        _init_setup_model: Build the network on construction.
    """

    algorithm_name: ClassVar[str] = None

    total_timesteps: int = Field(10_000_000, ge=1)
    learning_rate: Union[float, callable, LearningRateSchedulerCfg] = 0.0005
    batch_size: int = 256
    n_steps: Optional[int] = None
    gamma: float = 0.99
    stats_window_size: int = 100
    tensorboard_log: Optional[str] = None
    verbose: int = 0
    seed: Optional[int] = None
    device: Union[th.device, str] = "auto"
    show_progress_bar: bool = False
    _init_setup_model: bool = True

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# On-policy base
# ---------------------------------------------------------------------------


class OnPolicyParameters(SBAlgorithmParameters, ABC):
    """Shared parameters for on-policy algorithms (PPO, A2C, TRPO, RecurrentPPO).

    On-policy methods collect a fixed-size batch of transitions, then perform
    one or more optimisation epochs over that batch before discarding it.

    Attributes:
        total_batch_size: Total transitions to collect per update (split across
            ``n_envs``; ``n_steps`` is derived as ``total_batch_size // n_envs``).
        n_epochs: Number of gradient epochs per collected batch.
        gae_lambda: Lambda for Generalised Advantage Estimation.
        normalize_advantage: Whether to normalise advantages per minibatch.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value-function loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        use_sde: Use State-Dependent Exploration.
        sde_sample_freq: How often to resample the SDE noise matrix (-1 = once per rollout).
    """

    total_batch_size: int = 2048
    n_epochs: int = 5
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1


# ---------------------------------------------------------------------------
# Off-policy base
# ---------------------------------------------------------------------------


class OffPolicyParameters(SBAlgorithmParameters, ABC):
    """Shared parameters for off-policy algorithms (SAC, TD3, DDPG, TQC, CrossQ).

    Off-policy methods maintain a replay buffer and can reuse past transitions
    for many gradient steps.

    Attributes:
        buffer_size: Maximum replay-buffer capacity.
        learning_starts: Number of random-action warm-up steps before training.
        tau: Soft-update coefficient for target networks.
        train_freq: How many environment steps between gradient updates (or a
            ``(freq, unit)`` tuple).
        gradient_steps: Gradient steps per ``train_freq`` trigger (-1 = as many
            as steps collected).
        optimize_memory_usage: Trade compute for RAM in the replay buffer.
        use_sde: Use State-Dependent Exploration.
        sde_sample_freq: How often to resample the SDE noise matrix.
        use_sde_at_warmup: Use SDE during the warm-up phase.
    """

    buffer_size: int = 1_000_000
    learning_starts: int = 100
    tau: float = 0.005
    train_freq: Union[int, tuple] = 1
    gradient_steps: int = 1
    optimize_memory_usage: bool = False
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Wrapper that holds the full algorithm configuration
# ---------------------------------------------------------------------------


class SBAlgorithmCfg(BaseModel):
    """Top-level configuration envelope for any SB3 algorithm.

    Attributes:
        type: Discriminator tag used by Pydantic's tagged union in
            :class:`StableBaselinesCfg`.  Concrete subclasses override this
            with a ``Literal["PPO"]`` / ``Literal["SAC"]`` etc. value so that
            the correct sub-class is selected automatically when parsing from a
            plain dict (e.g. from a YAML config file).  The generic fallback is
            ``"generic"``; prefer using a specific typed sub-class instead.
        architecture_name: Registry key for the neural-network architecture /
            policy description (see :class:`AgentFactory`).
        checkpoint: Checkpoint file to load (without extension).
        transfer_weights: Optional weight-transfer configuration.
        parameters: Algorithm-specific hyper-parameters (auto-dispatched via
            discriminated union in concrete subclasses).
        normalization: Optional ``VecNormalize`` configuration.
        callbacks: Training callback configuration.
    """

    type: Literal["generic"] = "generic"
    architecture_name: str
    checkpoint: Optional[str] = "last_model"
    transfer_weights: Optional[TransferWeightsCfg] = None
    parameters: SBAlgorithmParameters
    normalization: Optional[NormalizationCfg] = None
    callbacks: CallbacksCfg = CallbacksCfg()
