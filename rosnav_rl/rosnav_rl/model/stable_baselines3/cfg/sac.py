from typing import Optional, Union

from stable_baselines3.sac import SAC

from .base import OffPolicyParameters, SBAlgorithmCfg


class SAC_Algorithm_Cfg(OffPolicyParameters):
    """Configuration parameters for Soft Actor-Critic (SAC).

    SAC is an off-policy actor-critic algorithm that maximises a trade-off
    between expected return and entropy, encouraging exploration.

    Attributes:
        ent_coef: Entropy regularisation coefficient.  ``"auto"`` enables
            automatic temperature tuning.
        target_update_interval: Steps between target-network soft updates.
        target_entropy: Target entropy for automatic temperature tuning.
            ``"auto"`` sets it to ``-dim(action_space)``.
    """

    algorithm_name = SAC.__name__

    # SAC defaults
    learning_rate: Union[float, callable] = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, tuple] = 1
    gradient_steps: int = 1

    ent_coef: Union[str, float] = "auto"
    target_update_interval: int = 1
    target_entropy: Union[str, float] = "auto"


class SAC_Cfg(SBAlgorithmCfg):
    """Top-level SAC configuration (architecture + hyper-parameters).

    Attributes:
        parameters: SAC-specific algorithm parameters.
    """

    parameters: SAC_Algorithm_Cfg
