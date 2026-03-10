from typing import Optional, Union

from sb3_contrib.tqc import TQC

from .base import OffPolicyParameters, SBAlgorithmCfg


class TQC_Algorithm_Cfg(OffPolicyParameters):
    """Configuration parameters for Truncated Quantile Critics (TQC).

    TQC extends SAC by maintaining an ensemble of quantile critics and
    dropping the highest quantile predictions, which controls over-estimation
    bias more aggressively than twin critics.

    Attributes:
        ent_coef: Entropy regularisation coefficient (``"auto"`` for automatic).
        target_update_interval: Steps between soft target-network updates.
        target_entropy: Target entropy for automatic ``alpha`` tuning.
        top_quantiles_to_drop_per_net: Number of top quantile predictions to
            discard per critic network when computing the target value.
    """

    algorithm_name = TQC.__name__

    # TQC defaults (SAC-like)
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
    top_quantiles_to_drop_per_net: int = 2


class TQC_Cfg(SBAlgorithmCfg):
    """Top-level TQC configuration (architecture + hyper-parameters).

    Attributes:
        parameters: TQC-specific algorithm parameters.
    """

    parameters: TQC_Algorithm_Cfg
