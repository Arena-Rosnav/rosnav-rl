from typing import Optional, Union

from sb3_contrib.crossq import CrossQ

from .base import OffPolicyParameters, SBAlgorithmCfg


class CrossQ_Algorithm_Cfg(OffPolicyParameters):
    """Configuration parameters for CrossQ.

    CrossQ is a sample-efficient off-policy algorithm that uses batch
    normalisation across the concatenation of state--action pairs from the
    current policy and the replay buffer, removing the need for target
    networks entirely.

    Attributes:
        ent_coef: Entropy regularisation coefficient (``"auto"`` for automatic).
        target_entropy: Target entropy for automatic temperature tuning.
        policy_delay: Number of critic updates per actor update.
    """

    algorithm_name = CrossQ.__name__

    # CrossQ defaults
    learning_rate: Union[float, callable] = 1e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, tuple] = 1
    gradient_steps: int = 1

    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    policy_delay: int = 1


class CrossQ_Cfg(SBAlgorithmCfg):
    """Top-level CrossQ configuration (architecture + hyper-parameters).

    Attributes:
        parameters: CrossQ-specific algorithm parameters.
    """

    parameters: CrossQ_Algorithm_Cfg
