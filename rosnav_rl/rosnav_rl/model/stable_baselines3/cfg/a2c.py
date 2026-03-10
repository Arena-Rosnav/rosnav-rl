from typing import Optional, Union

from stable_baselines3.a2c import A2C

from .base import OnPolicyParameters, SBAlgorithmCfg


class A2C_Algorithm_Cfg(OnPolicyParameters):
    """Configuration parameters for Advantage Actor-Critic (A2C).

    A2C is the synchronous, deterministic variant of Asynchronous Advantage
    Actor-Critic (A3C).  It uses multiple workers in parallel to collect
    experience and a single gradient step per batch.

    Attributes:
        rms_prop_eps: Epsilon for RMSProp optimiser stability.
        use_rms_prop: Whether to use RMSProp (``True``) or Adam (``False``).
    """

    algorithm_name = A2C.__name__

    # A2C typically uses n_epochs=1 (single gradient pass)
    n_epochs: int = 1
    # A2C default batch = full rollout (no minibatch splitting)
    total_batch_size: int = 2048

    rms_prop_eps: float = 1e-5
    use_rms_prop: bool = True


class A2C_Cfg(SBAlgorithmCfg):
    """Top-level A2C configuration (architecture + hyper-parameters).

    Attributes:
        parameters: A2C-specific algorithm parameters.
    """

    parameters: A2C_Algorithm_Cfg
