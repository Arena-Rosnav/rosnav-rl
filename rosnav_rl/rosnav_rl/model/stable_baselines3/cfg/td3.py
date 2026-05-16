from typing import Literal, Optional, Union

from stable_baselines3.td3 import TD3

from .base import OffPolicyParameters, SBAlgorithmCfg


class TD3_Algorithm_Cfg(OffPolicyParameters):
    """Configuration parameters for Twin Delayed DDPG (TD3).

    TD3 addresses the over-estimation bias of DDPG with three key ideas:
    twin Q-networks, delayed policy updates, and target policy smoothing.

    Attributes:
        policy_delay: Number of critic updates per actor update.
        target_policy_noise: Stddev of Gaussian noise added to the target policy
            for smoothing.
        target_noise_clip: Range to clip the target policy noise.
    """

    algorithm_name = TD3.__name__

    # TD3 defaults
    learning_rate: Union[float, callable] = 1e-3
    batch_size: int = 256  # lowered bar for robots
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, tuple] = 1
    gradient_steps: int = 1

    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5


class TD3_Cfg(SBAlgorithmCfg):
    """Top-level TD3 configuration (architecture + hyper-parameters).

    Attributes:
        type: Discriminator tag, always ``"TD3"``.
        parameters: TD3-specific algorithm parameters.
    """

    type: Literal["TD3"] = "TD3"
    parameters: TD3_Algorithm_Cfg
