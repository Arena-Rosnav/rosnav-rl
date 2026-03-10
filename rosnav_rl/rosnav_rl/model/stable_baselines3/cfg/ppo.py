from typing import Optional, Union

from stable_baselines3.ppo import PPO

from .base import OnPolicyParameters, SBAlgorithmCfg


class PPO_Algorithm_Cfg(OnPolicyParameters):
    """Configuration parameters for Proximal Policy Optimization (PPO).

    Extends :class:`OnPolicyParameters` with PPO-specific clipping and KL
    divergence settings.

    Attributes:
        clip_range: Clipping parameter for the surrogate policy loss.
        clip_range_vf: Clipping parameter for the value function.  ``None``
            disables value-function clipping.
        target_kl: Optional KL-divergence threshold for early stopping.
    """

    algorithm_name = PPO.__name__
    clip_range: Union[float, callable] = 0.2
    clip_range_vf: Union[None, float, callable] = None
    target_kl: Optional[float] = None


class PPO_Cfg(SBAlgorithmCfg):
    """Top-level PPO configuration (architecture + hyper-parameters).

    Attributes:
        parameters: PPO-specific algorithm parameters.
    """

    parameters: PPO_Algorithm_Cfg
