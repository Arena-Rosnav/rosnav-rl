from typing import Literal, Union

from rosnav_rl.cfg.framework import FrameworkCfg
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

from .base import SBAlgorithmCfg
from .ppo import PPO_Cfg


class StableBaselinesCfg(FrameworkCfg):
    """
    Configuration class for the Stable Baselines 3 framework.

    This class extends FrameworkCfg to provide specific configuration options
    for the Stable Baselines 3 reinforcement learning framework.

    Attributes:
        __name__ (Literal[SupportedRLFrameworks.STABLE_BASELINES3]): The name of the framework,
                 fixed to SupportedRLFrameworks.STABLE_BASELINES3.
        algorithm (Union[SBAlgorithmCfg, PPO_Cfg]): The reinforcement learning algorithm to use.
                 Can be either a general SB algorithm configuration or specifically a PPO configuration.
    """

    name: Literal[SupportedRLFrameworks.STABLE_BASELINES3] = (
        SupportedRLFrameworks.STABLE_BASELINES3
    )
    algorithm: Union[SBAlgorithmCfg, PPO_Cfg]
