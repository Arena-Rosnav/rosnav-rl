from typing import Literal, Union

from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

from ..framework import FrameworkCfg
from .base import SBAlgorithmCfg
from .ppo import PPO_Cfg


class StableBaselinesCfg(FrameworkCfg):
    __name__: Literal[SupportedRLFrameworks.STABLE_BASELINES3] = (
        SupportedRLFrameworks.STABLE_BASELINES3
    )
    algorithm: Union[SBAlgorithmCfg, PPO_Cfg]
