from typing import Literal, Union
from .base import SBAlgorithmCfg
from .ppo import PPO_Cfg

from ..framework import FrameworkCfg
from rosnav_rl.utils.type_aliases import SupportedRLFrameworks


class StableBaselinesCfg(FrameworkCfg):
    name: Literal[SupportedRLFrameworks.STABLE_BASELINES3] = (
        SupportedRLFrameworks.STABLE_BASELINES3
    )
    algorithm: Union[SBAlgorithmCfg, PPO_Cfg]
