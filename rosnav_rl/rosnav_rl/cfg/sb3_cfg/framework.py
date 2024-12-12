from typing import Literal, Union
from .base import SBAlgorithmCfg
from .ppo import PPO_Cfg

from ..framework import FrameworkCfg


class StableBaselinesCfg(FrameworkCfg):
    name: Literal["stable_baselines3"] = "stable_baselines3"
    algorithm: Union[SBAlgorithmCfg, PPO_Cfg]
