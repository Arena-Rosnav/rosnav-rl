from typing import Literal, Optional

from rosnav_rl.cfg.stable_baselines3.ppo import PPO_Algorithm_Cfg, PPO_Policy_Cfg
from rosnav_rl.cfg.framework import FrameworkCfg


class SB3_Model_Cfg(FrameworkCfg):
    name: Literal["stable_baselines3"] = "stable_baselines3"
    model: PPO_Policy_Cfg
    algorithm: Optional[PPO_Algorithm_Cfg] = PPO_Algorithm_Cfg()
