from typing import TYPE_CHECKING

from sb3_contrib import TQC, TRPO, CrossQ, RecurrentPPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

__all__ = [
    "_SupportedDreamerModels",
    "_SupportedRosnavRLModels",
    "_SupportedStableBaselinesModels",
]

if TYPE_CHECKING:
    from rosnav_rl.model.dreamerv3.dreamer import Dreamer

type _SupportedStableBaselinesModels = PPO | A2C | RecurrentPPO | TRPO | SAC | TD3 | DDPG | TQC | CrossQ
type _SupportedDreamerModels = Dreamer

type _SupportedRosnavRLModels = _SupportedStableBaselinesModels | _SupportedDreamerModels
