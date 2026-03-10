from typing import TYPE_CHECKING, TypeVar, Union

from sb3_contrib import CrossQ, RecurrentPPO, TQC, TRPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

if TYPE_CHECKING:
    import rosnav_rl.model.dreamerv3 as dreamerv3

_SupportedStableBaselinesModels = Union[
    PPO, A2C, RecurrentPPO, TRPO,       # on-policy
    SAC, TD3, DDPG, TQC, CrossQ,        # off-policy
]
_SupportedDreamerModels = "dreamerv3.Dreamer"

_SupportedRosnavRLModels = Union[_SupportedStableBaselinesModels, _SupportedDreamerModels]
