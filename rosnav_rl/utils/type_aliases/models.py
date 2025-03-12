from typing import TYPE_CHECKING, TypeVar, Union

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

if TYPE_CHECKING:
    import rosnav_rl.model.dreamerv3 as dreamerv3

_SupportedStableBaselinesModels = Union[PPO, RecurrentPPO]
_SupportedDreamerModels = "dreamerv3.Dreamer"

_SupportedRosnavRLModels = Union[_SupportedStableBaselinesModels, _SupportedDreamerModels]
