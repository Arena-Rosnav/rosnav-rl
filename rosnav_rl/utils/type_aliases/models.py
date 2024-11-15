from typing import TypeVar, Union

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

_SupportedStableBaselinesModels = Union[PPO, RecurrentPPO]

_SupportedRosnavRLModels = Union[_SupportedStableBaselinesModels]
