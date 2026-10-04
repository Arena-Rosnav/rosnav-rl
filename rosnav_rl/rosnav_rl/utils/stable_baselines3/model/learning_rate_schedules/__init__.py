from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .linear import linear_decay
from .square_root import square_root_decay

if TYPE_CHECKING:
    import rosnav_rl.model.stable_baselines3.cfg as sb3_cfg

def load_lr_schedule(
    settings: float | Callable[[float], float] | dict[str, Any] | sb3_cfg.LearningRateSchedulerCfg
) -> float | Callable[[float], float]:
    import rosnav_rl.model.stable_baselines3.cfg as sb3_cfg
    if isinstance(settings, sb3_cfg.LearningRateSchedulerCfg):
        return _get_lr_schedule(settings.type, settings.kwargs)
    elif isinstance(settings, dict):
        return _get_lr_schedule(settings["type"], settings["kwargs"])
    return settings


def _get_lr_schedule(type: str, settings: dict[str, Any]) -> Callable[[float], float]:
    if type == "linear":
        return linear_decay(**settings)
    elif type == "square_root":
        return square_root_decay(**settings)
    else:
        raise NotImplementedError(f"Learning rate schedule '{type}' not implemented!")
