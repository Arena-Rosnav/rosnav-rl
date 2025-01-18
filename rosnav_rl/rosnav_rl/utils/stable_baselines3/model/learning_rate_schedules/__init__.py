from typing import Callable, Union

from rosnav_rl.cfg.sb3_cfg.lr_schedule import LearningRateSchedulerCfg

from .linear import linear_decay
from .square_root import square_root_decay


def load_lr_schedule(
    settings: Union[float, dict, LearningRateSchedulerCfg]
) -> Callable:
    if isinstance(settings, LearningRateSchedulerCfg):
        return _get_lr_schedule(settings.type, settings.kwargs)
    elif isinstance(settings, dict):
        return _get_lr_schedule(settings["type"], settings["kwargs"])
    elif isinstance(settings, float):
        return settings


def _get_lr_schedule(type: str, settings: dict) -> Callable:
    if type == "linear":
        return linear_decay(**settings)
    elif type == "square_root":
        return square_root_decay(**settings)
    else:
        raise NotImplementedError(f"Learning rate schedule '{type}' not implemented!")
