import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Concatenate

import numpy as np

if TYPE_CHECKING:
    from .reward_units.base_reward_units import RewardUnit


def check_params[S: RewardUnit, **P](
    fn: Callable[Concatenate[S, P], None],
) -> Callable[Concatenate[S, P], None]:
    @functools.wraps(fn)
    def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> None:
        fn(self, *args, **kwargs)
        self.check_parameters()
        return

    return wrapper


def min_distance_from_pointcloud(point_cloud: np.ndarray) -> np.floating:
    return np.min(distances_from_pointcloud(point_cloud))


def distances_from_pointcloud(point_cloud: np.ndarray) -> np.ndarray:
    return np.sqrt(
        point_cloud["x"] ** 2 + point_cloud["y"] ** 2 + point_cloud["z"] ** 2
    )
