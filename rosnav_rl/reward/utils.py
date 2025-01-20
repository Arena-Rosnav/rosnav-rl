import functools
import numpy as np


def check_params(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.check_parameters()
        return

    return wrapper


def min_distance_from_pointcloud(point_cloud: np.ndarray):
    return np.min(distances_from_pointcloud(point_cloud))


def distances_from_pointcloud(point_cloud: np.ndarray):
    return np.sqrt(
        point_cloud["x"] ** 2 + point_cloud["y"] ** 2 + point_cloud["z"] ** 2
    )
