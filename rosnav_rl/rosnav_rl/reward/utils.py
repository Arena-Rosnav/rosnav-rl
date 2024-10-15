import functools

import numpy as np
import yaml
from tools.constants import TRAINING_CONSTANTS


def check_params(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.check_parameters()
        return

    return wrapper


def load_rew_fnc(config_name: str) -> dict:
    config_location = TRAINING_CONSTANTS.PATHS.REWARD_FUNCTIONS(config_name)
    with open(config_location, "r", encoding="utf-8") as target:
        config = yaml.load(target, Loader=yaml.FullLoader)
    return config


def min_distance_from_pointcloud(point_cloud: np.ndarray):
    return np.min(distances_from_pointcloud(point_cloud))


def distances_from_pointcloud(point_cloud: np.ndarray):
    return np.sqrt(
        point_cloud["x"] ** 2 + point_cloud["y"] ** 2 + point_cloud["z"] ** 2
    )


from rl_utils.utils.observation_collector import *


def get_ped_type_min_distances(observation_dict):
    ped_distances = {}

    relative_locations = observation_dict.get(PedestrianRelativeLocation.name, None)
    pedestrian_types = observation_dict.get(PedestrianTypeCollector.name, None)

    if relative_locations is None or pedestrian_types is None:
        return ped_distances

    if len(relative_locations) == 0 or len(pedestrian_types.points) == 0:
        return ped_distances

    distances = np.linalg.norm(relative_locations, axis=1)
    types = np.array([int(type_data.evidence) for type_data in pedestrian_types.points])

    # get the unique types
    for _type in np.unique(types):
        ped_distances[_type] = np.min(distances[types == _type])

    return ped_distances
