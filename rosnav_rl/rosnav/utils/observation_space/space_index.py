from enum import Enum
from pedsim_agents.utils import SemanticAttribute
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


class SPACE_FACTORY_KEYS(Enum):
    LASER = "laser"
    GOAL = "goal"
    LAST_ACTION = "last_action"

    STACKED_LASER_MAP = "stacked_laser_map"
    PEDESTRIAN_LOCATION = "ped_location"
    PEDESTRIAN_TYPE = "ped_type"
    PEDESTRIAN_VEL_X = "ped_vel_x"
    PEDESTRIAN_VEL_Y = "ped_vel_y"


class OBS_SPACE_TO_OBS_DICT_KEY(Enum):
    LASER = OBS_DICT_KEYS.LASER
    GOAL = OBS_DICT_KEYS.GOAL
    LAST_ACTION = OBS_DICT_KEYS.LAST_ACTION

    PEDESTRIAN_LOCATION = SemanticAttribute.PEDESTRIAN_LOCATION.value
    PEDESTRIAN_TYPE = SemanticAttribute.PEDESTRIAN_TYPE.value
    PEDESTRIAN_MOVING = SemanticAttribute.PEDESTRIAN_MOVING.value
    PEDESTRIAN_VEL_X = SemanticAttribute.PEDESTRIAN_VEL_X.value
    PEDESTRIAN_VEL_Y = SemanticAttribute.PEDESTRIAN_VEL_Y.value
