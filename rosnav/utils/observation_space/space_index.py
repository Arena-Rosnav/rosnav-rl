from enum import Enum

from ..observation_space.spaces.base.goal_space import GoalSpace
from ..observation_space.spaces.base.laser_space import LaserScanSpace
from ..observation_space.spaces.base.last_action_space import LastActionSpace
from ..observation_space.spaces.feature_maps.pedestrian_location_space import (
    PedestrianLocationSpace,
)
from ..observation_space.spaces.feature_maps.pedestrian_type_space import (
    PedestrianTypeSpace,
)
from ..observation_space.spaces.feature_maps.pedestrian_vel_x_space import (
    PedestrianVelXSpace,
)
from ..observation_space.spaces.feature_maps.pedestrian_vel_y_space import (
    PedestrianVelYSpace,
)
from ..observation_space.spaces.feature_maps.stacked_laser_map_space import (
    StackedLaserMapSpace,
)
from .spaces.feature_maps.pedestrian_social_state_space import (
    PedestrianSocialStateSpace,
)


class SPACE_INDEX(Enum):
    LASER = LaserScanSpace
    GOAL = GoalSpace
    LAST_ACTION = LastActionSpace

    STACKED_LASER_MAP = StackedLaserMapSpace
    PEDESTRIAN_LOCATION = PedestrianLocationSpace
    PEDESTRIAN_TYPE = PedestrianTypeSpace
    PEDESTRIAN_VEL_X = PedestrianVelXSpace
    PEDESTRIAN_VEL_Y = PedestrianVelYSpace
    PEDESTRIAN_SOCIAL_STATE = PedestrianSocialStateSpace
