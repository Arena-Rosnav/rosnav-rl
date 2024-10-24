from enum import Enum
from typing import Dict

import numpy as np
import torch as th

from ..observation_space.spaces.base_observation_space import BaseObservationSpace
from ..observation_space.spaces.base.dist_angle_to_goal_space import (
    DistAngleToGoalSpace,
)
from ..observation_space.spaces.base.dist_angle_to_subgoal_space import (
    DistAngleToSubgoalSpace,
)
from ..observation_space.spaces.base.laser_space import LaserScanSpace
from ..observation_space.spaces.base.last_action_space import LastActionSpace
from ..observation_space.spaces.base.rgbd_space import RGBDSpace
from .spaces.base.subgoal_in_robot_frame import SubgoalInRobotFrameSpace

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

TensorDict = Dict[str, th.Tensor]

ObservationSpaceName = str
ObservationEncoding = np.ndarray

EncodedObservationDict = Dict[ObservationSpaceName, ObservationEncoding]


# OLD SPACE INDEX - only keep for old code compatibility
class SPACE_INDEX(Enum):
    LASER = LaserScanSpace
    DIST_ANGLE_TO_GOAL = DistAngleToGoalSpace
    DIST_ANGLE_TO_SUBGOAL = DistAngleToSubgoalSpace
    LAST_ACTION = LastActionSpace
    RGBD = RGBDSpace

    STACKED_LASER_MAP = StackedLaserMapSpace
    PEDESTRIAN_LOCATION = PedestrianLocationSpace
    PEDESTRIAN_TYPE = PedestrianTypeSpace
    PEDESTRIAN_VEL_X = PedestrianVelXSpace
    PEDESTRIAN_VEL_Y = PedestrianVelYSpace
    PEDESTRIAN_SOCIAL_STATE = PedestrianSocialStateSpace
