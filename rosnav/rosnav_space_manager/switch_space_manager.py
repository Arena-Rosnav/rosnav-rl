from typing import Any, Dict, List, Union

import rospy
from rosnav.utils.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav.utils.observation_space.spaces.feature_maps.base_feature_map_space import (
    BaseFeatureMapSpace,
)

from .encoder.switch_space_encoder import SwitchSpaceEncoder
from .rosnav_space_manager import RosnavSpaceManager


class SwitchSpaceManager(RosnavSpaceManager):
    def __init__(
        self,
        observation_spaces: List[
            Union[BaseObservationSpace, BaseFeatureMapSpace]
        ] = None,
        observation_space_kwargs: Dict[str, Any] = None,
        action_space_kwargs: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        observation_space_kwargs = observation_space_kwargs or {}
        action_space_kwargs = action_space_kwargs or {}

        self._stacked = rospy.get_param_cached("rl_agent/frame_stacking/enabled")
        self._laser_num_beams = (
            rospy.get_param_cached("laser/num_beams")
            if not rospy.get_param("laser/reduce_num_beams")
            else rospy.get_param("laser/reduced_num_laser_beams")
        )
        self._laser_max_range = rospy.get_param_cached("laser/range")
        self._radius = rospy.get_param_cached("robot_radius")

        # TODO: add num_ped_types to rosparam
        self._num_ped_types = 5
        self._ped_min_speed_x = -5.0
        self._ped_max_speed_x = 5.0
        self._ped_min_speed_y = -5.0
        self._ped_max_speed_y = 5.0
        self._social_state_num = 99

        _observation_kwargs = {
            "min_linear_vel": -2.0,
            "max_linear_vel": 2.0,
            "min_angular_vel": -4.0,
            "max_angular_vel": 4.0,
            "laser_num_beams": self._laser_num_beams,
            "laser_max_range": self._laser_max_range,
            "num_ped_types": self._num_ped_types,
            "min_speed_x": self._ped_min_speed_x,
            "max_speed_x": self._ped_max_speed_x,
            "min_speed_y": self._ped_min_speed_y,
            "max_speed_y": self._ped_max_speed_y,
            "social_state_num": self._social_state_num,
            **observation_space_kwargs,
        }

        self._encoder = SwitchSpaceEncoder(
            action_space_kwargs=action_space_kwargs,
            observation_list=observation_spaces,
            observation_kwargs=_observation_kwargs,
        )
