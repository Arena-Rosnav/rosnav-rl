from typing import Any, Dict, List, Union

import rospy
from rl_utils.utils.observation_collector.traversal import get_required_observations
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav_rl.spaces.observation_space.spaces.feature_maps.base_feature_map_space import (
    BaseFeatureMapSpace,
)

from .base_space_manager import BaseSpaceManager


class RosnavSpaceManager(BaseSpaceManager):
    """
    Manages the observation and action spaces for the ROS navigation system.
    """

    def __init__(
        self,
        observation_space_list: List[
            Union[BaseObservationSpace, BaseFeatureMapSpace]
        ] = None,
        action_space_kwargs: Dict[str, Any] = None,
        observation_space_kwargs: Dict[str, Any] = None,
    ):
        """
        Initializes the RosnavSpaceManager. Retrieves the Agent parameters from the ROS parameter server and initializes the space encoder containing the observation and action spaces.

        Args:
            observation_spaces (List[Union[BaseObservationSpace, BaseFeatureMapSpace]], optional): The list of observation spaces. Defaults to None.
            observation_space_kwargs (Dict[str, Any], optional): Additional keyword arguments for the observation spaces. Defaults to None.
            action_space_kwargs (Dict[str, Any], optional): Additional keyword arguments for the action spaces. Defaults to None.
        """
        observation_space_kwargs = observation_space_kwargs or {}
        action_space_kwargs = action_space_kwargs or {}

        self._laser_num_beams = (
            rospy.get_param("laser/num_beams", 0)
            if not rospy.get_param("laser/reduce_num_beams", False)
            else rospy.get_param("laser/reduced_num_laser_beams")
        )
        self._laser_max_range = rospy.get_param("laser/range", 0.0)
        self._radius = rospy.get_param("robot_radius", 0.0)
        self._is_holonomic = rospy.get_param("is_holonomic", False)

        # TODO: add num_ped_types to rosparam
        self._num_ped_types = 5
        self._ped_min_speed_x = -5.0
        self._ped_max_speed_x = 5.0
        self._ped_min_speed_y = -5.0
        self._ped_max_speed_y = 5.0
        self._social_state_num = 99

        is_action_space_discrete = rospy.get_param(
            "rl_agent/action_space/discrete", False
        )
        actions = (
            rospy.get_param("actions/discrete")
            if is_action_space_discrete
            else rospy.get_param("actions/continuous")
        )

        _action_space_kwargs = {
            "holonomic": self._is_holonomic,
            "is_discrete": is_action_space_discrete,
            "actions": actions,
            **action_space_kwargs,
        }

        _observation_kwargs = {
            "min_linear_vel": -2.0,
            "max_linear_vel": 2.0,
            "min_angular_vel": -4.0,
            "max_angular_vel": 4.0,
            "laser_num_beams": self._laser_num_beams,
            "laser_max_range": self._laser_max_range,
            "ped_num_types": self._num_ped_types,
            "ped_min_speed_x": self._ped_min_speed_x,
            "ped_max_speed_x": self._ped_max_speed_x,
            "ped_min_speed_y": self._ped_min_speed_y,
            "ped_max_speed_y": self._ped_max_speed_y,
            "ped_social_state_num": self._social_state_num,
            **observation_space_kwargs,
        }

        super().__init__(
            action_space_kwargs=_action_space_kwargs,
            observation_space_list=observation_space_list,
            observation_space_kwargs=_observation_kwargs,
        )

    @property
    def required_observations(self):
        """
        Returns the required observation units from Arena-Rosnav.
        """
        return get_required_observations(self._observation_space_manager.space_list)
