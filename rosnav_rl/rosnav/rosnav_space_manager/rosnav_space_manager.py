from typing import Any, Dict, List, Union

import rospy
from rosnav.rosnav_space_manager.encoder_wrapper.feature_map_recorder import (
    FeatureMapRecorderWrapper,
)
from rosnav.utils.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav.utils.observation_space.spaces.feature_maps.base_feature_map_space import (
    BaseFeatureMapSpace,
)

from .default_encoder import DefaultEncoder
from .encoder_factory import BaseSpaceEncoderFactory
from .encoder_wrapper.reduced_laser_wrapper import ReducedLaserWrapper

"""
    Provides a uniform interface between model and environment.

    Offers encoders to scale observations to observation space.
    Offers the action and observation space sizes
"""


class RosnavSpaceManager:
    """
    Manages the observation and action spaces for the ROS navigation system.
    """

    def __init__(
        self,
        space_encoder_class=DefaultEncoder,
        observation_spaces: List[
            Union[BaseObservationSpace, BaseFeatureMapSpace]
        ] = None,
        observation_space_kwargs: Dict[str, Any] = None,
        action_space_kwargs: Dict[str, Any] = None,
    ):
        """
        Initializes the RosnavSpaceManager. Retrieves the Agent parameters from the ROS parameter server and initializes the space encoder containing the observation and action spaces.

        Args:
            space_encoder_class (class, optional): The class for encoding the observation and action spaces. Defaults to None.
            observation_spaces (List[Union[BaseObservationSpace, BaseFeatureMapSpace]], optional): The list of observation spaces. Defaults to None.
            observation_space_kwargs (Dict[str, Any], optional): Additional keyword arguments for the observation spaces. Defaults to None.
            action_space_kwargs (Dict[str, Any], optional): Additional keyword arguments for the action spaces. Defaults to None.
        """
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
        self._is_holonomic = rospy.get_param_cached("is_holonomic")

        # TODO: add num_ped_types to rosparam
        self._num_ped_types = 5
        self._ped_min_speed_x = -5.0
        self._ped_max_speed_x = 5.0
        self._ped_min_speed_y = -5.0
        self._ped_max_speed_y = 5.0
        self._social_state_num = 99

        is_action_space_discrete = rospy.get_param_cached(
            "rl_agent/action_space/discrete", False
        )
        actions = (
            rospy.get_param_cached("actions/discrete")
            if is_action_space_discrete
            else rospy.get_param_cached("actions/continuous")
        )

        _action_space_kwargs = {
            "radius": self._radius,
            "holonomic": self._is_holonomic,
            "action_space_discrete": is_action_space_discrete,
            "actions": actions,
            "stacked": self._stacked,
            **action_space_kwargs,
        }

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

        self._encoder = space_encoder_class(
            action_space_kwargs=_action_space_kwargs,
            observation_list=observation_spaces,
            observation_kwargs=_observation_kwargs,
        )

        if rospy.get_param("laser/reduce_num_beams"):
            self._encoder = ReducedLaserWrapper(self._encoder, self._laser_num_beams)

        if rospy.get_param("record_feature_maps", False):
            self._encoder = FeatureMapRecorderWrapper(
                encoder=self._encoder, save_every_x_obs=4
            )

    @property
    def observation_space_manager(self):
        """
        Gets the observation space manager.
        Returns:
            object: The observation space manager.
        """
        return self._encoder.observation_space_manager

    def get_observation_space(self):
        """
        Gets the observation space.
        Returns:
            object: The observation space.
        """
        return self._encoder.observation_space

    def get_action_space(self):
        """
        Gets the action space.
        Returns:
            object: The action space.
        """
        return self._encoder.action_space

    def encode_observation(self, observation, *args, **kwargs):
        """
        Encodes the given observation using the space encoder.
        Args:
            observation (object): The observation to encode.
        Returns:
            object: The encoded observation.
        """
        encoded_obs = self._encoder.encode_observation(observation, **kwargs)
        return encoded_obs

    def decode_action(self, action):
        """
        Decodes the given action using the space encoder.
        Args:
            action (object): The action to decode.
        Returns:
            object: The decoded action.
        """
        return self._encoder.decode_action(action)
