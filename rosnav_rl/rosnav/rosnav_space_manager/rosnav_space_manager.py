from .encoder_factory import BaseSpaceEncoderFactory
from .default_encoder import DefaultEncoder
from .reduced_laser_encoder import ReducedLaserEncoder
from .resnet_space_encoder import SemanticResNetSpaceEncoder

import rospy


"""
    Provides a uniform interface between model and environment.

    Offers encoders to scale observations to observation space.
    Offers the action and observation space sizes
"""


class RosnavSpaceManager:
    """
    Manages the space encoding and decoding for the ROS navigation system.
    """

    def __init__(self):
        self._stacked = rospy.get_param("rl_agent/frame_stacking/enabled")
        self._laser_num_beams = rospy.get_param("laser/num_beams")
        self._laser_max_range = rospy.get_param("laser/range")
        self._radius = rospy.get_param("robot_radius")
        self._is_holonomic = rospy.get_param("is_holonomic")

        self._num_ped_types = 5

        is_action_space_discrete = rospy.get_param(
            "rl_agent/action_space/discrete", False
        )
        actions = (
            rospy.get_param("actions/discrete")
            if is_action_space_discrete
            else rospy.get_param("actions/continuous")
        )

        self._encoder = BaseSpaceEncoderFactory.instantiate(
            self._determine_encoder_name(),
            action_space_kwargs={
                "radius": self._radius,
                "holonomic": self._is_holonomic,
                "action_space_discrete": is_action_space_discrete,
                "actions": actions,
                "stacked": self._stacked,
            },
            observation_list=None,  # use default_observation_list
            observation_kwargs={
                "min_linear_vel": -2.0,
                "max_linear_vel": 2.0,
                "min_angular_vel": -4.0,
                "max_angular_vel": 4.0,
                "laser_num_beams": self._laser_num_beams,
                "laser_max_range": self._laser_max_range,
                "num_ped_types": self._num_ped_types,
            },
        )

    def _determine_encoder_name(self) -> str:
        """
        Determines the name of the encoder based on the ROS parameters.
        Returns:
            str: The name of the encoder.
        """
        if rospy.get_param("rl_agent/reduce_num_beams/enabled", False):
            return "ReducedLaserEncoder"
        if rospy.get_param("rl_agent/resnet", False):
            return "SemanticResNetSpaceEncoder"
        else:
            return "DefaultEncoder"

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

    def encode_observation(self, observation, structure=None):
        """
        Encodes the given observation using the space encoder.
        Args:
            observation (object): The observation to encode.
            structure (object, optional): The structure of the observation. Defaults to None.
        Returns:
            object: The encoded observation.
        """
        encoded_obs = self._encoder.encode_observation(observation, structure)
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
