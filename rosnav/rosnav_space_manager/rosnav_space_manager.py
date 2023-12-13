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
    def __init__(self):
        self._stacked = rospy.get_param("rl_agent/frame_stacking/enabled")
        self._laser_num_beams = rospy.get_param("laser/num_beams")
        self._laser_max_range = rospy.get_param("laser/range")
        self._radius = rospy.get_param("robot_radius")
        self._is_holonomic = rospy.get_param("is_holonomic")

        is_action_space_discrete = rospy.get_param(
            "rl_agent/action_space/discrete", False
        )
        actions = (
            rospy.get_param("actions/discrete")
            if is_action_space_discrete
            else rospy.get_param("actions/continuous")
        )

        encoder_name = self._determine_encoder_name()

        self._encoder = BaseSpaceEncoderFactory.instantiate(
            encoder_name,
            laser_num_beams=self._laser_num_beams,
            laser_max_range=self._laser_max_range,
            radius=self._radius,
            is_holonomic=self._is_holonomic,
            actions=actions,
            is_action_space_discrete=is_action_space_discrete,
            stacked=self._stacked,
        )

    def _determine_encoder_name(self) -> str:
        if rospy.get_param("rl_agent/reduce_num_beams/enabled", False):
            return "ReducedLaserEncoder"
        if rospy.get_param("rl_agent/resnet", False):
            return "SemanticResNetSpaceEncoder"
        else:
            return "DefaultEncoder"

    def get_observation_space(self):
        return self._encoder.get_observation_space()

    def get_action_space(self):
        return self._encoder.get_action_space()

    def encode_observation(self, observation, structure):
        encoded_obs = self._encoder.encode_observation(observation, structure)

        return encoded_obs

    def decode_action(self, action):
        return self._encoder.decode_action(action)
