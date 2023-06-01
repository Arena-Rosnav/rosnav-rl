import contextlib
import json
import os
import sys
import traceback

import numpy as np
import rospkg
import rospy
from rosnav import *
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import PolicyType
from rosnav.model.custom_sb3_policy import *
from rosnav.msg import ResetStackedObs
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from rosnav.srv import GetAction, GetActionResponse
from rosnav.utils.constants import VALID_CONFIG_NAMES
from rosnav.utils.utils import load_json, load_vec_normalize, load_yaml, make_mock_env
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from tools.ros_param_distributor import (
    populate_discrete_action_space,
    determine_space_encoder,
)

sys.modules["rl_agent"] = sys.modules["rosnav"]
sys.modules["rl_utils.rl_utils.utils"] = sys.modules["rosnav.utils"]


class RosnavNode:
    DEFAULT_DONES = np.array([[False]])
    DEFAULT_INFOS = [{}]
    DEFAULT_EPS_START = np.array([True])

    def __init__(self):
        # Agent name and path
        self.agent_name = rospy.get_param("agent_name")
        self.agent_path = RosnavNode._get_model_path(self.agent_name)

        assert os.path.isdir(
            self.agent_path
        ), f"Model cannot be found at {self.agent_path}"

        # Load hyperparams
        self._hyperparams = RosnavNode._load_hyperparams(self.agent_path)

        self._obs_structure = RosnavNode._get_observation_space_structure(
            self._hyperparams["rl_agent"]
        )

        with contextlib.suppress(KeyError):
            if self._hyperparams["rl_agent"]["action_space"]["discrete"]:
                populate_discrete_action_space(self._hyperparams)

        # Load observation normalization and frame stacking
        self.load_env_wrappers(self._hyperparams)

        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager()

        # Load the model
        self._agent = self._get_model(self._hyperparams, self.agent_path)

        self._get_next_action_srv = rospy.Service(
            "rosnav/get_action", GetAction, self._handle_next_action_srv
        )
        self._sub_reset_stacked_obs = rospy.Subscriber(
            "rosnav/reset_stacked_obs", ResetStackedObs, self._reset_stacked_obs
        )

        self.state = None
        self._reset_state = True

    def load_env_wrappers(self, hyperparams):
        # Load observation normalization and frame stacking
        self._normalized_mode = hyperparams["rl_agent"]["normalize"]
        self._reduced_laser_mode = (
            hyperparams["rl_agent"]["laser"]["reduce_num_beams"]["enabled"]
            if "laser" in hyperparams["rl_agent"]
            else False
        )
        self._stacked_mode = (
            hyperparams["rl_agent"]["frame_stacking"]["enabled"]
            if "frame_stacking" in hyperparams["rl_agent"]
            else False
        )

        if self._normalized_mode:
            self._vec_normalize = RosnavNode._get_vec_normalize(
                self.agent_path, self._hyperparams
            )

        if self._stacked_mode:
            self._stacked_obs_container = RosnavNode._get_stacked_obs(self._hyperparams)

        # populate right encoder name based on params
        rospy.set_param(
            "space_encoder",
            determine_space_encoder(self._stacked_mode, self._reduced_laser_mode),
        )

    def _encode_observation(self, obs_msg: GetAction):
        return self._encoder.encode_observation(
            {
                "goal_in_robot_frame": obs_msg.goal_in_robot_frame,
                "laser_scan": obs_msg.laser_scan,
                "last_action": obs_msg.last_action,
            },
            self._obs_structure,
        )

    def _handle_next_action_srv(self, request: GetAction):
        observation = self._encode_observation(request)

        if self._normalized_mode:
            try:
                observation = self._vec_normalize.normalize_obs(observation)
            except ValueError as e:
                rospy.logerr(e)
                rospy.logerr(
                    "Check if the configuration file correctly specifies the observation space."
                )
                rospy.signal_shutdown("")

        if self._stacked_mode:
            observation, _ = self._stacked_obs_container.update(
                observation, self.DEFAULT_DONES, self.DEFAULT_INFOS
            )

        predict_dict = {"observation": observation, "deterministic": True}

        if self._recurrent_arch:
            predict_dict.update(
                {
                    "state": self.state,
                    "episode_start": self.DEFAULT_EPS_START
                    if self._reset_state
                    else None,
                }
            )

        action, self.state = self._agent.predict(**predict_dict)

        decoded_action = self._encoder.decode_action(action)

        response = GetActionResponse()
        response.action = decoded_action

        self._reset_state = False

        return response

    def _reset_stacked_obs(self, request: GetAction):
        self._reset_state = True

        if self._stacked_mode:
            observation = self._encode_observation(request)
            self._stacked_obs_container.reset(observation)

    def _get_model(self, hyperparams: dict, agent_path: str):
        action_state_sizes = [0, 3]

        try:
            arch_name = hyperparams["rl_agent"]["architecture_name"]
        except KeyError:
            net_type = None
        finally:
            net_type: PolicyType = AgentFactory.registry[arch_name].type

        for size in action_state_sizes:
            rospy.set_param(f"{rospy.get_namespace()}action_state_size", size)
            try:
                if not net_type or net_type != PolicyType.MLP_LSTM:
                    self._recurrent_arch = False
                    return PPO.load(os.path.join(agent_path, "best_model.zip")).policy
                else:
                    self._recurrent_arch = True
                    return RecurrentPPO.load(
                        os.path.join(agent_path, "best_model.zip")
                    ).policy
            except:
                pass

        rospy.signal_shutdown("")

    @staticmethod
    def _get_model_path(model_name):
        return os.path.join(rospkg.RosPack().get_path("rosnav"), "agents", model_name)

    @staticmethod
    def _load_hyperparams(agent_path):
        for cfg_name in VALID_CONFIG_NAMES:
            cfg_path = os.path.join(agent_path, cfg_name)
            if os.path.isfile(cfg_path):
                if cfg_name.endswith(".json"):
                    cfg_dict = {"rl_agent": load_json(cfg_path)}
                elif cfg_name.endswith(".yaml"):
                    cfg_dict = load_yaml(cfg_path)

                assert cfg_dict is not None, "Config file is empty."
                return cfg_dict
        raise ValueError("No valid config file found in agent folder.")

    @staticmethod
    def _get_observation_space_structure(hyperparams):
        return hyperparams.get(
            "observation_space",
            ["laser_scan", "goal_in_robot_frame", "last_action"],
        )

    @staticmethod
    def _get_vec_normalize(agent_path, hyperparams):
        vec_normalize_path = os.path.join(agent_path, "vec_normalize.pkl")
        return load_vec_normalize(vec_normalize_path, hyperparams["rl_agent"])

    @staticmethod
    def _get_stacked_obs(hyperparams: dict):
        venv = make_mock_env(hyperparams["rl_agent"])
        return StackedObservations(
            1,
            hyperparams["rl_agent"]["frame_stacking"]["stack_size"],
            venv.observation_space,
            "first",
        )


if __name__ == "__main__":
    rospy.init_node("rosnav_node")

    node = RosnavNode()

    while not rospy.is_shutdown():
        rospy.spin()
