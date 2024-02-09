import contextlib
import json
import os
import sys

import numpy as np
import rospkg
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS
import rospy
from rl_utils.utils.observation_collector.observation_manager import ObservationManager
from rosnav import *
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import PolicyType
from rosnav.model.custom_sb3_policy import *
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from rosnav.srv import GetAction, GetActionResponse
from rosnav.utils.constants import VALID_CONFIG_NAMES
from rosnav.utils.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav.utils.utils import (
    load_json,
    load_vec_normalize,
    load_yaml,
    make_mock_env,
    wrap_vec_framestack,
)
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from tools.ros_param_distributor import (
    determine_space_encoder,
    populate_discrete_action_space,
    populate_laser_params,
)
from std_msgs.msg import Int16


sys.modules["rl_agent"] = sys.modules["rosnav"]
sys.modules["rl_utils.rl_utils.utils"] = sys.modules["rosnav.utils"]

from typing import Any, Dict, List

from task_generator.shared import Namespace


class RosnavNode:
    DEFAULT_DONES = np.array([[False]])
    DEFAULT_INFOS = [{}]
    DEFAULT_EPS_START = np.array([True])

    def __init__(self, ns: Namespace = ""):
        self.ns = Namespace(ns)

        # Agent name and path
        self.agent_name = rospy.get_param("agent_name")
        self.agent_path = RosnavNode._get_model_path(self.agent_name)

        assert os.path.isdir(
            self.agent_path
        ), f"Model cannot be found at {self.agent_path}"

        # Load hyperparams
        self._hyperparams = RosnavNode._load_hyperparams(self.agent_path)

        self._setup_action_space(self._hyperparams)

        populate_laser_params(self._hyperparams)

        # Get Architecture Name and retrieve Observation spaces
        architecture_name = self._hyperparams["rl_agent"]["architecture_name"]
        agent: BaseAgent = AgentFactory.instantiate(architecture_name)
        observation_spaces: List[BaseObservationSpace] = agent.observation_spaces
        observation_spaces_kwargs = agent.observation_space_kwargs

        # Load observation normalization and frame stacking
        self.load_env_wrappers(self._hyperparams, agent)

        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager(
            space_encoder_class=DefaultEncoder,
            observation_spaces=observation_spaces,
            observation_space_kwargs=observation_spaces_kwargs,
            action_space_kwargs=None,
        )

        # Load the model
        self._agent = self._get_model(
            architecture_name=architecture_name,
            checkpoint_name=self._hyperparams["rl_agent"]["checkpoint"],
            agent_path=self.agent_path,
        )

        self._observation_manager = ObservationManager(self.ns)

        self._get_next_action_srv = rospy.Service(
            self.ns("rosnav/get_action"), GetAction, self._handle_next_action_srv
        )
        self._sub_reset_stacked_obs = rospy.Subscriber(
            "/scenario_reset", Int16, self._reset_stacked_obs
        )

        self.state = None
        self._last_action = [0, 0, 0]
        self._reset_state = True

    def _setup_action_space(self, hyperparams):
        is_action_space_discrete = (
            hyperparams["rl_agent"]["discrete_action_space"]
            if "discrete_action_space" in self._hyperparams["rl_agent"]
            else self._hyperparams["rl_agent"]["action_space"]["discrete"]
        )
        rospy.set_param("rl_agent/action_space/discrete", is_action_space_discrete)

        if is_action_space_discrete:
            populate_discrete_action_space(hyperparams)

    def load_env_wrappers(self, hyperparams, agent_description):
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

        if self._stacked_mode:
            self._vec_stacked = RosnavNode._get_vec_stacked(
                agent_description, self._hyperparams
            )
            self._stacked_obs_container = self._vec_stacked.stacked_obs
        else:
            self._vec_stacked = None

        if self._normalized_mode:
            self._vec_normalize = RosnavNode._get_vec_normalize(
                agent_description, self.agent_path, self._hyperparams, self._vec_stacked
            )

    def _encode_observation(self, observation: Dict[str, Any]):
        return self._encoder.encode_observation(observation)

    def get_action(self):
        observation = self._observation_manager.get_observations()
        observation[OBS_DICT_KEYS.LAST_ACTION] = self._last_action

        observation = self._encode_observation(observation)

        if self._stacked_mode:
            observation, _ = self._stacked_obs_container.update(
                observation, RosnavNode.DEFAULT_DONES, RosnavNode.DEFAULT_INFOS
            )

        if self._normalized_mode:
            try:
                observation = self._vec_normalize.normalize_obs(observation)
            except ValueError as e:
                rospy.logerr(e)
                rospy.logerr(
                    "Check if the configuration file correctly specifies the observation space."
                )
                rospy.signal_shutdown("")

        predict_dict = {"observation": observation, "deterministic": True}

        if self._recurrent_arch:
            predict_dict.update(
                {
                    "state": self.state,
                    "episode_start": (
                        RosnavNode.DEFAULT_EPS_START if self._reset_state else None
                    ),
                }
            )
            self._reset_state = False

        action, self.state = self._agent.predict(**predict_dict)

        decoded_action = self._encoder.decode_action(action)

        self._last_action = decoded_action

        return decoded_action

    def _handle_next_action_srv(self, request: GetAction):
        action = self.get_action()

        response = GetActionResponse()
        response.action = action

        return response

    def _reset_stacked_obs(self, request: GetAction):
        self._reset_state = True

        if self._stacked_mode:
            observation = self._encode_observation(request)
            self._stacked_obs_container.reset(observation)

    def _get_model(self, architecture_name: str, checkpoint_name: str, agent_path: str):
        net_type: PolicyType = AgentFactory.registry[architecture_name].type
        model_path = os.path.join(agent_path, f"{checkpoint_name}.zip")

        if not net_type or net_type != PolicyType.MLP_LSTM:
            self._recurrent_arch = False
            return PPO.load(model_path).policy
        else:
            self._recurrent_arch = True
            return RecurrentPPO.load(model_path).policy

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
    def _get_vec_normalize(agent_description, agent_path, hyperparams, venv=None):
        if venv is None:
            venv = make_mock_env(agent_description)
        checkpoint = hyperparams["rl_agent"]["checkpoint"]
        vec_normalize_path = os.path.join(agent_path, f"vec_normalize_{checkpoint}.pkl")
        return load_vec_normalize(vec_normalize_path, hyperparams["rl_agent"], venv)

    @staticmethod
    def _get_vec_stacked(agent_description, hyperparams: dict):
        venv = make_mock_env(agent_description)
        return wrap_vec_framestack(
            venv, hyperparams["rl_agent"]["frame_stacking"]["stack_size"]
        )


if __name__ == "__main__":
    rospy.init_node("rosnav_node")

    node = RosnavNode()

    while not rospy.is_shutdown():
        rospy.spin()
