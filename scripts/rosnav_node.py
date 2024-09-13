import argparse
import os
import sys
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np
import rospkg
import rospy
import torch as th
from rl_utils.topic import Namespace
from rl_utils.utils.observation_collector import ObservationDict
from rl_utils.utils.observation_collector.observation_manager import ObservationManager
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import PolicyType
from rosnav.model.custom_sb3_policy import *
from rosnav.model.sb3_policy.paper import *
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from rosnav.srv import GetAction, GetActionResponse
from rosnav.utils.constants import VALID_CONFIG_NAMES
from rosnav.utils.observation_space import EncodedObservationDict
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
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from std_msgs.msg import Int16
from task_generator.constants import Constants
from task_generator.utils import Utils
from tools.ros_param_distributor import (
    populate_discrete_action_space,
    populate_laser_params,
    populate_rgbd_params,
)


class RosnavNode:
    DEFAULT_DONES = np.array([[False]])
    DEFAULT_INFOS = [{}]
    DEFAULT_EPS_START = np.array([True])
    DEFAULT_EPS_NO_START = np.array([False])

    def __init__(self, ns: Namespace = None):
        """
        Initialize the RosnavNode class.

        Args:
            ns (Namespace, optional): The namespace for the node. Defaults to "".
        """
        # self.ns = Namespace(ns) if ns else Namespace(rospy.get_namespace()[:-1])
        self.ns = Namespace(ns) if ns else Namespace("/jackal")

        rospy.loginfo(f"Starting Rosnav-Node on {self.ns}")

        # Agent name and path
        self.agent_name = rospy.get_param("agent_name")
        self.agent_path = RosnavNode._get_model_path(self.agent_name)

        assert os.path.isdir(
            self.agent_path
        ), f"Model cannot be found at {self.agent_path}"

        # Load hyperparams
        self._hyperparams = RosnavNode._load_hyperparams(self.agent_path)
        rospy.set_param("rl_agent", self._hyperparams["rl_agent"])

        self._setup_action_space(self._hyperparams)

        populate_laser_params(self._hyperparams)
        populate_rgbd_params(self._hyperparams)

        # Get Architecture Name and retrieve Observation spaces
        architecture_name = self._hyperparams["rl_agent"]["architecture_name"]
        agent: BaseAgent = AgentFactory.instantiate(architecture_name)
        observation_spaces: List[BaseObservationSpace] = agent.observation_spaces
        observation_spaces_kwargs = agent.observation_space_kwargs

        rospy.loginfo("[RosnavNode] Setup action space and model settings.")

        # Load observation normalization and frame stacking
        self._load_env_wrappers(self._hyperparams, agent)

        rospy.loginfo("[RosnavNode] Loaded environment wrappers.")

        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager(
            ns=self.ns,
            observation_spaces=observation_spaces,
            observation_space_kwargs=observation_spaces_kwargs,
            action_space_kwargs=None,
        )

        # Load the model
        self._agent = self._get_model(
            agent_description=agent,
            checkpoint_name=self._hyperparams["rl_agent"]["checkpoint"],
            agent_path=self.agent_path,
        )
        self._agent.observation_space = (
            self._encoder.observation_space_manager.observation_space
        )
        self._agent.set_training_mode(False)

        obs_unit_kwargs = {
            # "subgoal_mode": self._hyperparams["rl_agent"].get("subgoal_mode", False),
            "ns_to_semantic_topic": rospy.get_param("/train_mode", False),
        }

        self._observation_manager = ObservationManager(
            self.ns,
            obs_structur=list(self._encoder.encoder.required_observations),
            obs_unit_kwargs=obs_unit_kwargs,
            is_single_env=True,
        )

        if Utils.get_simulator() == Constants.Simulator.UNITY:
            sleep(5)  # wait for unity collector unit to set itself up

        rospy.loginfo("[RosnavNode] Loaded model and ObsManager.")

        self._get_next_action_srv = rospy.Service(
            str(self.ns("rosnav/get_action")), GetAction, self._handle_next_action_srv
        )
        self._sub_reset_stacked_obs = rospy.Subscriber(
            "/scenario_reset", Int16, self._on_scene_reset
        )

        self.state = None
        self._episode_start = np.array([False for _ in range(1)])

        self._last_action = [0, 0, 0]
        self._reset_state = True
        self._is_reset = False

        while not rospy.is_shutdown():
            rospy.spin()

    def _setup_action_space(self, hyperparams: dict):
        is_action_space_discrete = (
            hyperparams["rl_agent"]["discrete_action_space"]
            if "discrete_action_space" in self._hyperparams["rl_agent"]
            else self._hyperparams["rl_agent"]["action_space"]["discrete"]
        )
        rospy.set_param("rl_agent/action_space/discrete", is_action_space_discrete)

        if is_action_space_discrete:
            populate_discrete_action_space(hyperparams)

    def _load_env_wrappers(self, hyperparams: dict, agent_description: BaseAgent):
        """
        Loads the environment wrappers based on the provided hyperparameters and agent description.

        Args:
            hyperparams (dict): The hyperparameters for the RL agent.
            agent_description (BaseAgent): The description of the agent.

        Returns:
            None
        """
        # Load observation normalization and frame stacking
        self._normalized_mode = hyperparams["rl_agent"]["normalize"]["enabled"]
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
                agent_description,
                self.agent_path,
                self._hyperparams,
                self._vec_stacked,
                ns=self.ns,
            )

    def _encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """
        Encodes the given observation using the encoder.

        Args:
            observation (Dict[str, Any]): The observation to be encoded.

        Returns:
            The encoded observation.
        """
        return self._encoder.encode_observation(observation, **kwargs)

    def _get_observation(self) -> ObservationDict:
        """
        Get the observation from the observation manager and append the last action.

        Returns:
            dict: The observation dictionary.
        """
        return self._observation_manager.get_observations()

    def get_action(self):
        """
        Get the action to be taken based on the current observation.

        Returns:
            The decoded action to be taken.
        """
        observation: EncodedObservationDict = self._encode_observation(
            self._get_observation(), is_done=self._is_reset
        )

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
                        RosnavNode.DEFAULT_EPS_START
                        if self._reset_state
                        else RosnavNode.DEFAULT_EPS_NO_START
                    ),
                }
            )
            self._reset_state = False

        action, self.state = self.predict(**predict_dict)

        decoded_action = self._encoder.decode_action(action)

        self._last_action = decoded_action

        return decoded_action

    def _handle_next_action_srv(self, request: GetAction):
        """
        Handles the service request to get the next action.

        Args:
            request (GetAction): The service request.

        Returns:
            GetActionResponse: The service response containing the next action.
        """
        action = self.get_action()

        response = GetActionResponse()
        response.action = action

        return response

    def _on_scene_reset(self, request: Int16):
        """
        Resets the last action and stacked observations.

        Args:
            request (Int16): The reset request.

        Returns:
            None
        """
        self._reset_last_action()
        self._reset_stacked_obs()

    def _reset_last_action(self):
        """
        Resets the last action to [0, 0, 0].
        """
        self._last_action = [0, 0, 0]

    def _reset_stacked_obs(self):
        """
        Resets the stacked observation.

        This method sets the `_reset_state` flag to True, clears the `state` variable,
        and resets the stacked observation container if the stacked mode is enabled.
        """
        self._reset_state = True
        self.state = None

        if self._stacked_mode:
            observation = self._encode_observation(
                self._get_observation(), is_done=True
            )
            self._stacked_obs_container.reset(observation)

    def _get_model(
        self, agent_description: BaseAgent, checkpoint_name: str, agent_path: str
    ) -> Union[ActorCriticPolicy, RecurrentActorCriticPolicy]:
        """
        Get the model based on the given architecture name, checkpoint name, and agent path.

        Args:
            architecture_name (str): The name of the architecture.
            checkpoint_name (str): The name of the checkpoint.
            agent_path (str): The path to the agent.

        Returns:
            policy: The loaded policy model.
        """
        net_type: PolicyType = agent_description.type
        model_path = os.path.join(agent_path, f"{checkpoint_name}.zip")

        custom_objects = {
            "policy_kwargs": agent_description.get_kwargs(
                observation_space_manager=self._encoder.observation_space_manager,
                stack_size=(
                    self._hyperparams["rl_agent"]["frame_stacking"]["stack_size"]
                    if self._hyperparams["rl_agent"]["frame_stacking"]["enabled"]
                    else 1
                ),
            )
        }

        if not net_type or net_type == PolicyType.MULTI_INPUT:
            self._recurrent_arch = False
            return PPO.load(model_path, custom_objects=custom_objects).policy
        else:
            self._recurrent_arch = True
            return RecurrentPPO.load(model_path, custom_objects=custom_objects).policy

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
    def _get_vec_normalize(
        agent_description: BaseAgent,
        agent_path: str,
        hyperparams: dict,
        venv=None,
        ns: Namespace = "",
    ):
        """
        Get the vector normalizer for the RL agent.

        Args:
            agent_description (str): Description of the agent.
            agent_path (str): Path to the agent.
            hyperparams (dict): Hyperparameters for the RL agent.
            venv (object, optional): Virtual environment. Defaults to None.

        Returns:
            object: Vector normalizer for the RL agent.
        """
        if venv is None:
            venv = make_mock_env(str(ns), agent_description)
        rospy.loginfo("[RosnavNode] Loaded mock env.")
        checkpoint = hyperparams["rl_agent"]["checkpoint"]
        vec_normalize_path = os.path.join(agent_path, f"vec_normalize_{checkpoint}.pkl")
        return load_vec_normalize(vec_normalize_path, venv)

    @staticmethod
    def _get_vec_stacked(
        agent_description: BaseAgent,
        hyperparams: dict,
        ns: Namespace = "",
    ):
        """
        Returns a vectorized environment with frame stacking.

        Args:
            agent_description (str): Description of the agent.
            hyperparams (dict): Hyperparameters for the RL agent.

        Returns:
            Vectorized environment with frame stacking.
        """
        venv = make_mock_env(str(ns), agent_description)
        return wrap_vec_framestack(
            venv, hyperparams["rl_agent"]["frame_stacking"]["stack_size"]
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if self._recurrent_arch:
            return self._predict_recurrent(
                observation=observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
        return self._predict(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        for key, value in observation.items():
            if value.ndim == 2:
                observation[key] = np.expand_dims(value, axis=0)

        with th.no_grad():
            actions = (
                self._agent._predict(
                    obs_as_tensor(observation, self._agent.device),
                    deterministic=deterministic,
                )
                .cpu()
                .numpy()
            )

        actions = np.clip(
            actions, self._agent.action_space.low, self._agent.action_space.high
        )

        return actions.squeeze(axis=0), self.state

    def _predict_recurrent(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        for key, value in observation.items():
            if value.ndim == 2:
                observation[key] = np.expand_dims(value, axis=0)

        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate(
                [np.zeros(self._agent.lstm_hidden_state_shape) for _ in range(1)],
                axis=1,
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(1)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(
                state[0], dtype=th.float32, device=self._agent.device
            ), th.tensor(state[1], dtype=th.float32, device=self._agent.device)
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self._agent.device
            )
            actions, states = self._agent._predict(
                obs_as_tensor(observation, self._agent.device),
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        actions = np.clip(
            actions, self._agent.action_space.low, self._agent.action_space.high
        )

        return actions.squeeze(axis=0), self.state


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ns", "--namespace", type=str, default=None)

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    rospy.init_node("rosnav_node")
    args = parse_args()

    RosnavNode(ns=args.namespace)
