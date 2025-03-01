from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
from gym import spaces

from rosnav_rl.utils.type_aliases.rl_frameworks import SupportedRLFrameworks


import rosnav_rl.cfg as rosnav_rl_cfg
from rosnav_rl.model.stable_baselines3 import StableBaselinesModel
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager
from rosnav_rl.states import AgentStateContainer
from rosnav_rl.utils.type_aliases import ObservationDict
from rosnav_rl.model.dreamerv3.dreamerv3_model import DreamerV3Model

from .model import RL_Model


class RL_Agent:
    """A class representing a Reinforcement Learning Agent that manages model training and inference.

    This class serves as a high-level interface for reinforcement learning agents, handling model
    initialization, training, action selection, and space management. It works with both custom RL
    models and those from stable-baselines3.

    Attributes:
        _name (str): The identifier name of the agent.
        _model (Union[RL_Model, StableBaselinesModel]): The underlying RL model implementation.
        _reward_function (Optional[RewardFunction]): Function to calculate rewards during training.
        _space_manager (BaseSpaceManager): Manages observation and action spaces.
        _agent_state_container (AgentStateContainer): Contains and manages agent state information.

    Properties:
        config (Dict[str, dict]): Configuration dictionary containing model, space, and state settings.
        model (StableBaselinesModel): Access to the underlying RL model.
        reward_function (Union[None, RewardFunction]): Access to the reward function if defined.
        space_manager (BaseSpaceManager): Access to the space management system.
        observation_space (spaces.Dict): The agent's observation space.
        action_space (Union[spaces.Discrete, spaces.Box]): The agent's action space.
        agent_state_container (AgentStateContainer): Access to the agent's state container.
        name (str): The agent's identifier name.

    Example:
        agent = RL_Agent(agent_cfg, agent_state_container)
        agent.initialize_model()
        action = agent.get_action(observation)
        agent.train()
    """

    _name: str
    _model: Union[RL_Model, StableBaselinesModel]
    _reward_function: Optional[RewardFunction] = None
    _space_manager: BaseSpaceManager
    _agent_state_container: AgentStateContainer

    def __init__(
        self,
        agent_cfg: rosnav_rl_cfg.AgentCfg,
        agent_state_container: AgentStateContainer,
    ):
        """Initialize the RLAgent class.

        This class represents a reinforcement learning agent that can be trained and used for
        navigation tasks.

        Args:
            agent_cfg (AgentCfg): Configuration object containing all settings for the agent,
                including name, framework settings, action space configuration, and reward settings.
            agent_state_container (AgentStateContainer): Container object that maintains the
                agent's state information.

        Attributes:
            _name (str): Name identifier for the agent.
            _agent_state_container (AgentStateContainer): Reference to the state container.
            _model (StableBaselinesModel): The underlying RL model using stable-baselines3.
            _space_manager (BaseSpaceManager): Manages observation and action spaces.
            _reward_function (RewardFunction, optional): Function to calculate rewards,
                initialized if reward configuration is provided.

        Note:
            The agent is configured using the provided agent_cfg which should contain all
            necessary parameters for initialization including model architecture, reward
            function specifications, and space configurations.
        """
        self._name = agent_cfg.name
        self._agent_cfg = agent_cfg
        self._agent_state_container = agent_state_container

        if isinstance(agent_cfg.framework, rosnav_rl_cfg.StableBaselinesCfg):
            self._model = StableBaselinesModel(
                rl_agent=self,
                algorithm_cfg=agent_cfg.framework.algorithm,
            )
        elif isinstance(agent_cfg.framework, rosnav_rl_cfg.DreamerV3Cfg):
            self._model = DreamerV3Model(
                rl_agent=self, algorithm_cfg=agent_cfg.framework
            )
        else:
            raise ValueError(
                f"Unsupported RL algorithm: {agent_cfg.framework.__name__}"
            )
        self._space_manager = BaseSpaceManager(
            action_space_kwargs={"is_discrete": agent_cfg.action_space.is_discrete},
            agent_state_container=self._agent_state_container,
            observation_space_list=self.model.observation_space_list,
            observation_space_kwargs=self.model.observation_space_kwargs,
        )
        if agent_cfg.reward is not None:
            self._reward_function = RewardFunction(
                function_dict=agent_cfg.reward.reward_function_dict,
                unit_kwargs=agent_cfg.reward.reward_unit_kwargs,
                verbose=agent_cfg.reward.verbose,
            )

    def initialize_model(self, *args, **kwargs):
        """
        Initialize the model for the reinforcement learning agent.
        This method sets up the model with the provided arguments and keyword arguments.

        Args:
            *args: Variable length argument list to be passed to model setup.
            **kwargs: Arbitrary keyword arguments to be passed to model setup.

        Returns:
            None
        """
        self.model.setup_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """
        Loads a pre-trained model if it hasn't been initialized.

        This method checks if the model is already initialized and if not,
        loads it using the provided arguments.

        Args:
            *args: Variable length argument list to pass to model.load()
            **kwargs: Arbitrary keyword arguments to pass to model.load()

        Returns:
            None

        Note:
            The method uses the internal model's load() function and checks
            the is_model_initialized flag to prevent reloading.
        """
        if not self.model.is_model_initialized:
            self.model.load(*args, **kwargs)

    # def get_reward(self, observation: ObservationDict) -> float:
    #     """
    #     Calculate and return the reward based on the given observation.

    #     Args:
    #         observation (ObservationDict): The current observation containing relevant state information.

    #     Returns:
    #         float: The calculated reward based on the observation and the current simulation state.
    #     """
    #     return self._reward_function.get_reward(
    #         observation, simulation_state_container=self._simulation_state_container
    #     )

    def train(self, *args, **kwargs):
        """Train the reinforcement learning model.

        This method trains the underlying model with the provided arguments.

        Args:
            *args: Variable length argument list passed to the model's train method
            **kwargs: Arbitrary keyword arguments passed to the model's train method

        Returns:
            None

        Note:
            This is a wrapper around the model's train method and passes all arguments through directly
        """
        self.model.train(*args, **kwargs)

    def get_action(self, observation: ObservationDict, *args, **kwargs) -> np.ndarray:
        """
        Retrieves the action from the model based on the given observation.

        Args:
            observation (ObservationDict): Current observation of the environment state
            *args: Variable length argument list passed to model's get_action
            **kwargs: Arbitrary keyword arguments passed to model's get_action

        Returns:
            np.ndarray: Action vector selected by the model

        Note:
            This method serves as a wrapper around the model's get_action method,
            directly passing through all arguments and returning the model's action output.
        """
        return self.model.get_action(observation=observation, *args, **kwargs)

    @property
    def config(self) -> Dict[str, dict]:
        config_dict = {
            "agent_cfg": asdict(self._agent_cfg),
            "model": self.model.config,
            "space": self._space_manager.config,
            "agent_state_container": asdict(self.agent_state_container),
            # "simulation_state_container": asdict(self._simulation_state_container),
        }
        if self._reward_function is not None:
            config_dict["reward"] = self._reward_function.config
        return config_dict

    @property
    def model(self) -> StableBaselinesModel:
        if self._model is None:
            raise ValueError("'RL_Model' not initialized.")
        return self._model

    @property
    def reward_function(self) -> Union[None, RewardFunction]:
        return self._reward_function

    @property
    def space_manager(self) -> BaseSpaceManager:
        if self._space_manager is None:
            raise ValueError("'SpaceManager' not initialized.")
        return self._space_manager

    @property
    def observation_space(self) -> spaces.Dict:
        return self._space_manager.observation_space

    @property
    def action_space(self) -> Union[spaces.Discrete, spaces.Box]:
        return self._space_manager.action_space

    @property
    def agent_state_container(self) -> AgentStateContainer:
        return self._space_manager.agent_state_container

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent_cfg(self) -> "AgentCfg":
        return self._agent_cfg
