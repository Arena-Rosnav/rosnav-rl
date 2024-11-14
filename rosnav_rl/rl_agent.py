from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
from gym import spaces
from rl_utils.state_container import SimulationStateContainer

if TYPE_CHECKING:
    from rosnav_rl.cfg import AgentCfg
from rosnav_rl.model.stable_baselines3 import StableBaselinesAgent
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager
from rosnav_rl.spaces.space_manager.rosnav_space_manager import RosnavSpaceManager
from rosnav_rl.utils.agent_state import AgentStateContainer
from rosnav_rl.utils.type_aliases import ObservationDict

from .model import RL_Model


class RL_Agent:
    """
    RL_Agent is a reinforcement learning agent that integrates a model, reward function,
    space manager, and simulation state container to interact with an environment.

    Attributes:
        model (RL_Model): The reinforcement learning model used by the agent.
        reward_function (Optional[RewardFunction]): The function used to calculate rewards.
        space_manager (BaseSpaceManager): Manages the action and observation spaces.
        simulation_state_container (SimulationStateContainer): Container for the simulation state.

    Methods:
        __init__(agent_cfg: AgentCfg, simulation_state_container: SimulationStateContainer):
            Initializes the RL_Agent with the given configuration and simulation state container.

        config() -> Dict[str, dict]:
            Returns the configuration of the agent, including model, reward, space, and state containers.

        observation_space() -> spaces.Dict:
            Returns the observation space managed by the space manager.

        action_space() -> Union[spaces.Discrete, spaces.Box]:
            Returns the action space managed by the space manager.

        agent_state_container() -> AgentStateContainer:
            Returns the agent state container managed by the space manager.

        get_reward(observation: ObservationDict) -> float:
            Calculates and returns the reward for a given observation.

        get_action(observation: ObservationDict) -> np.ndarray:
            Returns the action for a given observation by encoding the observation,
            getting the action from the model, and decoding the action.
    """

    _name: str
    _model: RL_Model
    _reward_function: Optional[RewardFunction] = None
    _space_manager: BaseSpaceManager
    _simulation_state_container: SimulationStateContainer

    def __init__(
        self,
        agent_cfg: "AgentCfg",
        simulation_state_container: SimulationStateContainer,
    ):
        """
        Initialize the Reinforcement Learning Agent.

        Args:
            agent_cfg (AgentCfg): Configuration for the agent.
            simulation_state_container (SimulationStateContainer): Container for the simulation state.
            name (str, optional): Name of the agent. Defaults to None.

        Attributes:
            _name (str): Name of the agent.
            _simulation_state_container (SimulationStateContainer): Container for the simulation state.
            _model (StableBaselinesAgent): The framework-specific RL model used by the agent.
            _space_manager (RosnavSpaceManager): Manages the action and observation spaces.
            _reward_function (RewardFunction, optional): The reward function used by the agent, if specified in the configuration.
        """
        self._name = agent_cfg.name
        self._simulation_state_container = simulation_state_container
        self._model = StableBaselinesAgent(
            algorithm_cfg=agent_cfg.framework.algorithm,
        )
        self._space_manager = RosnavSpaceManager(
            action_space_kwargs={"is_discrete": agent_cfg.action_space.is_discrete},
            simulation_state_container=simulation_state_container,
            observation_space_list=self.model.observation_space_list,
            observation_space_kwargs=self.model.observation_space_kwargs,
        )
        if agent_cfg.reward is not None:
            self._reward_function = RewardFunction(
                reward_file_name=agent_cfg.reward.file_name,
                reward_unit_kwargs=agent_cfg.reward.reward_unit_kwargs,
                verbose=agent_cfg.reward.verbose,
            )

    def initialize_model(self, *args, **kwargs):
        """
        Initialize the model if it has not been initialized yet.

        This method checks if the model is already initialized. If not, it calls the
        model's initialize method with the provided arguments.

        Args:
            *args: Variable length argument list to be passed to the model's initialize method.
            **kwargs: Arbitrary keyword arguments to be passed to the model's initialize method.
        """
        if not self.model.is_model_initialized:
            self.model.setup_model(*args, **kwargs)

    def get_reward(self, observation: ObservationDict) -> float:
        """
        Calculate and return the reward based on the given observation.

        Args:
            observation (ObservationDict): The current observation containing relevant state information.

        Returns:
            float: The calculated reward based on the observation and the current simulation state.
        """
        return self._reward_function.get_reward(
            observation, simulation_state_container=self._simulation_state_container
        )

    def get_action(self, observation: ObservationDict) -> np.ndarray:
        return self._space_manager.decode_action(
            self.model.get_action(self.space_manager.encode_observation(observation))
        )

    @property
    def config(self) -> Dict[str, dict]:
        return {
            "model": self.model.config,
            "reward": self._reward_function.config,
            "space": self._space_manager.config,
            "agent_state_container": asdict(self.agent_state_container),
            "simulation_state_container": asdict(self._simulation_state_container),
        }

    @property
    def model(self) -> RL_Model:
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
