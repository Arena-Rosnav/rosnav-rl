from dataclasses import asdict
from typing import Dict, Optional, Union

import numpy as np
from gym import spaces
from rl_utils.state_container import SimulationStateContainer

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

    model: RL_Model
    reward_function: Optional[RewardFunction] = None
    space_manager: BaseSpaceManager
    simulation_state_container: SimulationStateContainer

    def __init__(
        self,
        agent_cfg: AgentCfg,
        simulation_state_container: SimulationStateContainer,
    ):
        self.simulation_state_container = simulation_state_container
        self.model = StableBaselinesAgent(policy_cfg=agent_cfg.policy)
        self.space_manager = RosnavSpaceManager(
            action_space_kwargs={"is_discrete": agent_cfg.action_space.is_discrete},
            simulation_state_container=simulation_state_container,
            observation_space_list=self.model.observation_space_list,
            observation_space_kwargs=self.model.observation_space_kwargs,
        )
        if agent_cfg.reward is not None:
            reward_cfg = agent_cfg.reward
            self.reward_function = RewardFunction(
                reward_file_name=reward_cfg.file_name,
                simulation_state_container=simulation_state_container,
                reward_unit_kwargs=reward_cfg.reward_unit_kwargs,
                verbose=reward_cfg.verbose,
            )

    @property
    def config(self) -> Dict[str, dict]:
        return {
            "model": self.model.config,
            "reward": self.reward_function.config,
            "space": self.space_manager.config,
            "agent_state_container": asdict(self.agent_state_container),
            "simulation_state_container": asdict(self.simulation_state_container),
        }

    @property
    def observation_space(self) -> spaces.Dict:
        return self.space_manager.observation_space

    @property
    def action_space(self) -> Union[spaces.Discrete, spaces.Box]:
        return self.space_manager.action_space

    @property
    def agent_state_container(self) -> AgentStateContainer:
        return self.space_manager.agent_state_container

    def get_reward(self, observation: ObservationDict) -> float:
        return self.reward_function.get_reward(observation)

    def get_action(self, observation: ObservationDict) -> np.ndarray:
        return self.space_manager.decode_action(
            self.model.get_action(self.space_manager.encode_observation(observation))
        )
