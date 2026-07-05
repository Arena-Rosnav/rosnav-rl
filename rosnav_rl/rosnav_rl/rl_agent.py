from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from gymnasium import spaces

from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces.space_manager.base_space_manager import BaseSpaceManager
from rosnav_rl.utils.type_aliases import ObservationDict

from .model import RL_Model


class RL_Agent:
    """
    High-level interface for a Reinforcement Learning (RL) navigation agent.

    The RL_Agent class encapsulates the full lifecycle and interaction logic for a navigation agent powered by RL models. It is designed to be constructed exclusively from a validated AgentConfig specification, ensuring reproducibility and modularity across different frameworks and robot configurations.

    **Key Features:**
    - Unified interface for model initialization, training, inference, and reward computation
    - Supports multiple RL frameworks (Stable Baselines3, DreamerV3, custom RL_Model)
    - Handles observation and action spaces via BaseSpaceManager
    - Integrates reward function logic for environment feedback
    - Provides access to agent configuration, model, and space manager

    **Usage Example:**
        >>> from rosnav_rl.cfg.agent import AgentConfig
        >>> spec = AgentConfig.from_yaml("agent.yaml")
        >>> agent = RL_Agent(spec)
        >>> agent.initialize_model()
        >>> action = agent.get_action(observation)

    **Initialization:**
        RL_Agent(spec: AgentConfig)
        - spec: Validated AgentConfig instance containing robot, environment, framework, reward, and observation settings.
        - Internally constructs the RL model, space manager, and reward function.

    **Model Management:**
        - initialize_model(*args, **kwargs): Sets up the underlying RL model for training or inference.
        - load_model(*args, **kwargs): Loads a pre-trained model checkpoint if not already initialized.
        - train(*args, **kwargs): Trains the RL model using provided data or environment.

    **Action Selection:**
        - get_action(observation: ObservationDict, ...): Returns the action vector for a given observation, using the model's policy.

    **Reward Computation:**
        - reward_function: Accesses the RewardFunction instance (if specified in config).
        - (Optional) get_reward(observation): Computes reward for a given observation and simulation state.

    **Configuration Access:**
        - config: Returns a dictionary with agent, model, and space manager configuration.
        - spec: Returns the AgentConfig instance used for construction.
        - model: Returns the underlying RL model instance.
        - space_manager: Returns the BaseSpaceManager instance.
        - observation_space: Returns the Gymnasium Dict observation space.
        - action_space: Returns the Gymnasium action space (Discrete or Box).
        - name: Returns the agent's name.

    **Extensibility:**
        RL_Agent is designed for extension and integration with custom RL models, reward units, and observation generators. All major components are accessible via properties for advanced use cases.

    **Thread Safety:**
        RL_Agent is not inherently thread-safe. For concurrent environments, ensure proper synchronization when accessing model or reward function.

    **References:**
        - AgentConfig: rosnav_rl.cfg.agent.AgentConfig
        - RL_Model: rosnav_rl.model.RL_Model
        - StableBaselinesModel: rosnav_rl.model.stable_baselines3.StableBaselinesModel
        - DreamerV3Model: rosnav_rl.model.dreamerv3.DreamerV3Model
        - RewardFunction: rosnav_rl.reward.reward_function.RewardFunction
        - BaseSpaceManager: rosnav_rl.spaces.space_manager.base_space_manager.BaseSpaceManager
    """

    _name: str = ""
    _model: RL_Model
    _reward_function: Optional[RewardFunction] = None
    _space_manager: BaseSpaceManager

    def __init__(self, spec: AgentConfig, model_kwargs: Optional[Dict] = None):
        from rosnav_rl.model.model_factory import ModelFactory

        self._spec = spec
        self._name = spec.name or ""
        self._reward_function = None

        self._model = ModelFactory.create_model_instance(
            framework_cfg=spec.framework, rl_agent=self, **(model_kwargs or {}),
        )

        self._space_manager = BaseSpaceManager(
            spec=spec,
            observation_space_list=self.model.observation_space_list,
            observation_space_kwargs=self.model.observation_space_kwargs,
        )

        if spec.reward is not None:
            self._reward_function = RewardFunction(
                function_dict=spec.reward.reward_function_dict,
                unit_kwargs=spec.reward.reward_unit_kwargs,
                verbose=spec.reward.verbose,
            )

    @classmethod
    def from_agent_dir(cls, agent_dir: Union[str, Path]) -> "RL_Agent":
        """Load a saved agent for inference, dispatching via ``ModelFactory``.

        Unifies the SB3 vs. DreamerV3 deployment path (previously duplicated
        across ``arena_inference_node.py`` and ``action_server/arena_server.py``,
        both of which only knew how to load an SB3 ``best_model.zip``). Each
        backend's construction kwargs and load sequence are declared on its
        own ``RL_Model`` subclass (``inference_construction_kwargs`` /
        ``load_for_inference``) rather than branched here.

        Args:
            agent_dir: Directory containing ``training_config.yaml`` and the
                framework-specific checkpoint (``best_model.zip`` or
                ``latest.pt``).

        Returns:
            RL_Agent: Ready for ``get_action`` calls (already-decoded output).
        """
        from rosnav_rl.model.model_factory import ModelFactory
        from rosnav_rl.utils.agent_paths import load_agent_spec

        agent_dir = Path(agent_dir)
        spec = load_agent_spec(agent_dir)

        model_class = ModelFactory.get_model_class(spec.framework.name)
        agent = cls(
            spec, model_kwargs=model_class.inference_construction_kwargs(agent_dir)
        )
        agent.model.load_for_inference(agent_dir)

        return agent

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

    def reset(self) -> None:
        """
        Resets the agent's model state and observation space state for a new episode.

        Note:
            Must be called on every episode reset. Stateful observation spaces
            (e.g. stacked laser maps, pedestrian trajectory history) otherwise leak
            state across episode boundaries.
        """
        self.model.reset()
        self._space_manager.reset_spaces()

    @property
    def config(self) -> Dict[str, dict]:
        """Configuration dictionary for the agent."""
        config_dict = {
            "spec": self._spec.to_dict(),
            "model": self._model.config,
            "space": self._space_manager.config,
        }
        return config_dict

    @property
    def spec(self) -> AgentConfig:
        return self._spec

    @property
    def model(self) -> RL_Model:
        return self._model

    @property
    def reward_function(self) -> Union[None, RewardFunction]:
        return self._reward_function

    @property
    def space_manager(self) -> BaseSpaceManager:
        return self._space_manager

    @property
    def observation_space(self) -> spaces.Dict:
        return self._space_manager.observation_space

    @property
    def action_space(self) -> Union[spaces.Discrete, spaces.Box]:
        return self._space_manager.action_space

    @property
    def name(self) -> str:
        return self._name
