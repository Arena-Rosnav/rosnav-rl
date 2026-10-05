import pathlib
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import rosnav_rl.spaces.observation_space.spaces as spaces

from ...spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from ...utils.type_aliases.observation import ObservationDict
from ..dreamerv3 import tools
from ..model import RL_Model
from .dreamer import Dreamer, PolicyState
from .envs.wrappers import ChannelFirsttoLast, RenameObsForDreamer
from .helper import (
    create_agent,
    load_episodes,
    make_datasets,
    prefill_dataset,
    prepare_config,
    prepare_directories,
    prepare_logger,
    set_runtime_configuration,
    train,
)
from .parallel import Damy, Parallel

if TYPE_CHECKING:
    import rosnav_rl

    from .cfg import DreamerV3Cfg


class DreamerV3Model(RL_Model):
    """DreamerV3 model implementation for reinforcement learning.

    This class implements the DreamerV3 architecture for reinforcement learning.
    It handles model initialization, training, action prediction, and weight management.

    The model uses the world model approach where the agent learns to model the environment
    dynamics and then uses this model to plan and make decisions. This implementation
    integrates with a broader reinforcement learning framework.

    Attributes:
        _model: The DreamerV3 agent implementation
        _logger: Logging utility for training metrics and diagnostics
        _logdir: Path to the directory for storing logs and model weights

    Methods:
        setup_model: Initialize the DreamerV3 agent
        train: Train the model on environment interactions
        save: Save the model weights to disk
        load: Load model weights from disk
        get_action: Predict actions based on observations
        transfer_weights: Transfer weights between models (not implemented)
        observation_space_list: List of observation spaces required by the model
        observation_space_kwargs: Configuration for observation spaces
    """

    _model: Dreamer | None = None
    _logger: tools.Logger
    _logdir: pathlib.Path
    _policy_state: PolicyState | None = None

    def __init__(
        self,
        rl_agent: "rosnav_rl.RL_Agent",
        algorithm_cfg: "DreamerV3Cfg",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DreamerV3 Model.

        This method sets up the DreamerV3 model by initializing the base class,
        preparing the configuration directory, and setting up the logger.

        Args:
            rl_agent: The reinforcement learning agent instance
            algorithm_cfg: Configuration for the DreamerV3 algorithm
            *args: Additional positional arguments to pass to the parent class
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(rl_agent, algorithm_cfg, *args, **kwargs)

        if algorithm_cfg.general.logdir is None:
            algorithm_cfg.general.logdir = Path.cwd() / "agents"
        algorithm_cfg.general.logdir = (
            Path(algorithm_cfg.general.logdir) / rl_agent.name
        )
        self._logdir = prepare_config(algorithm_cfg)
        self._logger = prepare_logger(algorithm_cfg, self._logdir)

    def setup_model(
        self,
        train_dataset: Generator[dict[str, np.ndarray], None, None] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the DreamerV3 agent model.

        This method creates and configures the agent using the provided configuration,
        action space, observation space, logger, and training dataset.

        Args:
            train_dataset: Dataset used for training the model
            *args: Additional positional arguments passed to the underlying model
            **kwargs: Additional keyword arguments passed to the underlying model

        Returns:
            None

        Note:
            This method sets the internal _model attribute with the created agent.
        """
        self._model = create_agent(
            self._algorithm_cfg,
            self._rl_agent.action_space,
            self._rl_agent.observation_space,
            self._logger,
            train_dataset,
        )

    def train(
        self,
        train_envs: list[Parallel | Damy],
        eval_envs: list[Parallel | Damy],
        *args: Any,
        after_eval_fn: Callable[[dict[str, float]], None] | None = None,
        **kwargs: Any,
    ):
        """
        Train the DreamerV3 model using the provided simulation state container.

        This method performs the following steps:
        1. Configures runtime settings based on algorithm configuration
        2. Creates necessary directories for logging
        3. Loads training and evaluation episodes
        4. Prefills the dataset with initial experiences
        5. Sets up the model if not already initialized
        6. Loads the latest model checkpoint
        7. Runs the training process

        Args:
            simulation_state_container: Container for simulation state information
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            None
        """
        set_runtime_configuration(self._algorithm_cfg)
        prepare_directories(self._algorithm_cfg, self._logdir)
        train_eps, eval_eps = load_episodes(self._algorithm_cfg)

        action_space, observation_space = (
            self._rl_agent.action_space,
            self._rl_agent.observation_space,
        )

        state = prefill_dataset(
            self._algorithm_cfg,
            train_envs,
            train_eps,
            self._logger,
            action_space,
            observation_space,
        )

        train_dataset, eval_dataset = make_datasets(
            self._algorithm_cfg,
            train_eps,
            eval_eps,
        )

        # Setup model if not already initialized
        if self._model is None:
            self.setup_model(train_dataset)
        elif self._model.dataset is None:
            self._model.dataset = train_dataset
        else:
            train_dataset = self._model.dataset

        self.load("latest", missing_ok=True)
        train(
            self._algorithm_cfg,
            self._model,
            train_envs,
            eval_envs,
            train_eps,
            eval_eps,
            self._logger,
            eval_dataset,
            self._logdir,
            is_image_available=observation_space.get("image", None) is not None,
            state=state,
            log_wandb=True,
            after_eval_fn=after_eval_fn,
        )

    def save(self, file_name: str, *args: Any, **kwargs: Any):
        """
        Save the model's state dictionary and optimizer state dictionaries to a file.

        This method saves the state of the model and its optimizers to a PyTorch file (.pt)
        in the logging directory.

        Args:
            file_name (str): The name of the file to save the model to (without extension)
            *args: Variable length argument list (unused)
            **kwargs: Arbitrary keyword arguments (unused)

        Returns:
            None

        Example:
            >>> model.save("checkpoint_100k")  # Saves to /logdir/checkpoint_100k.pt
        """
        items_to_save = {
            "agent_state_dict": self._model.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(
                self._model
            ),
        }
        torch.save(items_to_save, self._logdir / f"{file_name}.pt")

    def load(self, file_name: str, *args: Any, missing_ok: bool = False, **kwargs: Any):
        """
        Load a pre-trained model from a checkpoint file.

        This method loads the model and optimizer states from a saved checkpoint file.
        It also resets the pretrain flag to ensure the model can continue training.

        Args:
            file_name (str): Name of the checkpoint file (without .pt extension)
            *args: Variable length argument list (unused)
            missing_ok (bool): Return without loading when the checkpoint does not exist
            **kwargs: Arbitrary keyword arguments (unused)

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint file does not exist and missing_ok is False
        """
        checkpoint_path = self._logdir / f"{file_name}.pt"
        if not checkpoint_path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(f"DreamerV3 checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self._model.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(self._model, checkpoint["optims_state_dict"])
        self._model._should_pretrain._once = False

    def get_action(self, observation: ObservationDict, *args: Any, **kwargs: Any) -> np.ndarray:
        """Encode one raw observation like the training env does and return the decoded action."""
        is_first = self._policy_state is None
        encoded = self._rl_agent.space_manager.encode_observation(
            {**observation, "is_first": is_first, "is_terminal": False}
        )
        obs = RenameObsForDreamer._remap(ChannelFirsttoLast._transform(dict(encoded)))
        batch = {key: np.asarray(value)[np.newaxis] for key, value in obs.items()}
        policy_output, self._policy_state = self._model(
            batch, np.array([is_first]), self._policy_state, training=False
        )
        return self._rl_agent.space_manager.decode_action(policy_output["action"][0].cpu().numpy())

    def reset(self) -> None:
        """Drop the latent state so the next action starts a new episode."""
        self._policy_state = None

    def transfer_weights(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @property
    def observation_space_list(self) -> list[type[BaseObservationSpace]]:
        """
        Returns the list of observation spaces used by the model.

        These spaces define the structure of the observation data expected by the model,
        including environmental information such as laser scans, pedestrian data,
        navigation goals, and episode state information.

        Returns:
            List[BaseObservationSpace]: A list of observation space classes that will be
            instantiated to create the actual observation spaces for the model.
        """
        return [
            spaces.environment.LaserCartesianMapSpace,
            spaces.environment.PedestrianVelXSpace,
            spaces.environment.PedestrianVelYSpace,
            spaces.environment.PedestrianTypeSpace,
            spaces.environment.PedestrianSocialStateSpace,
            spaces.navigation.DistAngleToGoalSpace,
            spaces.dynamics.LastActionSpace,
            spaces.meta.IsFirstStepSpace,
            spaces.meta.IsTerminalStepSpace,
        ]

    @property
    def observation_space_kwargs(self) -> dict[str, Any]:
        return {
            "reduced_num_beams": 72,  # 720 beams / 10 = 3.75° angular resolution
            "normalize": True,
            "goal_max_dist": 10,
            "roi_in_m": 40,
            "feature_map_size": 80,
            "laser_stack_size": 10,
        }
