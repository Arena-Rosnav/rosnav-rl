import pathlib
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import torch

from ...spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from ...utils.type_aliases import ObservationDict
from ..dreamerv3 import tools
from ..model import RL_Model
from .dreamer import Dreamer
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

DreamerEnvWrapper = Union[Parallel, Damy]


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

    _model: Dreamer = None
    _logger: tools.Logger = None
    _logdir: pathlib.Path = None
    _inference_only: bool = False
    _infer_state = None
    """Recurrent (latent, action, ctx_window) state carried across real-time
    ``get_action`` calls; cleared on :meth:`reset` (episode boundary)."""

    last_step_info: dict = {}
    """Per-step deploy diagnostics (e.g. ``kl_surprise``); populated by ``get_action`` only
    when ``behavior.expose_kl_surprise`` is enabled. Consumers (safety layer) read, never
    write. Empty dict when the flag is off — the deploy path stays byte-identical."""

    def __init__(
        self,
        rl_agent: "rosnav_rl.RL_Agent",
        algorithm_cfg: "DreamerV3Cfg",
        *args,
        inference_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the DreamerV3 Model.

        This method sets up the DreamerV3 model by initializing the base class,
        preparing the configuration directory, and setting up the logger.

        Args:
            rl_agent: The reinforcement learning agent instance
            algorithm_cfg: Configuration for the DreamerV3 algorithm
            *args: Additional positional arguments to pass to the parent class
            inference_only: When True, skip the training-only side effects
                (wandb/TensorBoard log-dir creation via ``prepare_config``/
                ``prepare_logger``) and put the model in ``eval()`` mode once
                built. Training call sites must not pass this.
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(rl_agent, algorithm_cfg, *args, **kwargs)
        self._inference_only = inference_only

        if algorithm_cfg.general.logdir is None:
            algorithm_cfg.general.logdir = Path.cwd() / "agents"
        algorithm_cfg.general.logdir = (
            Path(algorithm_cfg.general.logdir) / rl_agent.name
        )

        if inference_only:
            # No log-dir/wandb creation for a pure-inference construction;
            # self._logger is only ever read inside Dreamer.__call__'s
            # `if training:` branch, which inference (training=False) never hits.
            self._logdir = Path(algorithm_cfg.general.logdir)
            self._logger = None
        else:
            self._logdir = prepare_config(algorithm_cfg)
            self._logger = prepare_logger(algorithm_cfg, self._logdir)

    def setup_model(self, train_dataset: OrderedDict = None, *args, **kwargs):
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
        if self._inference_only:
            self._model.eval()

    @classmethod
    def inference_construction_kwargs(cls, agent_dir: Path) -> Dict[str, Any]:
        """Deploy construction needs ``inference_only=True`` (see ``__init__``)."""
        return {"inference_only": True}

    def load_for_inference(self, agent_dir: Path) -> None:
        """DreamerV3 needs ``setup_model()`` before a checkpoint can be loaded."""
        self.setup_model()
        self.load("latest")

    def train(
        self,
        train_envs: DreamerEnvWrapper,
        eval_envs: DreamerEnvWrapper,
        *args,
        after_eval_fn=None,
        **kwargs,
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

    def save(self, file_name: str, *args, **kwargs):
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

    def load(self, file_name: str, *args, missing_ok: bool = False, **kwargs):
        """
        Load a pre-trained model from a checkpoint file.

        This method loads the model and optimizer states from a saved checkpoint file.
        It also resets the pretrain flag to ensure the model can continue training.

        Args:
            file_name (str): Name of the checkpoint file (without .pt extension)
            *args: Variable length argument list (unused)
            missing_ok (bool): If True, silently skip loading when the checkpoint
                doesn't exist (used for fresh-training resume). If False, raise.
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
        tools.recursively_load_optim_state_dict(
            self._model, checkpoint["optims_state_dict"]
        )
        self._model._should_pretrain._once = False

        if self._inference_only:
            self._warmup()

    def _warmup(self) -> None:
        """Run one dummy forward pass so the first real ``get_action`` tick
        doesn't pay CUDA kernel/allocation latency (mirrors the SB3 backend's
        mock-env construction moved out of the hot path in ``load()``).

        Uses a random sample from the already-encoded observation space
        (``self._rl_agent.observation_space``, keyed identically to
        ``encode_observation``'s output) rather than a raw sensor dict, since
        no generic raw-observation mock exists for arbitrary space configs.
        The recurrent state produced here is discarded — it must not leak
        into the first real ``get_action`` call.
        """
        sample = self._rl_agent.observation_space.sample()
        batched = {key: np.asarray(value)[np.newaxis] for key, value in sample.items()}
        with torch.inference_mode():
            self._model(batched, reset=np.array([True]), state=None, training=False)

    def get_action(self, observation: "ObservationDict", *args, **kwargs) -> np.ndarray:
        """
        Predicts a decoded action from a raw observation, carrying recurrent
        state across calls (mirrors ``StableBaselinesModel.get_action``'s
        contract: raw observation in, decoded ``[vx, vy, wz]`` out).

        The underlying ``Dreamer.__call__`` expects a leading batch dimension
        (``_wm.preprocess`` does not insert one) and a required ``reset``
        array; both are supplied here. Recurrent state is stored on
        ``self._infer_state`` and cleared by :meth:`reset` at episode
        boundaries.

        Args:
            observation (ObservationDict): Raw (not yet encoded) observation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: Decoded action ready to publish.
        """
        encoded = self._rl_agent.space_manager.encode_observation(observation)
        batched = {key: np.asarray(value)[np.newaxis] for key, value in encoded.items()}

        with torch.inference_mode():
            policy_output, self._infer_state = self._model(
                batched,
                reset=np.array([False]),
                state=self._infer_state,
                training=False,
            )
            action = policy_output["action"].detach().cpu().numpy()
            # Attribute side-channel for the safety layer: present only when
            # behavior.expose_kl_surprise is on (see dreamer.py _policy).
            if "kl_surprise" in policy_output:
                self.last_step_info = {
                    "kl_surprise": float(policy_output["kl_surprise"].reshape(-1)[0])
                }

        return self._rl_agent.space_manager.decode_action(action.squeeze(axis=0))

    def reset(self) -> None:
        """Clears recurrent state carried across ``get_action`` calls.

        Must be invoked at episode boundaries (mirrors
        ``StableBaselinesModel.reset``); otherwise the recurrent latent from
        the previous episode would leak into the next one's first tick.
        """
        self._infer_state = None

    def transfer_weights(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def observation_space_list(self) -> List["BaseObservationSpace"]:
        """
        Returns the list of observation spaces used by the model.

        These spaces define the structure of the observation data expected by the model,
        including environmental information such as laser scans, pedestrian data,
        navigation goals, and episode state information.

        Resolved from ``algorithm_cfg.observation_space_list`` (names registered in
        ``SpaceFactory``), so a saved agent's space list is visible in its YAML and
        overridable, rather than hardcoded here.

        Returns:
            List[BaseObservationSpace]: A list of observation space classes that will be
            instantiated to create the actual observation spaces for the model.
        """
        from rosnav_rl.spaces.observation_space.observation_space_factory import (
            SpaceFactory,
        )

        return [
            SpaceFactory.registry[name]["class"]
            for name in self._algorithm_cfg.observation_space_list
        ]

    @property
    def observation_space_kwargs(self) -> Dict[str, Any]:
        return self._algorithm_cfg.observation_space_kwargs
