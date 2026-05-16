from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecFrameStack,
    VecNormalize,
)

from rosnav_rl.utils.stable_baselines3.config import check_batch_size
from rosnav_rl.utils.stable_baselines3.model.learning_rate_schedules import (
    load_lr_schedule,
)
from rosnav_rl.utils.stable_baselines3.transfer import transfer_weights
from rosnav_rl.utils.stable_baselines3.vec_env import (
    apply_vec_framestack,
    apply_vec_normalize,
    get_vec_framestack,
    get_vec_normalize,
)
from rosnav_rl.utils.type_aliases import (
    ObservationDict,
    _SupportedStableBaselinesModels,
)
from rosnav_rl.utils.utils import load_yaml, make_mock_env

from ..model import RL_Model
from .policy.agent_factory import AgentFactory
from .policy.base_policy import POLICY_TYPE, StableBaselinesPolicyDescription

if TYPE_CHECKING:
    from rosnav_rl.model.stable_baselines3 import cfg as sb3_cfg
    from rosnav_rl.rl_agent import RL_Agent
    from rosnav_rl.spaces.observation_space.spaces.base_observation_space import (
        BaseObservationSpace,
    )

DEVICE_CPU = "cpu"
DEVICE_AUTO = "auto"


class StableBaselinesModelState:
    """
    A class to manage the state of a Stable Baselines model.

    This class maintains the model's internal state between inference steps,
    tracking observations, actions, and reset status.

    Attributes:
        last_observation (np.ndarray): The most recent observation from the environment. Default is None.
        last_action (np.ndarray): The most recent action taken by the model. Default is [0, 0, 0].
        _reset_state (bool): Internal flag indicating if the state should be reset. Default is True.
        model_state (Tuple[np.ndarray, ...]): The internal state of the model. Default is None.

    Methods:
        reset(): Resets the model state to its initial values.
        reset_state (property): Gets the current reset state and updates the internal flag.
        reset_state (setter): Sets the internal reset state flag.
    """

    last_observation: np.ndarray = None
    last_action: np.ndarray = np.ndarray([0, 0, 0])
    _reset_state: bool = True
    model_state: Tuple[np.ndarray, ...] = None

    def reset(self):
        self.reset_state = True
        self.model_state = None
        self.last_action = np.ndarray([0, 0, 0])

    @property
    def reset_state(self):
        if self._reset_state:
            self._reset_state = False
            return True
        return self._reset_state

    @reset_state.setter
    def reset_state(self, value: bool):
        self._reset_state = value


class StableBaselinesEnv:
    """
    A wrapper class for Stable Baselines 3 vector environments that handles normalization and frame stacking.

    This class provides an interface to work with vector environments from Stable Baselines 3,
    specifically handling observation normalization via VecNormalize and frame stacking via VecFrameStack.

    Attributes:
        _env (VecEnv): The underlying vector environment.
        _norm_wrapper (Union[None, VecNormalize]): Normalization wrapper if present, otherwise None.
        _stack_wrapper (Union[None, VecFrameStack]): Frame stacking wrapper if present, otherwise None.

    Methods:
        save_normalization: Save the normalization statistics to a file.
        load_normalization: Load normalization statistics from a file.
        normalize: Normalize an observation using the normalization wrapper.
        stack: Update the stacked observations with a new observation.
        reset: Reset the stacked observations with an initial observation.

    Properties:
        has_norm_wrapper: Check if normalization wrapper is present.
        has_stack_wrapper: Check if frame stacking wrapper is present.
        env: Get the underlying vector environment.
    """

    _env: VecEnv
    _norm_wrapper: Union[None, VecNormalize] = None
    _stack_wrapper: Union[None, VecFrameStack] = None

    def __init__(self, env: VecEnv):
        self._env = env
        self._norm_wrapper = get_vec_normalize(env)
        self._stack_wrapper = get_vec_framestack(env)

    def save_normalization(self, path: Union[str, Path]) -> None:
        if self.has_norm_wrapper:
            self._norm_wrapper.save(path)

    def load_normalization(self, path: Union[str, Path]) -> None:
        if self.has_norm_wrapper:
            self._norm_wrapper.load(path)
        else:
            raise ValueError("Normalization wrapper not found.")

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        if self._norm_wrapper is None:
            raise ValueError("Normalization wrapper not found.")
        return self._norm_wrapper.normalize_obs(observation)

    def stack(self, observation: np.ndarray) -> np.ndarray:
        # encode_observation returns unbatched obs; VecFrameStack expects (n_envs, ...)
        if isinstance(observation, dict):
            batched = {k: np.asarray(v)[np.newaxis] for k, v in observation.items()}
        else:
            batched = np.asarray(observation)[np.newaxis]

        stacked, infos = self._stack_wrapper.stacked_obs.update(
            observations=batched,
            dones=np.array([False] * self._env.num_envs),
            infos=[{}] * self._env.num_envs,
        )
        # Remove batch dim: _predict_non_recurrent expects (stack_size, obs_dim) not (1, stack_size, obs_dim)
        if isinstance(stacked, dict):
            return {k: v[0] for k, v in stacked.items()}, infos
        return stacked[0], infos

    def reset(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, dict):
            batched = {k: np.asarray(v)[np.newaxis] for k, v in observation.items()}
        else:
            batched = np.asarray(observation)[np.newaxis]
        return self._stack_wrapper.stacked_obs.reset(observation=batched)

    @property
    def has_norm_wrapper(self) -> bool:
        return self._norm_wrapper is not None

    @property
    def has_stack_wrapper(self) -> bool:
        return self._stack_wrapper is not None

    @property
    def env(self) -> VecEnv:
        return self._env


class StableBaselinesModel(RL_Model):
    """A reinforcement learning model implementation using the Stable Baselines 3 (SB3) framework.

    This class provides an interface for training, loading, and using different types of
    reinforcement learning algorithms from the Stable Baselines 3 library. It supports various
    features such as frame stacking, observation normalization, weight transfer between models,
    and both recurrent and non-recurrent policies.

        _model (_SupportedStableBaselinesModels): The underlying SB3 model instance.
        _algorithm_cfg (sb3_cfg.SBAlgorithmCfg): Configuration for the SB3 algorithm.
        __env (StableBaselinesEnv): The environment wrapper used for interacting with the model.
        __state (StableBaselinesModelState): Maintains the state of the model across steps.
        _agent_factory (AgentFactory): Factory for creating agent instances.
        _policy_description (StableBaselinesPolicyDescription): Description of the policy architecture.

    Properties:
        observation_space_list: List of observation spaces defined in the policy.
        observation_space_kwargs: Additional keyword arguments for the observation space.
        stack_size: The number of frames to stack, if frame stacking is used.
        parameter_number: The total number of parameters in the policy network.
        config: Dictionary containing the algorithm configuration.
        environment: The environment wrapper used by the model.
        model: The underlying SB3 model instance.
    """

    _model: _SupportedStableBaselinesModels = None
    _algorithm_cfg: "sb3_cfg.SBAlgorithmCfg" = None
    __env: StableBaselinesEnv = None
    __state: StableBaselinesModelState = StableBaselinesModelState()

    def __init__(self, rl_agent: "RL_Agent", algorithm_cfg: "sb3_cfg.SBAlgorithmCfg"):
        """
        Initialize the SB3Model.

        Args:
            rl_agent (RL_Agent): The reinforcement learning agent.
            algorithm_cfg (sb3_cfg.SBAlgorithmCfg): The configuration for the SB3 algorithm.

        sb3_cfg.SBAlgorithmCfg:
            architecture_name (str): The name of the architecture.
            checkpoint (Optional[str]): The checkpoint to load.
            transfer_weights (Optional[TransferWeightsCfg]): The configuration for transferring weights.
            parameters (SBAlgorithmParameters): The parameters for the algorithm.
            normalization (Optional[NormalizationCfg]): The configuration for normalization.
        """
        super().__init__(rl_agent, algorithm_cfg)
        self.__setup_agent_factory_and_policy_description()

    @classmethod
    def from_framework_cfg(cls, rl_agent, framework_cfg, *args, **kwargs):
        """Create a :class:`StableBaselinesModel` from a framework config.

        Extracts the ``algorithm`` sub-config from the SB3 framework envelope.
        """
        return cls(
            rl_agent=rl_agent,
            algorithm_cfg=framework_cfg.algorithm,
            *args,
            **kwargs,
        )

    def __setup_agent_factory_and_policy_description(self):
        """
        Sets up the agent factory and policy description for the reinforcement learning model.

        This method imports the necessary models from the `rosnav_rl.model.stable_baselines3` package,
        initializes the agent factory, and instantiates the policy description based on the provided
        algorithm configuration.

        Attributes:
            self._agent_factory (AgentFactory): The factory responsible for creating agent instances.
            self._policy_description (StableBaselinesPolicyDescription): The policy description instantiated by the agent factory.
        """
        self._agent_factory: AgentFactory = AgentFactory
        self._policy_description: StableBaselinesPolicyDescription = (
            self._agent_factory.instantiate(self.algorithm_cfg.architecture_name)
        )

    def setup_model(
        self,
        env: Union[VecEnv, gym.Env],
        no_gpu: Optional[bool] = False,
        tensorboard_log_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Set up the RL model based on the specified environment and configuration.

        This method initializes or loads an RL model with the appropriate algorithm arguments
        based on the provided environment, GPU availability, and configuration settings.

        Args:
            env (Union[VecEnv, gym.Env]): The training environment for the model.
            no_gpu (Optional[bool], default=False): If True, disables GPU usage even if available.
            tensorboard_log_path (Optional[str], default=None): Path for TensorBoard logging.
            checkpoint_path (Optional[str], default=None): Path to load a pre-trained model checkpoint.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None: The method initializes the model internally but doesn't return it.

        Note:
            If checkpoint_path is provided, the model is loaded from the specified path.
            Otherwise, a new model is initialized using the configured algorithm parameters.
        """
        algorithm_args = self._setup_algorithm_arguments(
            self.algorithm_cfg.parameters, env, no_gpu, tensorboard_log_path
        )

        if checkpoint_path:
            self.model = self._load_model(
                path=checkpoint_path, env=env, algorithm_args=algorithm_args
            )
        else:
            self._initialize_model(algorithm_args)

    def save(self, dirpath: str, file_name: str) -> None:
        """
        Save the model and environment normalization to the specified directory.

        Args:
            dirpath (str): The directory path where the model and normalization data will be saved.
            file_name (str): The base name for the saved files.
        """
        model_path = Path(dirpath) / f"{file_name}.zip"
        self._model.save(model_path)
        self.__env.save_normalization(Path(dirpath) / f"vec_normalize_{file_name}.pkl")

    def load(self, path: str, env: VecEnv = None) -> None:
        """
        Load a pre-trained model from the specified path.

        Args:
            path (str): The file path to the pre-trained model.
            env (VecEnv, optional): The environment to which the model should be loaded. Defaults to None.
        """
        self._model = self._load_model(path=path, env=env)

    def get_action(
        self,
        observation: ObservationDict,
        deterministic: bool = True,
        is_first_observation: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Processes the observation and returns an action using the trained model.

        This method handles observation encoding, environment initialization if needed,
        stacking and normalization of observations (if applicable), and prediction of actions.
        It manages the agent's internal state tracking as well.

        Args:
            observation (ObservationDict): The current observation from the environment
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.
            is_first_observation (bool, optional): Whether this is the first observation in an episode.
                Will reset the agent's internal state if True. Defaults to False.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: The action to take in the environment (decoded from the model's output)
        """
        if is_first_observation:
            self.reset()

        observation = self._rl_agent.space_manager.encode_observation(
            observation
        )

        if self.__env is None:
            self.__env = StableBaselinesEnv(
                make_mock_env(
                    ns="",
                    space_manager=self._rl_agent.space_manager,
                    stack_size=self._policy_description.stack_size,
                )
            )

        if self.__env.has_stack_wrapper:
            observation, _ = self.__env.stack(observation)

        if self.__env.has_norm_wrapper:
            observation = self.__env.normalize(observation)

        self.__state.last_observation = observation

        action, self.__state.model_state = self._predict(
            observation=observation,
            deterministic=deterministic,
            state=self.__state.model_state,
            episode_start=(
                np.array([True] * self.__env.env.num_envs)
                if self.__state.reset_state
                else None
            ),
        )

        self.__state.last_action = self._rl_agent.space_manager.decode_action(action)
        return self.__state.last_action

    def train(self, *args, **kwargs) -> bool:
        """
        Train the model using the provided arguments.

        This method wraps the `learn` method of the underlying model and handles
        interruptions gracefully.

        Args:
            *args: Variable length argument list to be passed to the `learn` method.
            **kwargs: Arbitrary keyword arguments to be passed to the `learn` method.

        Returns:
            bool: True if training completes successfully, False if interrupted by the user.
        """
        try:
            self._model.learn(*args, **kwargs)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            return False
        return True

    def transfer_weights(
        self,
        source_dir: Union[str, Path],
        source_checkpoint: str,
        include: List[str] = None,
        exclude: List[str] = None,
        cfg_file_name: Optional[str] = "training_config.yaml",
    ) -> None:
        """
        Transfers weights from a source model checkpoint to the current model.

        Args:
            source_dir (Union[str, Path]): Directory containing the source model checkpoint.
            source_checkpoint (str): Filename of the source model checkpoint.
            include (List[str], optional): List of layer names to include in the transfer. Defaults to None.
            exclude (List[str], optional): List of layer names to exclude from the transfer. Defaults to None.
            cfg_file_name (Optional[str], optional): Name of the configuration file. Defaults to "training_config.yaml".

        Returns:
            None
        """
        import rosnav_rl.model.stable_baselines3.cfg as sb3_cfg

        config = load_yaml(source_dir / cfg_file_name)
        try:
            validated_algorithm_cfg = sb3_cfg.SBAlgorithmCfg.model_validate(
                config["agent_cfg"]["framework"]["algorithm"]
            )
        except Exception as e:
            print(f"Error validating algorithm configuration: {e}")
            validated_algorithm_cfg = sb3_cfg.PPO_Cfg(
                architecture_name="AGENT_1", parameters=sb3_cfg.PPO_Algorithm_Cfg()
            )

        source_model = StableBaselinesModel(
            rl_agent=self._rl_agent, algorithm_cfg=validated_algorithm_cfg
        )._load_model(Path(source_dir) / f"{source_checkpoint}")

        self.model.policy = transfer_weights(
            target_model=self.model.policy,
            source_model=source_model.policy,
            include=include,
            exclude=exclude,
        )

    def setup_environment(
        self, env: VecEnv, is_training: bool = True, *args, **kwargs
    ) -> VecEnv:
        """
        Set up the environment for training or evaluation.

        This method applies frame stacking and normalization to the given environment
        based on the configuration provided in `self.algorithm_cfg`.

        Args:
            env (VecEnv): The environment to be set up.
            is_training (bool, optional): Flag indicating whether the environment is for training or evaluation. Defaults to True.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            VecEnv: The modified environment with applied frame stacking and normalization.
        """
        if self.stack_size > 1:
            env = apply_vec_framestack(env, self.stack_size)
        if self.algorithm_cfg.normalization:
            env = apply_vec_normalize(
                env,
                path=self.algorithm_cfg.normalization.load_from,
                is_training=is_training,
                **self.algorithm_cfg.normalization.model_dump(exclude=["load_from"]),
            )
        return env

    def _setup_algorithm_arguments(
        self,
        parameters: "sb3_cfg.SBAlgorithmParameters",
        env: Union[VecEnv, gym.Env],
        no_gpu: bool,
        tensorboard_log_path: Optional[str],
    ) -> Dict[str, Any]:
        """
        Set up the arguments required for initializing the Stable Baselines 3 algorithm.

        Parameters from the config are filtered against the algorithm class's
        ``__init__`` signature so that fields which exist in the broad Pydantic
        base classes but are not accepted by a specific algorithm (e.g.
        ``batch_size`` for A2C, ``use_sde`` for TD3) are silently dropped.
        Fields whose value is ``None`` are also excluded so that SB3's own
        defaults are used instead of an explicit ``None`` override (e.g.
        ``n_steps=None`` for off-policy algorithms).

        Args:
            parameters (sb3_cfg.SBAlgorithmParameters): The parameters for the SB3 algorithm.
            env (Union[VecEnv, gym.Env]): The environment in which the algorithm will be trained.
            no_gpu (bool): Flag indicating whether to use GPU or not.
            tensorboard_log_path (Optional[str]): Path to the TensorBoard log directory.

        Returns:
            Dict[str, Any]: A dictionary containing the arguments for the SB3 algorithm.
        """
        import inspect

        from rosnav_rl.model.stable_baselines3.cfg.base import OnPolicyParameters

        parameters.learning_rate = load_lr_schedule(parameters.learning_rate)

        # On-policy algorithms derive n_steps from total_batch_size
        dump_exclude = {
            "tensorboard_log",
            "total_timesteps",
            "show_progress_bar",
        }

        if isinstance(parameters, OnPolicyParameters):
            check_batch_size(
                n_envs=env.num_envs,
                batch_size=parameters.total_batch_size,
                mn_batch_size=parameters.batch_size,
            )
            parameters.n_steps = parameters.total_batch_size // env.num_envs
            dump_exclude.add("total_batch_size")

        # Determine which kwargs the algorithm constructor actually accepts so
        # we never pass an unsupported parameter (e.g. `batch_size` to A2C or
        # `use_sde` to TD3).
        algo_cls = self._policy_description.algorithm_class
        algo_init_params = set(inspect.signature(algo_cls.__init__).parameters) - {"self"}

        # Always include these non-config kwargs that come from other sources.
        base_kwargs = {
            "env": env,
            "policy": POLICY_TYPE[algo_cls],
            "policy_kwargs": self._policy_description.get_kwargs(),
            "tensorboard_log": tensorboard_log_path or parameters.tensorboard_log,
            "device": DEVICE_CPU if no_gpu else DEVICE_AUTO,
        }

        # Dump config parameters, then filter to valid constructor kwargs and
        # drop any None values (so SB3 defaults are used for omitted params).
        config_dump = parameters.model_dump(exclude=dump_exclude)
        filtered_config = {
            k: v
            for k, v in config_dump.items()
            if k in algo_init_params and v is not None
        }

        return {**base_kwargs, **filtered_config}


    def _initialize_model(self, algorithm_parameters: Dict[str, Any]) -> None:
        """
        Initializes the model using the provided algorithm parameters.

        This method creates an instance of the algorithm class specified in the
        policy description, using the given algorithm parameters.

        Args:
            algorithm_parameters (Dict[str, Any]): A dictionary containing the parameters
                                         required to initialize the algorithm.
        """
        self._model = self._policy_description.algorithm_class(**algorithm_parameters)

    def _load_model(
        self,
        path: str,
        env: Optional[VecEnv] = None,
        algorithm_args: Optional[dict] = None,
    ) -> _SupportedStableBaselinesModels:
        """
        Load a pre-trained model from the specified path.

        Args:
            path (str): The path to the saved model.
            env (Optional[VecEnv]): The environment to which the model will be applied. Defaults to None.
            algorithm_args (Optional[dict]): Additional arguments for the algorithm. Defaults to None.

        Returns:
            _SupportedStableBaselinesModels: The loaded model.
        """
        if algorithm_args is None:
            algorithm_args = {}

        if env:
            algorithm_args["observation_space"] = env.observation_space

        # SB3 >= 2.3.0 calls torch.load(..., weights_only=True) which breaks
        # loading models saved with older SB3 or containing non-tensor objects
        # (e.g. numpy arrays).  Patch the load function used by SB3 to retry
        # transparently with weights_only=False on failure.
        import stable_baselines3.common.save_util as _sb3_save_util
        import torch as _th

        _orig_th_load = _th.load
        _orig_sb3_load = _sb3_save_util.th.load

        def _robust_load(*args, **kwargs):
            if kwargs.get("weights_only", False):
                try:
                    return _orig_th_load(*args, **kwargs)
                except Exception:
                    kwargs["weights_only"] = False
                    return _orig_th_load(*args, **kwargs)
            return _orig_th_load(*args, **kwargs)

        _th.load = _robust_load
        _sb3_save_util.th.load = _robust_load
        try:
            return self._policy_description.algorithm_class.load(
                path, env=env, custom_objects=algorithm_args
            )
        finally:
            _th.load = _orig_th_load
            _sb3_save_util.th.load = _orig_sb3_load

    @property
    def _is_recurrent(self) -> bool:
        """Check whether the current model uses a recurrent policy.

        The check is structural (``lstm_actor`` attribute on the policy)
        rather than based on ``isinstance`` so it remains valid for any
        future recurrent algorithm without code changes.
        """
        return (
            self._model is not None
            and hasattr(self._model.policy, "lstm_actor")
        )

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ):
        """Predict the action to take given an observation.

        Dispatches to the recurrent or non-recurrent prediction path based
        on whether the loaded policy carries LSTM layers.

        Args:
            observation: The observation input to the model.
            state: Hidden state for recurrent policies.
            episode_start: Boolean indicator for the start of an episode.
            deterministic: Use the mean action (greedy) instead of sampling.

        Returns:
            Tuple of ``(action, next_state)``.
        """
        if self._is_recurrent:
            return self._predict_recurrent(
                observation, state, episode_start, deterministic
            )
        return self._predict_non_recurrent(
            observation, state, episode_start, deterministic
        )

    def _predict_recurrent(
        self,
        observation: np.ndarray,
        state: Tuple[np.ndarray, ...],
        episode_start: Optional[np.ndarray] = None,
        deterministic: Optional[bool] = True,
    ):
        """Predict the next action using a recurrent policy.

        Args:
            observation: The current observation.
            state: The current hidden state of the recurrent policy.
            episode_start: Indicator for the start of an episode.
            deterministic: Use the mean action instead of sampling.

        Returns:
            ``(action, next_state)`` tuple.
        """
        return self.model.policy.predict(
            observation, state, episode_start, deterministic
        )

    def _predict_non_recurrent(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ):
        """
        Predict actions for a given observation using a non-recurrent policy.

        Args:
            observation (Union[np.ndarray, Dict[str, np.ndarray]]): The input observation(s) for the policy.
            state (Optional[Tuple[np.ndarray, ...]], optional): The hidden state(s) of the policy, if any. Defaults to None.
            episode_start (Optional[np.ndarray], optional): Indicator for the start of an episode. Defaults to None.
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.

        Returns:
            Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]: The predicted actions and the state.

        Note:
            We reimplement the predict function for non recurrent models because of issues with stacked
            observations with the official implementation.
        """
        for key, value in observation.items():
            if value.ndim == 2:
                observation[key] = np.expand_dims(value, axis=0)

        with th.no_grad():
            actions = (
                self._model.policy._predict(
                    obs_as_tensor(observation, self._model.device), deterministic
                )
                .cpu()
                .numpy()
            )

        actions = np.clip(
            actions, self._rl_agent.action_space.low, self._rl_agent.action_space.high
        )

        return actions.squeeze(axis=0), state

    def reset(self) -> None:
        """
        Resets the internal state of the model and the environment if necessary.

        This method resets the internal state of the model. If the environment has a stack wrapper and there is a
        last observation available in the state, it also resets the environment with the last observation.
        """
        self.__state.reset()
        if self.__env is not None and self.__env.has_stack_wrapper and self.__state.last_observation:
            self.__env.reset(self.__state.last_observation)

    @property
    def observation_space_list(self) -> List["BaseObservationSpace"]:
        """
        Returns a list of observation spaces defined in the policy description.

        Returns:
            List[BaseObservationSpace]: A list of observation spaces.
        """
        return self._policy_description.observation_spaces

    @property
    def observation_space_kwargs(self) -> dict:
        return self._policy_description.observation_space_kwargs

    @property
    def stack_size(self) -> int:
        return self._policy_description.stack_size

    @property
    def parameter_number(self) -> int:
        return sum(p.numel() for p in self.model.policy.parameters())

    @property
    def config(self):
        return {
            "algorithm_cfg": (
                self.algorithm_cfg.model_dump() if self.algorithm_cfg else {}
            ),
        }

    @property
    def environment(self) -> StableBaselinesEnv:
        """
        Returns the StableBaselinesEnv environment instance.

        Returns:
            StableBaselinesEnv: The environment instance used by the model.
        """
        return self.__env

    @environment.setter
    def environment(self, env: VecEnv):
        """
        Sets the environment for the model.

        Args:
            env (VecEnv): The environment to be used by the model. It should be an instance of VecEnv.

        Returns:
            None
        """
        self.__env = StableBaselinesEnv(env)
