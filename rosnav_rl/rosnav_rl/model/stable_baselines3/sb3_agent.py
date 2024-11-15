from pathlib import Path
from typing import List, Optional, Union

import gym
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

import rosnav_rl.cfg.sb3_cfg as sb3_cfg
from rosnav_rl.spaces import BaseObservationSpace
from rosnav_rl.utils.stable_baselines3.config import check_batch_size
from rosnav_rl.utils.stable_baselines3.model.learning_rate_schedules import (
    load_lr_schedule,
)
from rosnav_rl.utils.stable_baselines3.transfer import transfer_weights
from rosnav_rl.utils.type_aliases import _SupportedStableBaselinesModels
from rosnav_rl.utils.utils import load_yaml

# from rosnav_rl.cfg.sb3_cfg import
from ..model import RL_Model
from .policy.agent_factory import AgentFactory
from .policy.base_policy import POLICY_TYPE, StableBaselinesPolicy

DEVICE_CPU = "cpu"
DEVICE_AUTO = "auto"


class StableBaselinesAgent(RL_Model):
    """StableBaselinesAgent is a reinforcement learning agent that utilizes the Stable Baselines3 library to implement RL algorithms.

    Attributes:
        _algorithm_cfg (PPO_Algorithm_Cfg): Configuration for the algorithm.
    """

    _model: _SupportedStableBaselinesModels
    _algorithm_cfg: "sb3_cfg.BaseAlgorithmCfg"

    def __init__(self, algorithm_cfg: "sb3_cfg.BaseAlgorithmCfg"):
        """
        Initialize the SB3Agent with the given model and algorithm configurations.

        Args:
            algorithm_cfg (PPO_Cfg): Configuration for the PPO algorithm.
        """
        super().__init__(algorithm_cfg)
        self._setup_agent_factory_and_policy_description()

    def _setup_agent_factory_and_policy_description(self):
        import rosnav_rl.model.stable_baselines3 as sb3_pkg

        self._agent_factory: AgentFactory = sb3_pkg.import_models()
        self._policy_description: StableBaselinesPolicy = (
            self._agent_factory.instantiate(self.algorithm_cfg.architecture_name)
        )

    def setup_model(
        self,
        env: Union[VecEnv, gym.Env],
        no_gpu: Optional[bool] = False,
        tensorboard_log_path: Optional[str] = None,
        resume_model_file: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Set up the model for training or resuming from a checkpoint.

        Args:
            env (Union[VecEnv, gym.Env]): The environment to train the model on.
            no_gpu (Optional[bool]): If True, disable GPU usage. Defaults to False.
            tensorboard_log_path (Optional[str]): Path to save TensorBoard logs. Defaults to None.
            resume_model_file (Optional[str]): Path to a model file to resume training from. Defaults to None.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        algorithm_args = self._setup_algorithm_arguments(
            self.algorithm_cfg.parameters, env, no_gpu, tensorboard_log_path
        )
        if resume_model_file:
            self.model = self._load_model(
                path=resume_model_file,
                env=env,
                algorithm_args=algorithm_args,
            )
        else:
            self._initialize_model(algorithm_args)

    def save(self, dirpath: str, file_name: str, *args, **kwargs) -> None:
        """
        Save the model and its associated normalization parameters to the specified directory.

        Args:
            dirpath (str): The directory path where the model and normalization parameters will be saved.
            file_name (str): The base name of the file to save the model as (without extension).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        model_path = Path(dirpath) / f"{file_name}.zip"
        self._model.save(model_path)
        self._save_vec_normalize(dirpath, file_name)

    def load(self, path: str, env: VecEnv, *args, **kwargs) -> None:
        """
        Load a pre-trained model from the specified path and initialize it with the given environment.

        Args:
            path (str): The file path to the pre-trained model.
            env (VecEnv): The environment to initialize the model with.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._model = self._load_model(path=path, env=env)

    def get_action(self, observation, *args, **kwargs):
        raise NotImplementedError()
        return self._model.predict(observation, deterministic=True)

    def train(self, *args, **kwargs) -> bool:
        """
        Train the model using the provided arguments.

        Args:
            *args: Variable length argument list to be passed to the model's learn method.
            **kwargs: Arbitrary keyword arguments to be passed to the model's learn method.

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
        config = load_yaml(source_dir / cfg_file_name)
        validated_algorithm_cfg = sb3_cfg.BaseAlgorithmCfg.model_validate(
            config["agent_cfg"]["framework"]["algorithm"]
        )

        source_model = StableBaselinesAgent(
            algorithm_cfg=validated_algorithm_cfg
        )._load_model(Path(source_dir) / f"{source_checkpoint}")

        self.model.policy = transfer_weights(
            target_model=self.model.policy,
            source_model=source_model.policy,
            include=include,
            exclude=exclude,
        )

    def _get_vec_normalize(self) -> VecNormalize:
        """
        Retrieve the VecNormalize instance from the model's environment if it exists.

        This method checks if the model's environment or its nested environment (venv)
        is an instance of VecNormalize. If so, it returns the VecNormalize instance.
        Otherwise, it returns None.

        Returns:
            VecNormalize | None: The VecNormalize instance if found, otherwise None.
        """
        env = self.model.env
        if isinstance(env, VecNormalize):
            return env
        if hasattr(env, "venv") and isinstance(env.venv, VecNormalize):
            return env.venv
        return None

    def _save_vec_normalize(self, dirpath: Union[str, Path], file_name: str) -> None:
        """
        Save the VecNormalize instance to a file if it exists.

        Args:
            dirpath (Union[str, Path]): The directory path where the VecNormalize instance will be saved.
            file_name (str): The base name of the file to save the VecNormalize instance as (without extension).
        """
        vec_normalize = self._get_vec_normalize()
        if vec_normalize:
            vec_normalize_path = Path(dirpath) / f"vec_normalize_{file_name}.pkl"
            vec_normalize.save(vec_normalize_path)

    def _setup_algorithm_arguments(
        self,
        parameters: "sb3_cfg.BaseAlgorithmParameters",
        env: Union[VecEnv, gym.Env],
        no_gpu: bool,
        tensorboard_log_path: Optional[str],
    ) -> dict:
        """
        Set up the arguments required for initializing the PPO algorithm.

        Args:
            algorithm_cfg (PPO_Algorithm_Cfg): Configuration for the PPO algorithm.
            env (Union[VecEnv, gym.Env]): The environment to train the model on.
            no_gpu (bool): If True, disable GPU usage.
            tensorboard_log_path (Optional[str]): Path to save TensorBoard logs.

        Returns:
            dict: A dictionary containing the algorithm arguments.
        """
        check_batch_size(
            n_envs=env.num_envs,
            batch_size=parameters.total_batch_size,
            mn_batch_size=parameters.batch_size,
        )

        parameters.n_steps = parameters.total_batch_size // env.num_envs
        parameters.learning_rate = (
            load_lr_schedule(
                type=parameters.learning_rate["type"],
                settings=parameters.learning_rate["kwargs"],
            )
            if isinstance(parameters.learning_rate, dict)
            else parameters.learning_rate
        )

        return {
            "env": env,
            "policy": POLICY_TYPE[self._policy_description.algorithm_class],
            "policy_kwargs": self._policy_description.get_kwargs(),
            "tensorboard_log": tensorboard_log_path or parameters.tensorboard_log,
            "device": DEVICE_CPU if no_gpu else DEVICE_AUTO,
            **parameters.model_dump(exclude=["total_batch_size", "tensorboard_log"]),
        }

    def _initialize_model(self, algorithm_parameters: dict) -> None:
        """
        Initialize the PPO or Recurrent PPO model with the given parameters.

        Args:
            algorithm_parameters (dict): The parameters required to initialize the model.
        """
        self._model = self._policy_description.algorithm_class(**algorithm_parameters)

    def _load_model(
        self,
        path: str,
        env: Optional[VecEnv] = None,
        algorithm_args: Optional[dict] = None,
    ) -> _SupportedStableBaselinesModels:
        """
        Load a model from the specified path.

        Args:
            path (str): The path to the model file.
            env (Optional[VecEnv]): The environment to associate with the model. Defaults to None.

        Returns:
            Union[PPO, RecurrentPPO]: The loaded model, which can be either a PPO or RecurrentPPO instance.

        Raises:
            ValueError: If the policy type specified in self._policy_description is unsupported.
        """
        return self._policy_description.algorithm_class.load(
            path, env=env, custom_objects=algorithm_args
        )

    @property
    def observation_space_list(self) -> List[BaseObservationSpace]:
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
