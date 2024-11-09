import os
from typing import List, Optional, Union

import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rosnav_rl.cfg import AgentCfg, PPO_Algorithm_Cfg, PPO_Policy_Cfg
from rosnav_rl.spaces import BaseObservationSpace
from rosnav_rl.utils.stable_baselines3.config import check_batch_size
from rosnav_rl.utils.utils import load_yaml

from ..model import RL_Model
from .policy.base_policy import PolicyType, StableBaselinesPolicy

DEVICE_CPU = "cpu"
DEVICE_AUTO = "auto"


class StableBaselinesAgent(RL_Model):
    _model: Union[PPO, RecurrentPPO]
    _model_cfg: PPO_Policy_Cfg
    _algorithm_cfg: PPO_Algorithm_Cfg

    def __init__(self, model_cfg: PPO_Policy_Cfg, algorithm_cfg: PPO_Algorithm_Cfg):
        super().__init__(model_cfg, algorithm_cfg)
        self._setup_agent_factory_and_policy_description()

    def _setup_agent_factory_and_policy_description(self):
        import rosnav_rl.model.stable_baselines3 as sb3_pkg

        self._agent_factory = sb3_pkg.import_models()
        self._policy_description: StableBaselinesPolicy = (
            self._agent_factory.instantiate(self.model_cfg.architecture_name)
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
        algorithm_args = self._setup_algorithm_arguments(
            self.algorithm_cfg, env, no_gpu, tensorboard_log_path
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
        model_path = os.path.join(dirpath, f"{file_name}.zip")
        print(f"Saving model to: {model_path}")

        self._model.save(model_path)
        self._save_vec_normalize(dirpath, file_name)

    def load(self, path: str, env: VecEnv, *args, **kwargs) -> None:
        self._model = self._load_model(path=path, env=env)

    def get_action(self, observation, *args, **kwargs):
        return self._model.predict(observation, deterministic=True)

    def train(self, *args, **kwargs) -> bool:
        try:
            self._model.learn(*args, **kwargs)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            return False
        return True

    def _get_vec_normalize(self) -> Optional[VecNormalize]:
        if isinstance(self.model.env, VecNormalize):
            return self.model.env
        if hasattr(self.model.env, "venv") and isinstance(
            self.model.env.venv, VecNormalize
        ):
            return self.model.env.venv
        return None

    def _save_vec_normalize(self, dirpath: str, file_name: str):
        vec_normalize = self._get_vec_normalize()
        if vec_normalize:
            vec_normalize_path = os.path.join(dirpath, f"vec_normalize_{file_name}.pkl")
            print(f"Saving VecNormalize to: {vec_normalize_path}")
            vec_normalize.save(vec_normalize_path)

    def _setup_algorithm_arguments(
        self,
        algorithm_cfg: PPO_Algorithm_Cfg,
        env: Union[VecEnv, gym.Env],
        no_gpu: bool,
        tensorboard_log_path: str,
    ) -> dict:
        check_batch_size(
            n_envs=env.num_envs,
            batch_size=self.algorithm_cfg.total_batch_size,
            mn_batch_size=self.algorithm_cfg.batch_size,
        )

        self.algorithm_cfg.n_steps = int(
            self.algorithm_cfg.total_batch_size / env.num_envs
        )

        return {
            "env": env,
            "policy": self._policy_description.type.value,
            "policy_kwargs": self._policy_description.get_kwargs(),
            "tensorboard_log": algorithm_cfg.tensorboard_log or tensorboard_log_path,
            "device": DEVICE_CPU if no_gpu else DEVICE_AUTO,
            **algorithm_cfg.model_dump(exclude=["total_batch_size", "tensorboard_log"]),
        }

    def _initialize_model(self, algorithm_parameters: dict) -> None:
        is_lstm = "LSTM" in self._policy_description.type.name
        model_class = RecurrentPPO if is_lstm else PPO
        self._model = model_class(**algorithm_parameters)

    def _load_model(
        self,
        path: str,
        env: Optional[VecEnv] = None,
        algorithm_args: Optional[dict] = None,
    ) -> Union[PPO, RecurrentPPO]:
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
        # TODO: Load configs and compare against parsed configs
        # cfg_path = os.path.splitext(path)[0]
        # train_cfg_dict = load_yaml(os.path.join(cfg_path, "training_config.yaml"))
        # train_cfg = AgentCfg.model_validate(
        #     train_cfg_dict["agent_cfg"], strict=True, from_attributes=True
        # )

        if self._policy_description.type == PolicyType.MULTI_INPUT:
            return PPO.load(path, env=env, custom_objects=algorithm_args)
        elif self._policy_description.type == PolicyType.MULTI_INPUT_LSTM:
            return RecurrentPPO.load(path, env=env, custom_objects=algorithm_args)
        else:
            raise ValueError(
                f"Unsupported policy type: {self._policy_description.type}"
            )

    @property
    def model_cfg(self) -> PPO_Policy_Cfg:
        return self._model_cfg

    @property
    def algorithm_cfg(self) -> PPO_Algorithm_Cfg:
        return self._algorithm_cfg

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
            "policy_cfg": self.model_cfg.model_dump() if self.model_cfg else {},
        }
