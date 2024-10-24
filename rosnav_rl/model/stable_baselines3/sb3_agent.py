import os
from typing import List, Optional, Union

import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rosnav_rl.cfg import PPO_Cfg, PPO_Policy_Cfg
from rosnav_rl.cfg.stable_baselines3.lr_schedule import LearningRateSchedulerCfg
from rosnav_rl.spaces import BaseObservationSpace
from rosnav_rl.utils.model.sb3.learning_rate_schedules import load_lr_schedule

from ..model import RL_Model
from .policy.base_policy import PolicyType, StableBaselinesPolicy

DEVICE_CPU = "cpu"
DEVICE_AUTO = "auto"


class StableBaselinesAgent(RL_Model):
    model: Union[PPO, RecurrentPPO] = None
    algorithm_cfg: PPO_Cfg = None

    def __init__(
        self,
        policy_cfg: PPO_Policy_Cfg,
    ):
        import rosnav_rl.model.stable_baselines3 as sb3_pkg

        self.policy_cfg = policy_cfg
        self._agent_factory = sb3_pkg.import_models()

        self._policy_description: StableBaselinesPolicy = (
            self._agent_factory.instantiate(
                self.policy_cfg.architecture_name,
            )
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

    def initialize(
        self,
        algorithm_cfg: PPO_Cfg,
        env: Union[VecEnv, gym.Env],
        no_gpu: Optional[bool] = False,
        tensorboard_log_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.algorithm_cfg = algorithm_cfg
        self._setup_learning_rate_scheduler(algorithm_cfg)
        self._initialize_model(
            self._setup_algorithm_config(
                algorithm_cfg, env, no_gpu, tensorboard_log_path
            )
        )

    def load(self, path: str, env: VecEnv, *args, **kwargs) -> None:
        self.model = self._load_model(
            path=path,
            env=env,
            policy_description=self._policy_description,
            checkpoint_name=self.policy_cfg.checkpoint,
        )

    def get_action(self, observation, *args, **kwargs):
        return self.model.predict(observation, deterministic=True)

    def train(self, *args, **kwargs):
        self.model.learn(*args, **kwargs)

    def save(self, path: str, checkpoint_name: str, *args, **kwargs) -> None:
        model_path = os.path.join(path, f"{checkpoint_name}.zip")
        vec_normalize_path = os.path.join(path, f"vec_normalize_{checkpoint_name}.pkl")

        print(f"Saving model to: {model_path}")
        self.model.save(model_path)

        vec_normalize = self._get_vec_normalize()
        if vec_normalize:
            print(f"Saving VecNormalize to: {vec_normalize_path}")
            vec_normalize.save(vec_normalize_path)
        else:
            print("No VecNormalize object found to save.")

    def _get_vec_normalize(self) -> Optional[VecNormalize]:
        if isinstance(self.model.env, VecNormalize):
            return self.model.env
        if hasattr(self.model.env, "venv") and isinstance(
            self.model.env.venv, VecNormalize
        ):
            return self.model.env.venv
        return None

    def _setup_algorithm_config(
        self,
        algorithm_cfg: PPO_Cfg,
        env: Union[VecEnv, gym.Env],
        no_gpu: bool,
        tensorboard_log_path: str,
    ) -> dict:
        algorithm_cfg.tensorboard_log = (
            algorithm_cfg.tensorboard_log or tensorboard_log_path
        )
        algorithm_cfg.device = DEVICE_CPU if no_gpu else DEVICE_AUTO
        return {
            "env": env,
            "policy": self._policy_description.type.value,
            "policy_kwargs": self._policy_description.get_kwargs(),
            **algorithm_cfg.model_dump(exclude=["total_batch_size"]),
        }

    def _setup_learning_rate_scheduler(self, algorithm_cfg: PPO_Cfg):
        if isinstance(algorithm_cfg.learning_rate, LearningRateSchedulerCfg):
            algorithm_cfg.learning_rate = load_lr_schedule(
                algorithm_cfg.learning_rate.type,
                algorithm_cfg.learning_rate.kwargs,
            )

    def _initialize_model(self, alg_kwargs: dict) -> None:
        is_lstm = "LSTM" in self._policy_description.type.name
        model_class = RecurrentPPO if is_lstm else PPO
        self.model = model_class(**alg_kwargs)

    def _load_model(
        self,
        path: str,
        policy_description: StableBaselinesPolicy,
        env: Optional[VecEnv] = None,
        checkpoint_name: str = "best_model",
    ) -> Union[PPO, RecurrentPPO]:
        # TODO: Perhaps check policy kwargs with previous settings
        model_path = os.path.join(path, f"{checkpoint_name}.zip")
        custom_objects = {"policy_kwargs": policy_description.get_kwargs()}

        if policy_description.type == PolicyType.MULTI_INPUT:
            return PPO.load(model_path, env=env, custom_objects=custom_objects)
        elif policy_description.type == PolicyType.MULTI_INPUT_LSTM:
            return RecurrentPPO.load(model_path, env=env, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unsupported policy type: {policy_description.type}")

    @property
    def config(self):
        return {
            "algorithm_cfg": (
                self.algorithm_cfg.model_dump() if self.algorithm_cfg else {}
            ),
            "policy_cfg": self.policy_cfg.model_dump() if self.policy_cfg else {},
        }
