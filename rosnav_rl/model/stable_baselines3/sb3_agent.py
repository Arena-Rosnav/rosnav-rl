import os
from typing import List, Optional, Union

import gym
from rosnav_rl.config import PPO_Cfg, PPO_Policy_Cfg
from rosnav_rl.config.stable_baselines3.lr_schedule import LearningRateSchedulerCfg
from rosnav_rl.model.stable_baselines3 import import_models
from rosnav_rl.spaces import BaseObservationSpace
from rosnav_rl.utils.model.learning_rate_schedules import load_lr_schedule
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from ..model import RL_Model
from .policy.base_policy import PolicyType, StableBaselinesPolicy


class StableBaselinesAgent(RL_Model):
    model: Union[PPO, RecurrentPPO] = None

    def __init__(
        self,
        policy_cfg: PPO_Policy_Cfg,
    ):
        self.policy_cfg = policy_cfg
        self._agent_factory = import_models()

        self._policy_description: StableBaselinesPolicy = (
            self._agent_factory.instantiate(
                self.policy_cfg.architecture_name,
            )
        )

    @property
    def observation_space_list(self) -> List[BaseObservationSpace]:
        return self._policy_description.observation_spaces

    @property
    def observation_space_kwargs(self):
        return self._policy_description.observation_space_kwargs

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
        alg_kwargs: dict = self._setup_algorithm_config(
            algorithm_cfg, env, no_gpu, tensorboard_log_path
        )
        self._initialize_model(alg_kwargs)

    def _setup_learning_rate_scheduler(self, algorithm_cfg: PPO_Cfg):
        if isinstance(algorithm_cfg.learning_rate, LearningRateSchedulerCfg):
            algorithm_cfg.learning_rate = load_lr_schedule(
                algorithm_cfg.learning_rate.type,
                algorithm_cfg.learning_rate.kwargs,
            )

    def _setup_algorithm_config(
        self,
        algorithm_cfg: PPO_Cfg,
        env: Union[VecEnv, gym.Env],
        no_gpu: bool,
        tensorboard_log_path: str,
    ) -> dict:
        if algorithm_cfg.tensorboard_log is None:
            algorithm_cfg.tensorboard_log = tensorboard_log_path
        algorithm_cfg.device = "cpu" if no_gpu else "auto"
        return dict(
            env=env,
            tensorboard_log_path=tensorboard_log_path,
            policy=self._policy_description.type,
            policy_kwargs=self._policy_description.get_kwargs(),
            device="cpu" if no_gpu else "auto",
            **algorithm_cfg.model_dump(),
        )

    def _initialize_model(self, alg_kwargs: dict):
        is_lstm = "LSTM" in self._policy_description.type.name
        model_class = RecurrentPPO if is_lstm else PPO
        self.model = model_class(**alg_kwargs)

    def load(self, path: str, *args, **kwargs):
        self.model = self._load_model(
            path,
            self._policy_description,
            self.policy_cfg.checkpoint,
        )

    def _load_model(
        self,
        agent_path: str,
        agent_description: StableBaselinesPolicy,
        checkpoint_name: str = "best_model",
    ) -> Union[PPO, RecurrentPPO]:
        # TODO: HANDLE NEW CONFIGURATION
        model_path = os.path.join(agent_path, f"{checkpoint_name}.zip")
        custom_objects = {"policy_kwargs": agent_description.get_kwargs()}

        if agent_description.type == PolicyType.MULTI_INPUT:
            return PPO.load(model_path, custom_objects=custom_objects)
        elif agent_description.type == PolicyType.MULTI_INPUT_LSTM:
            return RecurrentPPO.load(model_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unsupported policy type: {agent_description.type}")

    def get_action(self, observation, *args, **kwargs):
        raise NotImplementedError()
        return self.model.predict(observation, deterministic=True)

    def train(self, *args, **kwargs):
        self.model.learn(*args, **kwargs)

    @property
    def config(self):
        return {
            "algorithm_cfg": self.algorithm_cfg.model_dump(),
            "policy_cfg": self.policy_cfg.model_dump(),
        }
