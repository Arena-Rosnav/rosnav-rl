import os
from typing import Optional, Union, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    VecEnv,
    sync_envs_normalization,
    unwrap_vec_normalize,
)


class RosnavEvalCallback(EvalCallback):

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_eval_end: Optional[BaseCallback] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.last_success_rate = -np.inf
        self.callback_on_eval_end = callback_on_eval_end

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                self.last_success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * self.last_success_rate:.2f}%")
                self.logger.record("eval/success_rate", self.last_success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )

                    vec_normalize = unwrap_vec_normalize(self.training_env)
                    if vec_normalize is not None:
                        vec_normalize.save(
                            os.path.join(
                                self.best_model_save_path,
                                "vec_normalize_best_model.pkl",
                            )
                        )

                self.best_mean_reward = mean_reward

                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback_on_eval_end is not None:
                self.callback_on_eval_end.parent = self
                self.callback_on_eval_end._on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    @property
    def last_success_rate(self):
        return self._last_success_rate

    @last_success_rate.setter
    def last_success_rate(self, value):
        self._last_success_rate = value


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    :param is_last_state_getter: A callable that returns whether the last state has been reached
    """

    def __init__(
        self,
        reward_threshold: float,
        verbose: int = 0,
        is_last_state_getter: Callable[[], bool] = lambda: True,
    ):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold
        self.is_last_state_getter = is_last_state_getter

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingOnMinimumReward`` callback must be used "
            "with an ``EvalCallback``"
        )
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(
            self.parent.best_mean_reward < self.reward_threshold
            and not self.is_last_state_getter()
        )
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training


class StopTrainingOnSuccessThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic success
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    :param is_last_state_getter: A callable that returns whether the last state has been reached
    """

    def __init__(
        self,
        success_threshold: float,
        verbose: int = 0,
        is_last_state_getter: Callable[[], bool] = lambda: True,
    ):
        super().__init__(verbose=verbose)
        self.success_threshold = success_threshold
        self.last_state_getter = is_last_state_getter

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingOnMinimumReward`` callback must be used "
            "with an ``EvalCallback``"
        )
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(
            self.parent.last_success_rate < self.success_threshold
            and not self.last_state_getter()
        )
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the success rate {self.parent.last_success_rate:.2f} "
                f" is above the threshold {self.success_threshold}"
            )
        return continue_training
