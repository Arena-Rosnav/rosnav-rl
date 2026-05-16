from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import (
    StackedObservations,
    TObs,
)


class UpdatedStackedObservations(StackedObservations):
    def __init__(
        self,
        num_envs: int,
        n_stack: int,
        observation_space: Union[spaces.Box, spaces.Dict],
        channels_order: Optional[Union[str, Mapping[str, Optional[str]]]] = None,
    ) -> None:
        self.n_stack = n_stack
        self.observation_space = observation_space
        self._enforce_time_dimension = False
        if isinstance(observation_space, spaces.Dict):
            if not isinstance(channels_order, Mapping):
                channels_order = {
                    key: channels_order for key in observation_space.spaces.keys()
                }
            self.sub_stacked_observations = {
                key: UpdatedStackedObservations(num_envs, n_stack, subspace, channels_order[key])  # type: ignore[arg-type]
                for key, subspace in observation_space.spaces.items()
            }
            self.stacked_observation_space = spaces.Dict(
                {
                    key: substack_obs.stacked_observation_space
                    for key, substack_obs in self.sub_stacked_observations.items()
                }
            )  # type: Union[spaces.Dict, spaces.Box] # make mypy happy
        elif isinstance(observation_space, spaces.Box):
            if isinstance(channels_order, Mapping):
                raise TypeError(
                    "When the observation space is Box, channels_order can't be a dict."
                )

            if len(observation_space.shape) == 1:
                observation_space = spaces.Box(
                    low=np.array([observation_space.low]),
                    high=np.array([observation_space.high]),
                    shape=(1, observation_space.shape[0]),
                    dtype=observation_space.dtype,
                )  # type: ignore[arg-type]
                self._enforce_time_dimension = True

            (
                self.channels_first,
                self.stack_dimension,
                self.stacked_shape,
                self.repeat_axis,
            ) = self.compute_stacking(n_stack, observation_space, channels_order)
            low = np.repeat(observation_space.low, n_stack, axis=self.repeat_axis)
            high = np.repeat(observation_space.high, n_stack, axis=self.repeat_axis)
            self.stacked_observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=observation_space.dtype,  # type: ignore[arg-type]
            )
            self.stacked_obs = np.zeros(
                (num_envs, *self.stacked_shape), dtype=observation_space.dtype
            )
        else:
            raise TypeError(
                f"StackedObservations only supports Box and Dict as observation spaces. {observation_space} was provided."
            )

    def update(
        self,
        observations: TObs,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[TObs, List[Dict[str, Any]]]:
        # Also expand time dimension on terminal_observation in infos,
        # otherwise SB3's np.concatenate(previous_stack_2D, terminal_obs_1D)
        # fails with a dimension mismatch.
        if self._enforce_time_dimension:
            for info in infos:
                if "terminal_observation" in info:
                    info["terminal_observation"] = np.expand_dims(
                        info["terminal_observation"], 0
                    )
        return super().update(self.add_time_dimension(observations), dones, infos)

    def reset(self, observation: TObs) -> TObs:
        return super().reset(self.add_time_dimension(observation))

    def add_time_dimension(self, observation: TObs) -> TObs:
        if self._enforce_time_dimension and isinstance(
            self.stacked_observation_space, spaces.Box
        ):
            return np.expand_dims(observation, 1)
        return observation

    # @staticmethod
    # def compute_stacking(
    #     n_stack: int,
    #     observation_space: spaces.Box,
    #     channels_order: Optional[str] = None,
    # ) -> Tuple[bool, int, Tuple[int, ...], int]:
    #     # Handle 1D observation spaces by reshaping to 2D
    #     original_shape = observation_space.shape
    #     if len(original_shape) == 1:
    #         adjusted_shape = (1,) + original_shape  # Reshape to (1, D)
    #         force_channels_first = True
    #     else:
    #         adjusted_shape = original_shape
    #         force_channels_first = False

    #     if channels_order is None:
    #         if is_image_space(observation_space):
    #             channels_first = is_image_space_channels_first(observation_space)
    #         else:
    #             # Default to first axis for reshaped 1D observations
    #             channels_first = force_channels_first
    #     else:
    #         channels_first = channels_order == "first"

    #     stack_dimension = 1 if channels_first else -1
    #     repeat_axis = 0 if channels_first else -1

    #     # Use adjusted shape for calculations
    #     stacked_shape = list(adjusted_shape)
    #     stacked_shape[repeat_axis] *= n_stack  # Multiply the repeat axis by stack size

    #     return channels_first, stack_dimension, tuple(stacked_shape), repeat_axis


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment. Designed for image observations.

    :param venv: Vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
        Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    """

    def __init__(
        self,
        venv: VecEnv,
        n_stack: int,
        channels_order: Optional[Union[str, Mapping[str, str]]] = None,
    ) -> None:
        assert isinstance(
            venv.observation_space, (spaces.Box, spaces.Dict)
        ), "VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"

        self.stacked_obs = UpdatedStackedObservations(
            venv.num_envs, n_stack, venv.observation_space, channels_order
        )
        observation_space = self.stacked_obs.stacked_observation_space
        super().__init__(venv, observation_space=observation_space)

    def step_wait(
        self,
    ) -> Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
        List[Dict[str, Any]],
    ]:
        observations, rewards, dones, infos = self.venv.step_wait()
        observations, infos = self.stacked_obs.update(observations, dones, infos)  # type: ignore[arg-type]
        return observations, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        """
        observation = self.venv.reset()
        observation = self.stacked_obs.reset(observation)  # type: ignore[arg-type]
        return observation
