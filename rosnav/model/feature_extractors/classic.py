import os
from typing import Tuple

import gymnasium as gym
import rospkg
import rospy
import torch as th
import yaml
from stable_baselines3.common.policies import BaseFeaturesExtractor
from torch import nn

from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX

from abc import ABC, abstractmethod


class RosnavBaseExtractor(BaseFeaturesExtractor, ABC):
    REQUIRED_OBSERVATIONS = [
        SPACE_INDEX.LASER,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int,
        stacked_obs: bool = False,
    ):
        # assert observation_space == observation_space_manager.observation_space
        self._observation_space_manager = observation_space_manager
        self._stacked_obs = stacked_obs
        self._num_stacks = observation_space.shape[0] if self._stacked_obs else 1

        super(RosnavBaseExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )

        self._setup_network()

    @abstractmethod
    def _setup_network(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class EXTRACTOR_1(RosnavBaseExtractor):
    REQUIRED_OBSERVATIONS = [
        SPACE_INDEX.LASER,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        self._laser_size, self._goal_size, self._last_action_size = (
            observation_space_manager[SPACE_INDEX.LASER].shape[0],
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[0],
        )

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1

        super(EXTRACTOR_1, self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim
            + (self._goal_size + self._last_action_size) * self._num_stacks,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, self._features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        1. Extract laser
        2. Extract robot state
        3. Forward laser data through CNN
        4. Return concatenation of extracted laser feats and robot states

        :return: (th.Tensor) features,
            extracted features by the network
        """
        _robot_state_size = self._goal_size + self._last_action_size
        if not self._stacked_obs:
            # observations in shape [batch_size, obs_size]
            laser_scan = th.unsqueeze(observations[:, :-_robot_state_size], 1)
            robot_state = observations[:, -_robot_state_size:]

            extracted_features = self.fc(self.cnn(laser_scan))
            return th.cat((extracted_features, robot_state), 1)
        else:
            # observations in shape [batch_size, num_stacks, obs_size]
            laser_scan = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].squeeze(0)

            extracted_features = self.fc(self.cnn(laser_scan))
            return th.cat((extracted_features, robot_state.flatten().unsqueeze(0)), 1)


class EXTRACTOR_2(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_3(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_4(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_5(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_6(EXTRACTOR_1):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_7(EXTRACTOR_1):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_8(EXTRACTOR_1):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 128,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 12, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 8, 4),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )
