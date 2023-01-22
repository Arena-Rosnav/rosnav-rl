import os
from typing import Tuple

import gym
import rospkg
import rospy
import torch as th
import yaml
from stable_baselines3.common.policies import BaseFeaturesExtractor
from torch import nn

from ..utils.utils import get_observation_space_from_file


class EXTRACTOR_1(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.

    Note:
        self._rs: Robot state size - placeholder for robot related inputs to the NN
        self._l: Number of laser beams - placeholder for the laser beam data
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 128,
    ):

        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_1, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_2(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841
    (DRLself._lOCAL_PLANNER)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 128,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_2, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_3(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(EXTRACTOR_3, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc_2(self.fc_1(self.cnn(laser_scan)))
        # return self.fc_2(features)
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_4(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_4, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_5(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_5, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_6(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super(EXTRACTOR_6, self).__init__(observation_space, features_dim + self._rs)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, self._l)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, : -self._rs], 1)
        robot_state = observations[:, -self._rs :]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class UNIFIED_SPACE_EXTRACTOR(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        robot_model: str = None,
        features_dim: int = 32,
    ):
        self._l, self._rs = get_observation_space_from_file(robot_model)
        super().__init__(observation_space, features_dim)

        self.model = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        # obs = th.unsqueeze(observations, 0)

        return self.model(observations)
