from typing import Tuple, Union

import gymnasium as gym
import rosnav_rl.spaces.observation_space as SPACE
import torch as th
from rosnav_rl.spaces.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from torch import nn

from .base_extractor import RosnavBaseExtractor, TensorDict


class EXTRACTOR_1(RosnavBaseExtractor):
    """
    Feature extractor class that extracts features from observations for the EXTRACTOR_1 model.

    Args:
        observation_space (gym.spaces.Box): The observation space.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int): The dimension of the extracted features. Default is 128.
        stack_size (bool): Whether the observations are stacked. Default is False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        self._laser_size, self._goal_size, self._last_action_size = (
            observation_space[SPACE.LaserScanSpace.name].shape[-1],
            observation_space[SPACE.DistAngleToSubgoalSpace].shape[-1],
            observation_space[SPACE.LastActionSpace].shape[-1],
        )

        self._stack_size = stack_size

        super(EXTRACTOR_1, self).__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                self._features_dim,
            ),
            nn.ReLU(),
        )

    def get_input(
        self, observations: Union[th.Tensor, TensorDict]
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if isinstance(observations, th.Tensor):
            raise NotImplementedError("Not implemented for th.Tensor.")
            # if observations.dim() == 2:
            #     laser_scan = observations[
            #         :, : -(self._goal_size + self._last_action_size)
            #     ].unsqueeze(1)
            #     goal = observations[
            #         :, self._laser_size : (self._laser_size + self._goal_size)
            #     ]
            #     last_action = observations[:, -self._last_action_size :]
            # else:
            #     laser_scan = observations[
            #         :, :, : -(self._goal_size + self._last_action_size)
            #     ]
            #     goal = observations[
            #         :, :, self._laser_size : (self._laser_size + self._goal_size)
            #     ].flatten(1, 2)
            #     last_action = observations[:, :, -self._last_action_size :].flatten(
            #         1, 2
            #     )
        elif isinstance(observations, dict):
            laser_scan = observations[SPACE.LaserScanSpace.name]
            goal = observations[SPACE.DistAngleToSubgoalSpace.name]
            last_action = observations[SPACE.LastActionSpace.name]
        else:
            raise ValueError("Invalid input type.")

        return laser_scan, goal, last_action

    def forward(self, observations: Union[th.Tensor, TensorDict]) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (th.Tensor): The input observations.

        Returns:
            th.Tensor: The extracted features by the network.
        """
        laser_scan, goal, last_action = self.get_input(observations)

        cnn_features = self.cnn(laser_scan)
        extracted_features = th.cat(
            (cnn_features, goal.flatten(1, 2), last_action.flatten(1, 2)), 1
        )
        return self.fc(extracted_features)


class EXTRACTOR_2(EXTRACTOR_1):
    """
    Feature extractor class that extends EXTRACTOR_1.

    Args:
        observation_space (gym.spaces.Box): The observation space.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimension of the extracted features. Defaults to 128.
        stack_size (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                self._laser_size,
            ),
            nn.ReLU(),
        )


class EXTRACTOR_3(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_4(EXTRACTOR_1):
    """
    Feature extractor class that extends EXTRACTOR_1.

    Args:
        observation_space (gym.spaces.Box): The observation space.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimension of the extracted features. Defaults to 128.
        stack_size (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_5(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_5_extended(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                512,
            ),
            nn.ReLU(),
            nn.Linear(512, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_6(EXTRACTOR_1):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_7(EXTRACTOR_1):
    """
    Feature extractor class that implements a specific network architecture (EXTRACTOR_7).

    Args:
        observation_space (gym.spaces.Box): The observation space of the environment.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimensionality of the extracted features. Defaults to 128.
        stack_size (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_8(EXTRACTOR_1):
    """
    Feature extractor class that implements a specific network architecture (EXTRACTOR_8).

    Args:
        observation_space (gym.spaces.Box): The observation space of the environment.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimensionality of the extracted features. Defaults to 128.
        stack_size (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        cnn (nn.Sequential): The convolutional neural network.
        fc (nn.Sequential): The fully connected layers.

    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 8, 4),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_9(EXTRACTOR_1):
    """
    Feature extractor class that implements a specific network architecture (EXTRACTOR_9).

    Args:
        observation_space (gym.spaces.Box): The observation space of the environment.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimensionality of the extracted features. Defaults to 128.
        stack_size (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        cnn (nn.Sequential): The convolutional neural network.
        fc (nn.Sequential): The fully connected layers.

    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        stack_size: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            stack_size=stack_size,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._stack_size, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 8, 4),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._stack_size, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._stack_size,
                self._features_dim,
            ),
            nn.ReLU(),
        )
