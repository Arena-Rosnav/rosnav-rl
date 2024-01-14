import gymnasium as gym
import torch as th
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from torch import nn

from .base_extractor import RosnavBaseExtractor


class EXTRACTOR_1(RosnavBaseExtractor):
    """
    Feature extractor class that extracts features from observations for the EXTRACTOR_1 model.

    Args:
        observation_space (gym.spaces.Box): The observation space.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int): The dimension of the extracted features. Default is 128.
        stacked_obs (bool): Whether the observations are stacked. Default is False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

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
            features_dim=features_dim,
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._features_dim,
            ),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (th.Tensor): The input observations.

        Returns:
            th.Tensor: The extracted features by the network.
        """
        _robot_state_size = self._goal_size + self._last_action_size
        if not self._stacked_obs:
            # observations in shape [batch_size, obs_size]
            laser_scan = th.unsqueeze(observations[:, :-_robot_state_size], 1)
            robot_state = observations[:, -_robot_state_size:]

            cnn_features = self.cnn(laser_scan)
            extracted_features = th.cat((cnn_features, robot_state), 1)
            return self.fc(extracted_features)
        else:
            # observations in shape [batch_size, num_stacks, obs_size]
            laser_scan = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(1, 2)

            cnn_features = self.cnn(laser_scan)
            extracted_features = th.cat((cnn_features, robot_state), 1)
            return self.fc(extracted_features)


class EXTRACTOR_2(EXTRACTOR_1):
    """
    Feature extractor class that extends EXTRACTOR_1.

    Args:
        observation_space (gym.spaces.Box): The observation space.
        observation_space_manager (ObservationSpaceManager): The observation space manager.
        features_dim (int, optional): The dimension of the extracted features. Defaults to 128.
        stacked_obs (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._laser_size,
            ),
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
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
        stacked_obs (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                256,
            ),
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, self._features_dim),
            nn.ReLU(),
        )


class EXTRACTOR_6(EXTRACTOR_1):
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
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
        stacked_obs (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
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
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
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
        stacked_obs (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        cnn (nn.Sequential): The convolutional neural network.
        fc (nn.Sequential): The fully connected layers.

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
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
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
        stacked_obs (bool, optional): Whether to use stacked observations. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        cnn (nn.Sequential): The convolutional neural network.
        fc (nn.Sequential): The fully connected layers.

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
            desired_shape = (1, self._num_stacks, self._laser_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._features_dim,
            ),
            nn.ReLU(),
        )
