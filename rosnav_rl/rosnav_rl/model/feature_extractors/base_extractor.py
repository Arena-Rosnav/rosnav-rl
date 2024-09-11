from abc import ABC, abstractmethod
from typing import Dict, Union

import rosnav_rl.utils.observation_space as SPACE
import torch as th
from gymnasium import spaces
from rosnav_rl.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from stable_baselines3.common.policies import BaseFeaturesExtractor

TensorDict = Dict[str, th.Tensor]


class RosnavBaseExtractor(BaseFeaturesExtractor, ABC):
    """
    Base class for feature extractors used in RosNav.

    Default observations spaces:
        - Laser scan
        - Goal
        - Last action

    Note:
        RosNav uses a custom observation space manager to handle the different observation spaces.
    """

    REQUIRED_OBSERVATIONS = [
        SPACE.LaserScanSpace,
        SPACE.DistAngleToSubgoalSpace,
        SPACE.LastActionSpace,
    ]

    def __init__(
        self,
        observation_space: spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int,
        stack_size: int,
        *args,
        **kwargs
    ):
        """
        Initialize the base feature extractor.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            observation_space_manager (ObservationSpaceManager): The observation space manager.
            features_dim (int): The dimension of the extracted features.
            stacked_obs (bool, optional): Whether the observations are stacked. Defaults to False.
        """
        self._observation_space_manager = observation_space_manager
        self._stack_size = stack_size

        super(RosnavBaseExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )

        self._setup_network(**kwargs)

    @abstractmethod
    def _setup_network(self, *args, **kwargs):
        """
        Set up the network architecture for feature extraction.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations: Union[th.Tensor, SPACE.TensorDict]) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (th.Tensor): The input observations.

        Returns:
            th.Tensor: The extracted features.
        """
        raise NotImplementedError
