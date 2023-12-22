from abc import ABC, abstractmethod

import torch as th
from gymnasium import spaces
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from stable_baselines3.common.policies import BaseFeaturesExtractor


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
        SPACE_INDEX.LASER,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]

    def __init__(
        self,
        observation_space: spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int,
        stacked_obs: bool = False,
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
        self._stacked_obs = stacked_obs
        self._num_stacks = observation_space.shape[0] if self._stacked_obs else 1

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
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (th.Tensor): The input observations.

        Returns:
            th.Tensor: The extracted features.
        """
        raise NotImplementedError
