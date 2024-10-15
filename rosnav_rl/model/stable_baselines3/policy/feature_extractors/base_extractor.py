from abc import ABC, abstractmethod
from typing import Dict, List, Union

import rosnav_rl.spaces.observation_space as SPACE
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseFeaturesExtractor

TensorDict = Dict[str, th.Tensor]


class RosnavBaseExtractor(BaseFeaturesExtractor, ABC):
    REQUIRED_OBSERVATIONS: List[SPACE.BaseObservationSpace] = []

    def __init__(
        self,
        observation_space: spaces.Dict,
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
