from abc import ABC, abstractmethod

import torch as th
from gymnasium import spaces
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from stable_baselines3.common.policies import BaseFeaturesExtractor


class RosnavBaseExtractor(BaseFeaturesExtractor, ABC):
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
