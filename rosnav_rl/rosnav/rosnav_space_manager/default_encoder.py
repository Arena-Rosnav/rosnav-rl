from typing import List

import numpy as np
from gymnasium import spaces

from ..utils.action_space.action_space_manager import ActionSpaceManager
from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import SPACE_INDEX
from .base_space_encoder import BaseSpaceEncoder
from .encoder_factory import BaseSpaceEncoderFactory

"""
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("DefaultEncoder")
class DefaultEncoder(BaseSpaceEncoder):
    DEFAULT_OBS_LIST = [
        SPACE_INDEX.LASER,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]

    def __init__(
        self,
        action_space_kwargs: dict,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
        *args,
        **kwargs
    ):
        super().__init__(**action_space_kwargs, **observation_kwargs, **kwargs)
        self._observation_list = observation_list
        self.setup_action_space(action_space_kwargs)
        self.setup_observation_space(observation_list, observation_kwargs)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space_manager.observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space_manager.action_space

    @property
    def action_space_manager(self):
        return self._action_space_manager

    @property
    def observation_space_manager(self):
        return self._observation_space_manager

    def setup_action_space(self, action_space_kwargs: dict):
        self._action_space_manager = ActionSpaceManager(**action_space_kwargs)

    def setup_observation_space(
        self,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
    ):
        if not observation_list:
            observation_list = self.DEFAULT_OBS_LIST

        self._observation_space_manager = ObservationSpaceManager(
            observation_list,
            space_kwargs=observation_kwargs,
            frame_stacking=self._stacked,
        )

    def decode_action(self, action) -> np.ndarray:
        return self._action_space_manager.decode_action(action)

    def encode_observation(self, observation, structure) -> np.ndarray:
        return self._observation_space_manager.encode_observation(observation)
