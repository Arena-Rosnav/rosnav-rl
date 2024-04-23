from typing import List

import numpy as np
from gymnasium import spaces

from rosnav.utils.action_space.action_space_manager import ActionSpaceManager
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from .base_space_encoder import BaseSpaceEncoder

"""
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


class DefaultEncoder(BaseSpaceEncoder):
    """
    DefaultEncoder class is responsible for encoding and decoding actions and observations
    using the default action and observation space managers.
    """

    DEFAULT_OBS_LIST = [
        SPACE_INDEX.LASER,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]
