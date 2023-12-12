from abc import ABC, abstractmethod
from typing import List
from gymnasium import spaces

import numpy as np


class BaseObservationSpace(ABC):
    @abstractmethod
    def get_gym_space(self) -> spaces.Space:
        raise NotImplementedError()
