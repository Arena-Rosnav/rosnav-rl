"""Scalar crowd observation spaces."""

from typing import Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import CrowdTimeToContact
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class CrowdTimeToContactSpace(BaseObservationSpace):
    """Minimum, second minimum, imminent count and minimum ahead of the crowd time to contact."""

    name = "CrowdTimeToContactSpace"
    requires = {"crowd_ttc": CrowdTimeToContact}

    def __init__(self, ttc_horizon: float = 10.0, ttc_count_cap: int = 10, *args: Any, **kwargs: Any) -> None:
        self._horizon = ttc_horizon
        self._count_cap = ttc_count_cap
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.array([self._horizon, self._horizon, self._count_cap, self._horizon], dtype=np.float32),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, crowd_ttc: CrowdTimeToContact, *args: Any, **kwargs: Any) -> CrowdTimeToContact:
        return np.clip(np.asarray(crowd_ttc, dtype=np.float32), self._space.low, self._space.high)
