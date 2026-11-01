"""Scalar crowd observation spaces."""

from typing import Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import CrowdTimeToContact
from rosnav_rl.spaces.observation_space.observation_space_factory import SpaceFactory
from rosnav_rl.spaces.observation_space.space_categories import SpaceCategory
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import BaseObservationSpace

#add
from rosnav_rl.observations.utils.types import CrowdSpectralScalars

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


#class for 15 spectral crowd-graph scalars
@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class CrowdSpectralSpace(BaseObservationSpace):
    """15 spectral crowd-graph scalars."""

    name = "CrowdSpectralSpace"
    requires = {"crowd_spectral": CrowdSpectralScalars}

    def get_gym_space(self) -> spaces.Space:
        low = np.array(
            [0.0] * 6 + [0.0] * 6 + [0.0] + [-np.inf] * 2,
            dtype=np.float32,
        )
        high = np.array(
            [2.0] * 6 + [1.0] * 6 + [10.0] + [np.inf] * 2,
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @BaseObservationSpace.apply_normalization
    def encode_observation(
        self,
        crowd_spectral: CrowdSpectralScalars,
        *args: Any,
        **kwargs: Any,
    ) -> CrowdSpectralScalars:
        return np.clip(
            np.asarray(crowd_spectral, dtype=np.float32),
            self._space.low,
            self._space.high,
        )