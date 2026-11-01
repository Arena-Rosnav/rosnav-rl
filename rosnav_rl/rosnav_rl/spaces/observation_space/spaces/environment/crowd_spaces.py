"""Scalar crowd observation spaces."""

from typing import Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.reachability_tube import ReachabilityTube
from rosnav_rl.observations.utils.types import CrowdReachabilityTube, CrowdTimeToContact
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


@SpaceFactory.register(auto_name=True, category=SpaceCategory.ENVIRONMENT)
class CrowdReachabilityTubeSpace(BaseObservationSpace):
    """Minimum, second minimum, count inside the tube and minimum ahead of the crowd HJ reachability value."""

    name = "CrowdReachabilityTubeSpace"
    requires = {"crowd_tube": CrowdReachabilityTube}

    def __init__(
        self,
        value_cap: float = 1.5,
        count_cap: int = 10,
        robot_id: str = "jackal",
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._value_cap = value_cap
        self._count_cap = count_cap
        tube = ReachabilityTube.from_package(robot_id)
        if not tube.matches_robot(max_linear_vel, max_angular_vel):
            raise ValueError(
                f"reachability table for robot_id={robot_id!r} was generated for rated speed "
                f"{tube.robot_rated_speed_mps} m/s and turn rate {tube.robot_max_yaw_rate_radps} rad/s, "
                f"but this agent's params are max_linear_vel={max_linear_vel}, max_angular_vel={max_angular_vel}"
            )
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        return spaces.Box(
            low=np.array([-self._value_cap, -self._value_cap, 0.0, -self._value_cap], dtype=np.float32),
            high=np.array([self._value_cap, self._value_cap, self._count_cap, self._value_cap], dtype=np.float32),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, crowd_tube: CrowdReachabilityTube, *args: Any, **kwargs: Any) -> CrowdReachabilityTube:
        return np.clip(np.asarray(crowd_tube, dtype=np.float32), self._space.low, self._space.high)
