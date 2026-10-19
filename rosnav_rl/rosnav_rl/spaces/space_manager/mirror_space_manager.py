from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.type_aliases import ObservationDict

from ..observation_space.spaces.dynamics import LastActionSpace
from ..observation_space.spaces.environment import (
    CrowdTimeToContactSpace,
    LaserCartesianMapSpace,
    PedestrianLocationSpace,
    PedestrianSocialStateSpace,
    PedestrianTypeSpace,
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)
from ..observation_space.spaces.navigation import DistAngleToGoalSpace, DistAngleToSubgoalSpace
from .base_space_manager import BaseSpaceManager

if TYPE_CHECKING:
    from rosnav_rl.cfg.action_spaces import BaseActionSpace
    from rosnav_rl.cfg.agent import AgentConfig

    from ..observation_space.spaces.base_observation_space import BaseObservationSpace

MirrorMode = Literal["canonical", "random"]

_SYMMETRIC_NORMALIZERS = frozenset({"max_abs", "identity", "none"})

_DRIVE_TYPES = frozenset({"differential_drive", "omnidirectional"})


def _flip_rows(observation: np.ndarray) -> np.ndarray:
    if observation.ndim < 2:
        raise ValueError("row reflection needs an unflattened feature map")
    return np.ascontiguousarray(np.flip(observation, axis=-2))


def _negated_rows(observation: np.ndarray) -> np.ndarray:
    return -_flip_rows(observation)


def mirrored_beams(num_beams: int, angle_min: float, increment: float) -> np.ndarray:
    """Index of the beam at angle -a for each beam at angle a, for a scan symmetric about the forward axis."""
    angles = angle_min + increment * np.arange(num_beams)
    partner = np.mod(np.rint((-angles - angle_min) / increment).astype(int), num_beams)
    error = np.angle(np.exp(1j * (angles[partner] + angles)))
    if np.max(np.abs(error)) > 1e-3 * increment:
        raise ValueError(
            f"a scan of {num_beams} beams from {angle_min} rad every {increment} rad has no beam at the mirror angle of every beam"
        )
    return partner


def _negate(indices: tuple[int, ...]) -> Callable[[np.ndarray], np.ndarray]:
    def reflect(observation: np.ndarray) -> np.ndarray:
        reflected = np.array(observation, dtype=observation.dtype, copy=True)
        reflected[..., list(indices)] *= -1
        return reflected

    return reflect


def _identity(observation: np.ndarray) -> np.ndarray:
    return observation


class MirrorSpaceManager(BaseSpaceManager):
    """Reflects observations and the commands decoded from them across the robot's forward axis."""

    def __init__(
        self,
        spec: AgentConfig,
        observation_space_list: list[type[BaseObservationSpace]],
        observation_space_kwargs: dict[str, Any] | None = None,
        mode: MirrorMode = "canonical",
        pedestrian_range: float = 12.0,
        seed: int | None = None,
    ):
        super().__init__(spec, observation_space_list, observation_space_kwargs)
        if any(sub.type == "arm" for sub in self._action_space_manager.spec.sub_spaces):
            raise ValueError("mirroring has no reflection rule for arm actions")
        if spec.parameters.normalize and spec.parameters.normalizer not in _SYMMETRIC_NORMALIZERS:
            raise ValueError(f"mirroring needs a sign-preserving normalizer, got '{spec.parameters.normalizer}'")
        self._mode = mode
        self._pedestrian_range = pedestrian_range
        self._rng = np.random.default_rng(seed)
        self._laser_order: np.ndarray | None = None
        self._rules = {name: self._rule(name) for name in self._observation_space_manager.spaces}
        self._episode_reflected = False
        self._reflected = False

    def _rule(self, name: str) -> Callable[[np.ndarray], np.ndarray]:
        space = self._observation_space_manager.spaces[name].space
        rows = (
            PedestrianVelXSpace.name,
            PedestrianTypeSpace.name,
            PedestrianSocialStateSpace.name,
            PedestrianLocationSpace.name,
            LaserCartesianMapSpace.name,
        )
        if name in rows:
            return _flip_rows
        if name == PedestrianVelYSpace.name:
            return _negated_rows
        if name == StackedLaserMapSpace.name:
            parameters = self._spec.parameters
            if parameters.laser_angle_increment is None:
                raise ValueError("mirroring the laser map needs laser_angle_increment")
            self._laser_order = mirrored_beams(
                parameters.laser_num_beams, parameters.laser_angle_min, parameters.laser_angle_increment
            )
            return _identity
        if name in (DistAngleToSubgoalSpace.name, DistAngleToGoalSpace.name):
            return self._checked_negate(name, space, (1,))
        if name == LastActionSpace.name:
            return self._checked_negate(name, space, tuple(range(1, space.shape[-1])))
        if name == CrowdTimeToContactSpace.name:
            return _identity
        raise ValueError(f"no reflection rule for observation space '{name}'")

    @staticmethod
    def _checked_negate(name: str, space: spaces.Space, indices: tuple[int, ...]) -> Callable[[np.ndarray], np.ndarray]:
        if not isinstance(space, spaces.Box):
            raise ValueError(f"'{name}' must be a Box to be reflected")
        low, high = space.low[list(indices)], space.high[list(indices)]
        if not np.allclose(low, -high):
            raise ValueError(f"'{name}' bounds are not symmetric on the reflected components {indices}")
        return _negate(indices)

    @property
    def reflected(self) -> bool:
        return self._reflected

    def _nearest_pedestrian_on_right(self, obs_dict: ObservationDict) -> bool:
        positions = np.asarray(obs_dict["pedestrian_relative_locations"], dtype=np.float64).reshape(-1, 2)
        if len(positions) == 0:
            return False
        distances = np.hypot(positions[:, 0], positions[:, 1])
        nearest = int(np.argmin(distances))
        return bool(distances[nearest] <= self._pedestrian_range and positions[nearest, 1] < 0.0)

    def encode_observation(self, obs_dict: ObservationDict) -> dict[str, np.ndarray]:
        encoded = super().encode_observation(obs_dict)
        if not isinstance(encoded, dict):
            raise ValueError("mirroring needs a dict observation")
        self._reflected = (
            self._episode_reflected if self._mode == "random" else self._nearest_pedestrian_on_right(obs_dict)
        )
        if not self._reflected:
            return encoded
        return {
            name: self._observation_space_manager.spaces[name].render(self._laser_order)
            if name == StackedLaserMapSpace.name
            else self._rules[name](value)
            for name, value in encoded.items()
        }

    def _reflect(self, command: np.ndarray) -> np.ndarray:
        if not self._reflected:
            return command
        reflected = np.array(command, dtype=np.float32, copy=True)
        reflected[1:3] *= -1
        return reflected

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        return self._reflect(super().decode_action(action))

    def decode_split_action(self, action: np.ndarray) -> list[tuple[BaseActionSpace, np.ndarray]]:
        return [
            (sub, self._reflect(command) if sub.type in _DRIVE_TYPES else command)
            for sub, command in super().decode_split_action(action)
        ]

    def reset_spaces(self) -> None:
        super().reset_spaces()
        self._episode_reflected = bool(self._rng.random() < 0.5)
        self._reflected = self._episode_reflected if self._mode == "random" else False
