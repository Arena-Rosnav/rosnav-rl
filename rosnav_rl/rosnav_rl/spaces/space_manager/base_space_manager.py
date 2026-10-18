from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from rosnav_rl.utils.type_aliases import ObservationDict

from ...spaces import (
    ActionSpaceManager,
    ObservationSpaceManager,
)
from ...spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)

if TYPE_CHECKING:
    from rosnav_rl.cfg.agent import AgentConfig


class BaseSpaceManager:
    """Manages action and observation spaces for an RL agent.

    Constructed from an :class:`~rosnav_rl.cfg.AgentConfig` — the typed action
    space drives the ``ActionSpaceManager`` and the ``ObservationConfig`` is
    merged with per-model ``observation_space_kwargs`` to configure each
    observation encoder.
    """

    _action_space_manager: ActionSpaceManager
    _observation_space_manager: ObservationSpaceManager

    def __init__(
        self,
        spec: AgentConfig,
        observation_space_list: list[type[BaseObservationSpace]],
        observation_space_kwargs: dict[str, Any] | None = None,
    ):
        self._spec = spec
        self._action_space_manager = ActionSpaceManager(spec.action_space)
        self._normalized = spec.normalized_actions
        command_space = self._action_space_manager.action_space
        if self._normalized and not isinstance(command_space, spaces.Box):
            raise ValueError("normalized_actions needs a continuous (Box) action space")

        obs_kwargs = observation_space_kwargs.copy() if observation_space_kwargs else {}
        obs_kwargs.update(spec.parameters.observation_kwargs())

        self._observation_space_manager = ObservationSpaceManager(
            validate_observations=True,
        )
        self._observation_space_manager.load_configuration(
            config={s.__name__: obs_kwargs for s in observation_space_list},
        )

    # -- encode / decode ---------------------------------------------------

    def encode_observation(
        self, obs_dict: ObservationDict
    ) -> np.ndarray | bool | dict[str, np.ndarray] | dict[str, np.ndarray | bool | dict[str, np.ndarray]]:
        return self._observation_space_manager.encode_observation(obs_dict)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        if self._normalized:
            space = self._action_space_manager.action_space
            action = space.low + (np.clip(action, -1.0, 1.0) + 1.0) / 2.0 * (space.high - space.low)
        return self._action_space_manager.decode_action(action)

    def reset_spaces(self) -> None:
        """Reset the internal state of all observation spaces for a new episode."""
        self._observation_space_manager.reset_spaces()

    # -- properties --------------------------------------------------------

    @property
    def action_space_manager(self) -> ActionSpaceManager:
        return self._action_space_manager

    @property
    def observation_space_manager(self) -> ObservationSpaceManager:
        return self._observation_space_manager

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space_manager.observation_space

    @property
    def observation_space_list(self) -> list[BaseObservationSpace]:
        return self._observation_space_manager.space_list

    @property
    def action_space(self) -> spaces.Space:
        space = self._action_space_manager.action_space
        if self._normalized:
            return spaces.Box(low=-1.0, high=1.0, shape=space.shape, dtype=space.dtype)
        return space

    @property
    def config(self) -> dict[str, dict[str, Any]]:
        return {
            "observation": self._observation_space_manager.config,
            "action": self._action_space_manager.config,
        }
