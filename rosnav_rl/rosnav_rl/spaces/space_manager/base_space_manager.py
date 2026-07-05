from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces

from ...spaces import (
    ActionSpaceManager,
    ObservationSpaceManager,
)
from ...spaces.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)

if TYPE_CHECKING:
    from rosnav_rl.cfg.agent import AgentConfig

_logger = logging.getLogger(__name__)

EncodedObservationDict = Dict[str, np.ndarray]
ObservationDict = Dict[str, Any]
ObservationSpaceList = List[BaseObservationSpace]


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
        observation_space_list: ObservationSpaceList,
        observation_space_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._spec = spec
        self._action_space_manager = ActionSpaceManager(spec.action_space)

        obs_kwargs = observation_space_kwargs.copy() if observation_space_kwargs else {}
        obs_kwargs.update(spec.parameters.observation_kwargs())

        unknown_kwargs = set(obs_kwargs) - self._known_observation_kwargs(
            observation_space_list
        )
        if unknown_kwargs:
            _logger.warning(
                "observation_space_kwargs contains keys not accepted by any "
                "configured observation space (possible typo?): %s",
                sorted(unknown_kwargs),
            )

        self._observation_space_manager = ObservationSpaceManager(
            validate_observations=True,
        )
        self._observation_space_manager.load_configuration(
            config={
                s.__name__: obs_kwargs for s in observation_space_list
            },
        )

    @staticmethod
    def _known_observation_kwargs(space_classes: List[type]) -> set:
        """Union of keyword-argument names accepted by the given space classes."""
        known = {"normalize", "normalizer"}
        for cls in space_classes:
            for name, param in inspect.signature(cls.__init__).parameters.items():
                if name == "self" or param.kind in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                known.add(name)
        return known

    # -- encode / decode ---------------------------------------------------

    def encode_observation(self, obs_dict: ObservationDict) -> EncodedObservationDict:
        return self._observation_space_manager.encode_observation(obs_dict)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        return self._action_space_manager.decode_action(action)

    def reset_spaces(self) -> None:
        """Reset internal state of all observation spaces for a new episode."""
        self._observation_space_manager.reset_spaces()

    # -- properties --------------------------------------------------------

    @property
    def action_space_manager(self) -> ActionSpaceManager:
        return self._action_space_manager

    @property
    def observation_space_manager(self) -> ObservationSpaceManager:
        return self._observation_space_manager

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space_manager.observation_space

    @property
    def observation_space_list(self) -> List[BaseObservationSpace]:
        return self._observation_space_manager.space_list

    @property
    def action_space(self) -> Union[spaces.Dict, spaces.Box]:
        return self._action_space_manager.action_space

    @property
    def config(self):
        return {
            "observation": self._observation_space_manager.config,
            "action": self._action_space_manager.config,
        }
