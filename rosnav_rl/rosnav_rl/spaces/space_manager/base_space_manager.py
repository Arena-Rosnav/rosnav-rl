from __future__ import annotations

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

        self._observation_space_manager = ObservationSpaceManager(
            validate_observations=True,
        )
        self._observation_space_manager.load_configuration(
            config={
                s.__name__: obs_kwargs for s in observation_space_list
            },
        )

    # -- encode / decode ---------------------------------------------------

    def encode_observation(self, obs_dict: ObservationDict) -> EncodedObservationDict:
        return self._observation_space_manager.encode_observation(obs_dict)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        return self._action_space_manager.decode_action(action)

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
