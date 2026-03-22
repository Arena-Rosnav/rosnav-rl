from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from rosnav_rl.cfg.action_spaces import BaseActionSpace


class ActionSpaceManager:
    """Manages the action space for an RL agent.

    Takes a typed :class:`~rosnav_rl.cfg.action_spaces.BaseActionSpace`
    specification and delegates gym-space creation and action decoding to it::

        from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace

        manager = ActionSpaceManager(DifferentialDriveActionSpace(
            linear_range=(-0.5, 1.0),
            angular_range=(-1.0, 1.0),
        ))
        gym_space = manager.action_space
        cmd = manager.decode_action(model_output)
    """

    def __init__(self, spec: BaseActionSpace) -> None:
        self._spec = spec
        self._space = spec.get_gym_space()

    @property
    def spec(self) -> BaseActionSpace:
        """The typed action space specification."""
        return self._spec

    @property
    def action_space(self) -> spaces.Space:
        return self._space

    @property
    def shape(self):
        return self._space.shape

    def decode_action(self, action) -> np.ndarray:
        return self._spec.decode(action)

    @property
    def config(self) -> dict:
        return self._spec.model_dump()
