from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from rosnav_rl.cfg.action_spaces import BaseActionSpace

_HolonomicAction = tuple[float, float, float] | np.ndarray | list[float]


class DecodedAction(NamedTuple):
    """One decoded sub-action: the action-space it belongs to and its command vector."""

    space: "BaseActionSpace"
    action: np.ndarray
