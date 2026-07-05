"""
Backward-compat re-export shim.

Moved to rosnav_rl.utils.observation_types (P4.4, audit 2026-07-04) so
spaces/reward code depending only on these type annotations doesn't need
to import the observations package. Import from the new location in new
code; this module re-exports for existing callers.
"""

from rosnav_rl.utils.observation_types import *  # noqa: F401,F403
from rosnav_rl.utils.observation_types import (
    ArenaPedestrianStates,
    GoalRelativePosition,
    IsFirst,
    PedestrianDistances,
)
from rosnav_rl.utils.observation_types import __all__ as __all__
