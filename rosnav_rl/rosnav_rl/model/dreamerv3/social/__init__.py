"""Social-Dreamer augmentation subpackage (C1 Social-RSSM/GAT + C2 DALI).

See ``dims.py`` for the dimensional contract that keeps all augmented-feature consumers in sync.
"""

from .dims import (
    augmented_feat_size,
    base_feat_size,
    dynamics_context_size,
    social_context_size,
)

__all__ = [
    "augmented_feat_size",
    "base_feat_size",
    "dynamics_context_size",
    "social_context_size",
]
