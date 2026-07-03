"""Dimensional contract for the Social-Dreamer augmented state.

This module is the single source of truth for the width of the feature vector that feeds the
reward head, the value network, and the actor. Every head constructed in ``WorldModel.__init__``
and ``ImagBehavior.__init__`` must read its input width from :func:`augmented_feat_size` so that
C1 (GAT context c_t) and C2 (DALI context d_t) cannot drift out of dimensional sync.

Augmented state:  s_hat_t = (h_t, z_t, c_t, d_t)
    base_feat = |z_t| + |h_t|              (the existing ``RSSM.get_feat`` output)
    aug_feat  = base_feat + |c_t| + |d_t|  (only when the respective component is enabled)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cfg import DreamerV3Cfg


def base_feat_size(config: "DreamerV3Cfg") -> int:
    """Return the width of the baseline DreamerV3 feature ``concat(stoch, deter)``.

    Mirrors the ``feat_size`` computation in ``WorldModel.__init__`` / ``ImagBehavior.__init__``.

    Args:
        config: The DreamerV3 configuration.

    Returns:
        The base feature width in scalars.
    """
    model = config.model
    if model.dyn_discrete:
        return model.dyn_stoch * model.dyn_discrete + model.dyn_deter
    return model.dyn_stoch + model.dyn_deter


def social_context_size(config: "DreamerV3Cfg") -> int:
    """Return the width contributed by the GAT social context c_t (0 when social disabled)."""
    social = config.model.social
    if not social.enabled:
        return 0
    return social.gat.out_dim


def dynamics_context_size(config: "DreamerV3Cfg") -> int:
    """Return the width contributed by the DALI dynamics context d_t (0 when DALI disabled)."""
    social = config.model.social
    if not social.enabled or not social.dali.enabled:
        return 0
    return social.dali.out_dim


def augmented_feat_size(config: "DreamerV3Cfg") -> int:
    """Return the width of the augmented feature feeding reward/value/actor heads.

    When ``social.enabled`` is False this equals :func:`base_feat_size`, guaranteeing the baseline
    network is byte-identical to the pre-social implementation.

    Args:
        config: The DreamerV3 configuration.

    Returns:
        The augmented feature width in scalars.
    """
    return (
        base_feat_size(config)
        + social_context_size(config)
        + dynamics_context_size(config)
    )
