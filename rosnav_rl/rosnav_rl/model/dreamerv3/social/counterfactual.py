"""Counterfactual b-swap imagination divergence (§4.3 diagnostic (ii)).

cSRSSM's context ``b`` conditions the RSSM transition (``img_step``), not just a head
(see ``social/context.py``). If ``b`` carries real crowd-regime information, swapping it
while holding the start state and action sequence fixed should change the imagined
pedestrian rollout; if ``b`` has collapsed to the prior, swapping it should not.
``bswap_divergence`` turns that qualitative claim into a number.
"""

from __future__ import annotations

import torch


def bswap_divergence(
    wm,
    start_state: dict,
    actions: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
) -> torch.Tensor:
    """Per-step L2 divergence between pedestrian predictions imagined under two contexts.

    Imagines the same action sequence from the same start state twice, once conditioned
    on ``b_a`` and once on ``b_b``, then decodes both imagined rollouts through the world
    model's pedestrian decoder head and measures how far apart the decoded predictions are
    at each imagined step.

    Args:
        wm: World model exposing ``.dynamics`` (an RSSM/TSSM instance with
            ``imagine_with_action`` and ``get_feat``) and ``.heads["PedestrianNodeSetSpace"]``
            (the pedestrian reconstruction head, ``models.py:237``).
        start_state: RSSM/TSSM state dict to imagine from. Must come from a posterior
            ``obs_step``/``observe`` output, not ``dynamics.initial()`` -- for the
            TransformerCell/TSSM backbones the state carries a K/V cache, and imagining
            from an empty cache changes the rollout for reasons unrelated to ``b``.
        actions: ``(B, T, action_dim)`` action sequence, identical for both rollouts.
        b_a: ``(B, b_dim)`` context for the first rollout.
        b_b: ``(B, b_dim)`` context for the second rollout.

    Returns:
        ``(T,)`` tensor: mean-over-batch L2 divergence between the two decoded pedestrian
        predictions at each imagined step.

    Note:
        ``img_step`` samples the stochastic state (``sample=True``, not exposed as a
        parameter through ``imagine_with_action``). A counterfactual comparison is only
        meaningful if the two rollouts share every source of randomness except ``context``,
        so this snapshots the RNG state before the first rollout and restores it before the
        second -- without that, the two draws would differ even when ``b_a == b_b``, and the
        reported divergence would be contaminated by sampling noise rather than isolating the
        effect of swapping ``b``.
    """
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    prior_a = wm.dynamics.imagine_with_action(actions, start_state, context=b_a)

    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    prior_b = wm.dynamics.imagine_with_action(actions, start_state, context=b_b)

    feat_a = wm.dynamics.get_feat(prior_a)
    feat_b = wm.dynamics.get_feat(prior_b)

    peds_a = wm.heads["PedestrianNodeSetSpace"](feat_a).mode()
    peds_b = wm.heads["PedestrianNodeSetSpace"](feat_b).mode()

    return (peds_a - peds_b).pow(2).sum(-1).sqrt().mean(dim=0)
