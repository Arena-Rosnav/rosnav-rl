"""Crowd-behavior context encoder for the Contextual Social RSSM (cSRSSM).

Casts social navigation as a contextual MDP: a slow, probabilistic crowd-behavior code
``b`` (aggressiveness, yielding, preferred speed, personal-space, local policy) is inferred
once per trajectory window and held fixed while it conditions the RSSM *transition*
(``RSSM.img_step``), not just the feature. This is the social analogue of DALI's latent
environment-dynamics context (friction/gravity), and is what enables counterfactual social
imagination: perturbing ``b`` with ``(s, a)`` held fixed changes the imagined ped rollout.

Architecture (repurposes the DALI trajectory GRU, see ``social/dali.py``):
  1. Per-timestep pooling: validity-weighted mean over the N pedestrian nodes at each of the
     K window steps -> a (B, K, F) pooled social summary sequence. Cheap (no attention),
     consistent with DALI's own per-ped pooling.
  2. Temporal GRU over the K-step window -> final hidden state h_K.
  3. Two linear heads on h_K -> (mean, std) of q(b | window), a diagonal Gaussian posterior.
     KL(q(b) || N(0,1)) is added to the ELBO as an information-bottleneck regularizer
     (see ``social_context_kl``), making b the minimal sufficient dynamics context.
  4. Prediction head: from (b, pooled crowd summary at step t) predict the pooled summary at
     step t+1, for steps *beyond* the inference window (see ``context_pred_loss``). This is
     the primary identifiability mechanism (VariBAD-style: the context is trained by decoding
     future dynamics, counteracting the KL bottleneck's collapse pressure). Out-of-window
     targets force b to extrapolate the crowd regime rather than memorize the window.

Realtime budget: one GRU pass over the K-step window per training sequence (K=16 in the
production config, K<batch_length required so out-of-window prediction targets exist), and one
amortized GRU step per deploy tick (b is only re-inferred when the window slides) — negligible
next to the RSSM/GAT forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..cfg import SocialContextCfg


class SocialContextEncoder(nn.Module):
    """Infers the crowd-behavior context posterior q(b | window) from a ped trajectory window.

    Args:
        cfg:           Context configuration (b_dim, hidden, window, kl_scale, infonce_scale).
        node_feat_dim: Width F of a single pedestrian node feature vector.
        device:        Torch device string.
    """

    def __init__(
        self,
        cfg: "SocialContextCfg",
        node_feat_dim: int = 5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._F = node_feat_dim

        # Temporal GRU over the K-step window of pooled per-step ped summaries.
        # Input per step: F pooled kinematic features; batch_first=True -> (B, K, F).
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=cfg.hidden,
            num_layers=1,
            batch_first=True,
        )
        self.mean_head = nn.Linear(cfg.hidden, cfg.b_dim)
        self.std_head = nn.Linear(cfg.hidden, cfg.b_dim)

        # Prediction head for the identifiability loss: (b, pooled_t) -> pooled_{t+1}.
        # Small on purpose — its only job is to route gradient pressure into b, not to be
        # a good crowd forecaster (that is the RSSM's job).
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.b_dim + node_feat_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, node_feat_dim),
        )

    def pool_step(self, peds: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        """Validity-weighted mean-pool of pedestrian nodes at a single timestep.

        Args:
            peds:     ``(B, N, F)`` pedestrian node features at one step.
            validity: ``(B, N)`` float32 in [0, 1]. 1.0 = real ped, 0.0 = padding.

        Returns:
            ``(B, F)`` pooled social summary for that step.
        """
        w = validity.unsqueeze(-1).clamp(0.0, 1.0)  # (B, N, 1)
        w_sum = w.sum(1).clamp(min=1.0)  # (B, 1)
        return (peds * w).sum(1) / w_sum  # (B, F)

    def forward(
        self,
        window: torch.Tensor,  # (B, K, N, F) — K-step window of decoded/observed ped node-sets
        validity: torch.Tensor,  # (B, K, N)    — per-step ped validity mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute q(b | window) = (mean, std).

        Args:
            window:   ``(B, K, N, F)`` float32. Left-padded with zeros for windows shorter
                      than K (e.g. near episode start).
            validity: ``(B, K, N)`` float32 in [0, 1].

        Returns:
            mean: ``(B, b_dim)``.
            std:  ``(B, b_dim)``, positive (softplus).
        """
        B, K, N, Fd = window.shape

        # Per-step pooling: (B, K, N, F) -> (B, K, F).
        pooled = self.pool_step(
            window.reshape(B * K, N, Fd), validity.reshape(B * K, N)
        ).reshape(B, K, Fd)

        _, h_n = self.gru(pooled)  # h_n: (1, B, hidden)
        h_last = h_n.squeeze(0)  # (B, hidden)

        mean = self.mean_head(h_last)
        std = F.softplus(self.std_head(h_last)) + 0.1
        return mean, std

    def sample(
        self, mean: torch.Tensor, std: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Draw b ~ q(b|window) at train time, or b = mean at deploy (eval_state_mean-style)."""
        if not sample:
            return mean
        eps = torch.randn_like(mean)
        return mean + eps * std


def context_kl_loss(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """KL(q(b|window) || N(0,1)), the information-bottleneck term of the ELBO extension.

    Args:
        mean: ``(B, b_dim)``.
        std:  ``(B, b_dim)``, positive.

    Returns:
        Scalar KL loss, mean-reduced over batch and dims.
    """
    kl = 0.5 * (std.pow(2) + mean.pow(2) - 1.0 - 2.0 * std.clamp(min=1e-6).log())
    return kl.sum(-1).mean()


def context_infonce_loss(b: torch.Tensor, episode_id: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE identifiability term: same-episode b's are positives, other episodes negatives.

    Secondary/ablation-only mechanism (config ``infonce_scale``, default 0): the primary
    identifiability pressure is ``context_pred_loss``. Kept because it is a useful contrast
    axis, but note its known weaknesses: the MI bound is capped at log(#distinct episodes per
    batch), and under domain randomization distinct episodes sharing the same simulator regime
    are pushed apart as false negatives.

    Args:
        b:          ``(B, b_dim)`` sampled context codes (one per batch element).
        episode_id: ``(B,)`` integer/long episode identifiers; elements sharing an id are
                    treated as positives of each other.
        temperature: Softmax temperature.

    Returns:
        Scalar InfoNCE loss. Returns 0 if the batch has fewer than 2 elements.
    """
    Bsz = b.shape[0]
    if Bsz < 2:
        return b.new_zeros(())

    b_norm = F.normalize(b, dim=-1)
    sim = b_norm @ b_norm.t() / temperature  # (B, B)

    pos_mask = episode_id.unsqueeze(0) == episode_id.unsqueeze(1)  # (B, B)
    self_mask = torch.eye(Bsz, dtype=torch.bool, device=b.device)
    pos_mask = pos_mask & ~self_mask

    # Rows with no positive (unique episode in batch) are excluded from the loss.
    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return b.new_zeros(())

    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    # 0 * -inf = nan at the (masked-out) self entries, so select with `where` rather than
    # multiply by the mask.
    log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_log_prob = log_prob.sum(-1) / pos_mask.sum(-1).clamp(min=1.0)

    return -pos_log_prob[has_pos].mean()


def context_pred_loss(
    encoder: SocialContextEncoder,
    b: torch.Tensor,
    pooled_future: torch.Tensor,
    step_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prediction-driven identifiability term: from (b, pooled_t) predict pooled_{t+1}.

    Primary identifiability mechanism for b (VariBAD-style: the context posterior is trained
    by decoding future dynamics, [Zintgraf et al., ICLR 2020]). The targets are pooled crowd
    summaries of steps *after* the window b was inferred from, so b cannot satisfy the loss by
    memorizing window content — it must carry regime information that extrapolates. This is
    the counter-pressure to the KL bottleneck's collapse pull: a collapsed b reduces the head
    to a persistence baseline, which costs prediction accuracy exactly when crowd regimes
    differ.

    Deliberately regime-level (pooled crowd summaries, not per-ped states): per-ped next-step
    prediction is DALI's job (``dali.py``, d_t); giving b the same target would make the two
    codes redundant and the context on/off ablation meaningless.

    Args:
        encoder:       The SocialContextEncoder owning ``pred_head``.
        b:             ``(B, b_dim)`` sampled context codes.
        pooled_future: ``(B, M, F)`` pooled crowd summaries (``pool_step`` output) of the M
                       steps from the last window step onward. Consecutive pairs form
                       (input, target).
        step_valid:    Optional ``(B, M)`` bool/float — True where the step had at least one
                       valid ped. Pairs whose *target* step is empty are excluded (predicting
                       pooled zero-padding would corrupt b's gradient signal).

    Returns:
        Scalar MSE loss over valid consecutive pairs. Returns 0 if M < 2 or no pair is valid.
    """
    if pooled_future.shape[1] < 2:
        return b.new_zeros(())

    inp = pooled_future[:, :-1]  # (B, M-1, F)
    target = pooled_future[:, 1:]  # (B, M-1, F)
    M = inp.shape[1]
    b_exp = b.unsqueeze(1).expand(-1, M, -1)  # (B, M-1, b_dim)
    pred = encoder.pred_head(torch.cat([inp, b_exp], dim=-1))  # (B, M-1, F)

    err = (pred - target).pow(2).mean(-1)  # (B, M-1)
    if step_valid is not None:
        pair_valid = step_valid[:, 1:].to(err.dtype)  # target step must be non-empty
        denom = pair_valid.sum()
        if denom < 1:
            return b.new_zeros(())
        return (err * pair_valid).sum() / denom
    return err.mean()


def context_pred_floor(
    pooled_future: torch.Tensor,
    step_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-velocity persistence baseline for ``context_pred_loss``: the MSE of predicting
    pooled_{t+1} = pooled_t, i.e. what the head achieves using no context at all.

    Diagnostic only (never enters the loss). The logged gap ``floor - pred_loss`` is the
    identifiability health check of §2.1: b earns its keep only by beating persistence, so a
    vanishing gap is the early-warning signal for context collapse — the head has learned to
    ignore b and coast on the fact that pooled crowd summaries barely move between steps. Uses
    the exact masking semantics of ``context_pred_loss`` so the two numbers are comparable.

    Args:
        pooled_future: ``(B, M, F)`` pooled crowd summaries (``pool_step`` output).
        step_valid:    Optional ``(B, M)`` bool/float; pairs whose *target* step is empty are
                       excluded, matching ``context_pred_loss``.

    Returns:
        Scalar MSE of the persistence baseline. Returns 0 if M < 2 or no pair is valid.
    """
    if pooled_future.shape[1] < 2:
        return pooled_future.new_zeros(())

    err = (pooled_future[:, 1:] - pooled_future[:, :-1]).pow(2).mean(-1)  # (B, M-1)
    if step_valid is not None:
        pair_valid = step_valid[:, 1:].to(err.dtype)
        denom = pair_valid.sum()
        if denom < 1:
            return pooled_future.new_zeros(())
        return (err * pair_valid).sum() / denom
    return err.mean()
