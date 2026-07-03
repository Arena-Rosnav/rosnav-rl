"""DALI dynamics-context encoder for Social-Dreamer (C2, M4.1).

Encodes K=40 decoded pedestrian trajectory steps per ped into a per-ped latent
dynamics context d_t, then mean-pools across valid peds to get the batch context.

Architecture:
  For each ped slot i:
    GRU(hidden) over K decoded position+velocity frames → h_i^K
    Linear(h_i^K) → d_i ∈ R^{out_dim}
  d_t = validity-weighted mean of {d_i} (same mask as GAT, peds=decoded validity>0.5)

Auxiliary forward-prediction loss (L_dyn):
  Using d_t and the current decoded ped state x_i^t, predict next step x_hat_i^{t+1}.
  L_dyn = (1/N_valid) Σ_i || x_hat_i^{t+1} - x_i^{t+1} ||^2
  where x_i^{t+1} is the decoded ped at the NEXT rollout step (no simulator label).

Training notes:
  - L_dyn is computed ONLY at observe-time (not inside imagination rollouts).
  - Imagination backprop cap: for imagination step > imag_backprop_steps (=5),
    the decoded peds fed to DALI are detached (handled in WorldModel._get_augmented_feat).
  - The GRU runs over the K-step trajectory buffer, not a single decoded frame.
    At observe-time the buffer comes from PedestrianTrajectoryBufferGenerator.
    At imagine-time the buffer is seeded from the K real steps before horizon start,
    then extended by decoding imagined RSSM states.

Real-time budget:
  GRU(hidden=64, K=40, N=8): 8 parallel GRU passes × 40 steps × 64 hidden = fast.
  Total FLOPS ≈ 8 × 40 × 64 × 64 × 6 (GRU gates) ≈ 786 kFLOPS — sub-millisecond.
  All N peds are batched into one GRU call: input (N, K, F), batch-first GRU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..cfg import SocialDALICfg


class DALI(nn.Module):
    """Trajectory GRU producing the dynamics context d_t and a forward-prediction head.

    Args:
        cfg:           DALI configuration (out_dim, hidden, k_steps, lambda_dyn).
        node_feat_dim: Width F of a single pedestrian node feature vector.
        device:        Torch device string.
    """

    def __init__(
        self,
        cfg: "SocialDALICfg",
        node_feat_dim: int = 5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._F = node_feat_dim

        # GRU over K-step trajectory per ped.
        # Input per step: F kinematic features (pos_x, pos_y, vel_x, vel_y, [social]).
        # batch_first=True → input (N, K, F); output (N, K, hidden).
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=cfg.hidden,
            num_layers=1,
            batch_first=True,
        )
        # Project final GRU hidden state → per-ped dynamics latent d_i.
        self.d_proj = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.out_dim, bias=False),
            nn.LayerNorm(cfg.out_dim),
            nn.SiLU(),
        )
        # Forward-prediction head: (d_i ∥ x_i^t) → x_hat_i^{t+1}.
        # Input: out_dim + node_feat_dim; output: node_feat_dim.
        self.pred_head = nn.Linear(cfg.out_dim + node_feat_dim, node_feat_dim, bias=True)

    def forward(
        self,
        traj: torch.Tensor,    # (B, N, K, F) — K-step decoded ped trajectories
        validity: torch.Tensor, # (B, N)        — decoded ped validity mask (0/1)
    ) -> torch.Tensor:         # (B, out_dim) = d_t
        """Compute the pooled dynamics context d_t from K-step decoded trajectories.

        The K-step trajectory buffer is built from:
          - Observe-time: PedestrianTrajectoryBufferGenerator (real observations)
          - Imagine-time: K decoded RSSM steps preceding imagination start, then
                         extended by decoding imagined states (up to cap=5 steps).

        Args:
            traj:     ``(B, N, K, F)`` float32. Left-padded with zeros for peds
                      with fewer than K history frames.
            validity: ``(B, N)`` float32 in [0, 1].  1.0 = real ped, 0.0 = padding.

        Returns:
            d_t: ``(B, out_dim)`` dynamics context.
        """
        B, N, K, F = traj.shape

        # Merge batch and ped dimensions for batched GRU: (B*N, K, F).
        traj_flat = traj.reshape(B * N, K, F)
        _, h_n = self.gru(traj_flat)   # h_n: (1, B*N, hidden)
        h_last = h_n.squeeze(0).reshape(B, N, self._cfg.hidden)  # (B, N, hidden)

        # Per-ped dynamics latent.
        d_per_ped = self.d_proj(h_last)   # (B, N, out_dim)

        # Validity-weighted mean-pool over ped dimension.
        w = validity.unsqueeze(-1).clamp(0.0, 1.0)   # (B, N, 1)
        w_sum = w.sum(1).clamp(min=1.0)               # (B, 1)
        d_t = (d_per_ped * w).sum(1) / w_sum          # (B, out_dim)
        return d_t

    def forward_predict(
        self,
        traj: torch.Tensor,     # (B, N, K, F) — K-step decoded trajectories
        validity: torch.Tensor,  # (B, N) float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-ped next-step predictions x_hat^{t+1} for L_dyn.

        Returns:
            x_hat:    ``(B, N, F)`` predicted next-step features for valid peds.
            d_per_ped: ``(B, N, out_dim)`` per-ped dynamics latents (for diagnostics).
        """
        B, N, K, F = traj.shape

        # Run GRU (same as forward but we need per-ped latents before pooling).
        traj_flat = traj.reshape(B * N, K, F)
        _, h_n = self.gru(traj_flat)
        h_last = h_n.squeeze(0).reshape(B, N, self._cfg.hidden)
        d_per_ped = self.d_proj(h_last)   # (B, N, out_dim)

        # Current ped state = last frame of trajectory (most recent decoded position).
        x_current = traj[:, :, -1, :]    # (B, N, F)

        # Concat d_i and x_i^t → predict x_hat_i^{t+1}.
        inp = torch.cat([d_per_ped, x_current], dim=-1)   # (B, N, out_dim + F)
        x_hat = self.pred_head(inp)                        # (B, N, F)
        return x_hat, d_per_ped


def dali_loss(
    dali: DALI,
    traj_t: torch.Tensor,    # (B, N, K, F) at step t
    traj_tp1: torch.Tensor,  # (B, N, K, F) at step t+1 (shifted replay buffer)
    validity: torch.Tensor,  # (B, N) float32
) -> torch.Tensor:           # scalar
    """Compute the auxiliary forward-prediction loss L_dyn.

    L_dyn = mean over valid peds of || x_hat_i^{t+1} - x_i^{t+1} ||^2

    Called at observe-time only; never inside imagination rollouts.

    Args:
        dali:      DALI module.
        traj_t:    Trajectory buffer at current step t.
        traj_tp1:  Trajectory buffer at next step t+1 (shifted slice from replay).
        validity:  Validity mask (B, N).

    Returns:
        Scalar L_dyn loss.
    """
    x_hat, _ = dali.forward_predict(traj_t, validity)    # (B, N, F)
    x_true = traj_tp1[:, :, -1, :]                       # (B, N, F) — last frame of t+1 buffer

    sq_err = F.mse_loss(x_hat, x_true, reduction="none").sum(-1)   # (B, N)

    # Mask out padded peds.
    mask = (validity > 0.5).float()   # (B, N)
    n_valid = mask.sum().clamp(min=1.0)
    loss = (sq_err * mask).sum() / n_valid
    return loss
