"""HEIGHT-style heterogeneous graph attention for Social-RSSM (Part 2, supporting).

HEIGHT (IEEE T-ASE) — heterogeneous spatio-temporal graph with genuinely separate per-edge-type
multi-head attention, vs. the shared-Q soft-blend in ``social/gat.py`` (``_HeteroGATLayer``:
one shared query projection, K/V linearly blended between two relation types via a soft mask).
This module gives each relation type its own independent Q/K/V projections and attention
distribution; per-destination contributions from each type are summed. Same in/out contract as
``GAT`` (``out_dim`` unchanged), selectable via ``SocialGATCfg.variant``.

Relation types (3, vs. GAT's 2):
    0 = robot -> ped   (EDGE_ROBOT2PED, from social/edges.py)
    1 = ped   -> ped   (EDGE_PED2PED, from social/edges.py)
    2 = ped   -> robot (new here: reverse of robot->ped, lets the robot node aggregate the
        crowd through the graph itself rather than only via the h_t bias injection)

Real-time budget: 3 typed attention passes over the same E_max = N + N² <= 72 edges (N=8);
still tiny, fully batched scatter ops, no Python loop over edges (only over the 3 types).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .edges import build_edges, EDGE_ROBOT2PED, EDGE_PED2PED

if TYPE_CHECKING:
    from ..cfg import SocialGATCfg

EDGE_PED2ROBOT: int = 2
_NUM_EDGE_TYPES: int = 3


class _HeteroTypedAttnLayer(nn.Module):
    """Heterogeneous GAT layer with fully independent Q/K/V per relation type.

    Unlike ``_HeteroGATLayer`` (shared Q, blended K/V), each type computes its own attention
    distribution over its own edges; per-destination results are summed across types.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self._H = num_heads
        self._D = out_dim // num_heads
        self._out = out_dim
        self._scale = self._D ** -0.5

        self.q_proj = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(_NUM_EDGE_TYPES)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(_NUM_EDGE_TYPES)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(_NUM_EDGE_TYPES)]
        )
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def _typed_attention(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        active_t: torch.Tensor,  # (B, E) bool, edges of this type
        type_id: int,
    ) -> torch.Tensor:
        B, Np1, _ = h.shape
        H, D = self._H, self._D
        E = edge_index.shape[2]
        dev = h.device

        src = edge_index[:, 0].clamp(min=0)
        dst = edge_index[:, 1].clamp(min=0)
        b_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, E)

        src_h = h[b_idx, src]  # (B, E, in_dim)
        k = self.k_proj[type_id](src_h).view(B, E, H, D)
        v = self.v_proj[type_id](src_h).view(B, E, H, D)
        q = self.q_proj[type_id](h)[b_idx, dst].view(B, E, H, D)

        attn = (q * k).sum(-1) * self._scale  # (B, E, H)
        attn = attn.masked_fill(~active_t.unsqueeze(-1), float("-inf"))

        dst_exp = dst.unsqueeze(-1).expand(B, E, H)
        attn_max = torch.full((B, Np1, H), float("-inf"), device=dev, dtype=attn.dtype)
        attn_max.scatter_reduce_(1, dst_exp, attn, reduce="amax", include_self=True)

        shifted = (attn - attn_max.gather(1, dst_exp)).masked_fill(
            ~active_t.unsqueeze(-1), float("-inf")
        )
        exp_a = torch.exp(shifted).masked_fill(~active_t.unsqueeze(-1), 0.0)  # (B, E, H)

        exp_sum = torch.zeros(B, Np1, H, device=dev, dtype=attn.dtype)
        exp_sum.scatter_add_(1, dst_exp, exp_a)
        norm_a = exp_a / (exp_sum.gather(1, dst_exp) + 1e-6)  # (B, E, H)

        wv = (norm_a.unsqueeze(-1) * v).to(h.dtype)  # (B, E, H, D)
        agg = torch.zeros(B, Np1, H, D, device=dev, dtype=h.dtype)
        dst_exp4 = dst_exp.unsqueeze(-1).expand(B, E, H, D)
        agg.scatter_add_(1, dst_exp4, wv)
        return agg.reshape(B, Np1, self._out)

    def forward(
        self,
        h: torch.Tensor,           # (B, Np1, in_dim)
        edge_index: torch.Tensor,  # (B, 2, E)  -1 = inactive
        edge_type: torch.Tensor,   # (B, E)     -1 = inactive; 0=r2p, 1=p2p, 2=p2r
    ) -> torch.Tensor:             # (B, Np1, out_dim)
        base_active = (edge_index[:, 0] >= 0) & (edge_index[:, 1] >= 0) & (edge_type >= 0)

        agg_sum = 0.0
        for t in range(_NUM_EDGE_TYPES):
            active_t = base_active & (edge_type == t)
            if not bool(active_t.any()):
                continue
            agg_sum = agg_sum + self._typed_attention(h, edge_index, active_t, t)

        if isinstance(agg_sum, float):
            agg_sum = torch.zeros(
                h.shape[0], h.shape[1], self._out, device=h.device, dtype=h.dtype
            )

        return self.norm(self.out_proj(agg_sum) + self.res(h))


def _reverse_edges(edge_index: torch.Tensor, edge_type: torch.Tensor) -> tuple:
    """Ped->robot edges: robot->ped edges (type EDGE_ROBOT2PED) with src/dst swapped."""
    is_r2p = edge_type == EDGE_ROBOT2PED
    src, dst = edge_index[:, 0], edge_index[:, 1]
    p2r_src = torch.where(is_r2p, dst, torch.full_like(dst, -1))
    p2r_dst = torch.where(is_r2p, src, torch.full_like(src, -1))
    p2r_type = torch.where(is_r2p, torch.full_like(edge_type, EDGE_PED2ROBOT), torch.full_like(edge_type, -1))
    p2r_index = torch.stack([p2r_src, p2r_dst], dim=1)
    return p2r_index, p2r_type


class HeightGAT(nn.Module):
    """Social context encoder C1 (HEIGHT variant): per-edge-type multi-head attention.

    Same interface as ``GAT`` (``social/gat.py``) so ``WorldModel`` can select between the two
    via ``SocialGATCfg.variant`` without changing any downstream shape (``out_dim`` unchanged).

    Args:
        cfg:           SocialGATCfg — out_dim, heads, hidden, layers, radius_m, converge_angle_deg.
        node_feat_dim: F — number of kinematic features per node (excluding validity flag).
        deter_size:    Dimensionality of the RSSM deterministic state h_t.
        device:        Torch device string.
    """

    def __init__(
        self,
        cfg: "SocialGATCfg",
        node_feat_dim: int = 5,
        deter_size: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self._cfg = cfg

        self.h_proj = nn.Linear(deter_size, cfg.hidden, bias=True)
        self.node_embed = nn.Linear(node_feat_dim, cfg.hidden, bias=False)

        self.gat_layers = nn.ModuleList()
        d = cfg.hidden
        for _ in range(cfg.layers):
            self.gat_layers.append(_HeteroTypedAttnLayer(d, d, cfg.heads))

        # Readout: robot-node (crowd-aggregated via ped->robot edges) concat ped-pool -> c_t.
        self.readout = nn.Sequential(
            nn.Linear(2 * cfg.hidden, cfg.out_dim, bias=False),
            nn.LayerNorm(cfg.out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        decoded_peds_flat: torch.Tensor,   # (B, N*(F+1))
        h_t: torch.Tensor,                 # (B, deter_size)
        max_peds: int,
        node_feat_dim: int,
    ) -> torch.Tensor:                     # (B, out_dim) = c_t
        B = decoded_peds_flat.shape[0]
        N, F = max_peds, node_feat_dim

        peds = decoded_peds_flat.view(B, N, F + 1)
        validity = peds[:, :, -1]
        ped_feats = peds[:, :, :F]

        robot_feats = torch.zeros(
            B, 1, F, device=decoded_peds_flat.device, dtype=decoded_peds_flat.dtype
        )
        node_feats = torch.cat([robot_feats, ped_feats], dim=1)  # (B, N+1, F)

        mask = (validity > 0.5).float()
        r2p_index, r2p_type = build_edges(nodes=node_feats, mask=mask, cfg=self._cfg)
        p2r_index, p2r_type = _reverse_edges(r2p_index, r2p_type)
        edge_index = torch.cat([r2p_index, p2r_index], dim=2)
        edge_type = torch.cat([r2p_type, p2r_type], dim=1)

        h_node = self.node_embed(node_feats)
        h_bias = self.h_proj(h_t).unsqueeze(1)
        h_node = h_node + h_bias

        for layer in self.gat_layers:
            h_node = layer(h_node, edge_index, edge_type)

        robot_h = h_node[:, 0, :]         # (B, hidden) -- now crowd-aggregated via p2r edges
        ped_h = h_node[:, 1:, :]          # (B, N, hidden)
        w = mask.unsqueeze(-1)
        w_sum = w.sum(1).clamp(min=1.0)
        ped_pooled = (ped_h * w).sum(1) / w_sum   # (B, hidden)

        return self.readout(torch.cat([robot_h, ped_pooled], dim=-1))  # (B, out_dim)
