"""Heterogeneous Graph Attention Network for Social-RSSM (C1, M3.2).

Computes social context c_t ∈ ℝ^{out_dim} from the decoded pedestrian node set
and the RSSM deterministic hidden state h_t.

Architecture (decode-then-GAT invariant):
  Input nodes: decoded peds (B, N+1, F) — node 0 = robot, 1..N = decoded peds.
  Two relation types: robot→ped (0) and ped→ped (1).
  ``layers`` stacked heterogeneous GAT layers.
  Output: validity-weighted mean-pool of ped nodes → c_t ∈ ℝ^{out_dim}.

Real-time budget:
  N=8 → adjacency is (B, N+N², H) = B×73×H attention scores per layer.
  All ops are batched scatter-ops; no Python loops inside forward().
  Inference at 20 Hz is dominated by the h_t→bias projection (one linear).

Conditioning on h_t:
  h_t is projected to a node-bias vector added to the initial embedding of all
  nodes. This injects world-model context without coupling the structural
  attention computation to h_t dimensionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .edges import build_edges, EDGE_PED2PED

if TYPE_CHECKING:
    from ..cfg import SocialGATCfg


class _HeteroGATLayer(nn.Module):
    """Single heterogeneous GAT layer with two relation types.

    Each relation type has independent key/value projections.
    Query projection is shared across relations.
    Output: residual + LayerNorm (h = LayerNorm(h + Attn(h))).

    All scatter ops run over E_max ≤ N + N² = 72 edges for N=8 — tiny.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self._H = num_heads
        self._D = out_dim // num_heads
        self._out = out_dim
        self._scale = self._D ** -0.5

        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        # Independent K/V per relation type (2 types).
        self.k_proj = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(2)])
        self.v_proj = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(2)])
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,           # (B, Np1, in_dim)
        edge_index: torch.Tensor,  # (B, 2, E)  -1 = inactive
        edge_type: torch.Tensor,   # (B, E)     -1 = inactive; 0=r2p, 1=p2p
    ) -> torch.Tensor:             # (B, Np1, out_dim)
        B, Np1, _ = h.shape
        H, D = self._H, self._D
        E = edge_index.shape[2]
        dev = h.device

        src = edge_index[:, 0]          # (B, E)
        dst = edge_index[:, 1]          # (B, E)
        active = (src >= 0) & (dst >= 0) & (edge_type >= 0)   # (B, E)

        src_s = src.clamp(min=0)
        dst_s = dst.clamp(min=0)

        # Gather source features for K/V.
        b_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, E)
        src_h = h[b_idx, src_s]   # (B, E, in_dim)

        # Both relation projections (relation selected via soft mask → avoids if-branch).
        k0 = self.k_proj[0](src_h).view(B, E, H, D)
        k1 = self.k_proj[1](src_h).view(B, E, H, D)
        v0 = self.v_proj[0](src_h).view(B, E, H, D)
        v1 = self.v_proj[1](src_h).view(B, E, H, D)

        is_p2p = (edge_type == EDGE_PED2PED).float().view(B, E, 1, 1)
        k = k0 + is_p2p * (k1 - k0)   # (B, E, H, D)
        v = v0 + is_p2p * (v1 - v0)

        # Destination-node queries.
        q = self.q_proj(h)[b_idx, dst_s].view(B, E, H, D)   # (B, E, H, D)

        # Scaled dot-product attention scores.
        attn = (q * k).sum(-1) * self._scale   # (B, E, H)

        # Mask inactive edges → -inf.
        attn = attn.masked_fill(~active.unsqueeze(-1), float("-inf"))

        # Per-destination softmax via scatter.
        # dst_exp: (B, E, H) — bucket indices for scatter ops.
        dst_exp = dst_s.unsqueeze(-1).expand(B, E, H)

        attn_max = torch.full((B, Np1, H), float("-inf"), device=dev, dtype=attn.dtype)
        attn_max.scatter_reduce_(1, dst_exp, attn, reduce="amax", include_self=True)
        attn_max = attn_max.clamp(min=float("-inf"))  # keep -inf for nodes with no in-edges

        shifted = (attn - attn_max.gather(1, dst_exp)).masked_fill(
            ~active.unsqueeze(-1), float("-inf")
        )
        exp_a = torch.exp(shifted).masked_fill(~active.unsqueeze(-1), 0.0)  # (B, E, H)

        exp_sum = torch.zeros(B, Np1, H, device=dev, dtype=attn.dtype)
        exp_sum.scatter_add_(1, dst_exp, exp_a)
        norm_a = exp_a / (exp_sum.gather(1, dst_exp) + 1e-6)  # (B, E, H)

        # Weighted value aggregation.
        wv = (norm_a.unsqueeze(-1) * v).to(h.dtype)           # (B, E, H, D)
        agg = torch.zeros(B, Np1, H, D, device=dev, dtype=h.dtype)
        dst_exp4 = dst_exp.unsqueeze(-1).expand(B, E, H, D)
        agg.scatter_add_(1, dst_exp4, wv)
        agg = agg.reshape(B, Np1, self._out)

        return self.norm(self.out_proj(agg) + self.res(h))


class GAT(nn.Module):
    """Social context encoder C1: heterogeneous GAT conditioned on h_t.

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

        # h_t → per-node additive bias (broadcast to all nodes before first layer).
        self.h_proj = nn.Linear(deter_size, cfg.hidden, bias=True)
        # Initial node embedding (same projection for robot and ped nodes).
        self.node_embed = nn.Linear(node_feat_dim, cfg.hidden, bias=False)

        # Stacked heterogeneous GAT layers.
        self.gat_layers = nn.ModuleList()
        d = cfg.hidden
        for _ in range(cfg.layers):
            self.gat_layers.append(_HeteroGATLayer(d, d, cfg.heads))

        # Readout: validity-weighted mean-pool → c_t.
        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.out_dim, bias=False),
            nn.LayerNorm(cfg.out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        decoded_peds_flat: torch.Tensor,   # (B, N*(F+1))  from heads["PedestrianNodeSetSpace"].mode()
        h_t: torch.Tensor,                 # (B, deter_size)
        max_peds: int,
        node_feat_dim: int,
    ) -> torch.Tensor:                     # (B, out_dim) = c_t
        """Compute c_t from decoded peds and RSSM hidden h_t.

        Decode-then-GAT invariant: decoded_peds_flat comes from the ped
        reconstruction head, NOT from raw observations — so this forward is
        identical at observe-time and imagination-time.
        """
        B = decoded_peds_flat.shape[0]
        N, F = max_peds, node_feat_dim

        # Unpack decoded flat tensor.
        peds = decoded_peds_flat.view(B, N, F + 1)   # (B, N, F+1)
        validity = peds[:, :, -1]                     # (B, N) decoded validity scores
        ped_feats = peds[:, :, :F]                    # (B, N, F)

        # Robot node: all-zero features (robot frame origin; zero velocity).
        robot_feats = torch.zeros(B, 1, F, device=decoded_peds_flat.device,
                                  dtype=decoded_peds_flat.dtype)

        # Full node feature tensor: (B, N+1, F). Index 0 = robot.
        node_feats = torch.cat([robot_feats, ped_feats], dim=1)   # (B, N+1, F)

        # Adjacency: threshold decoded validity at 0.5 for edge construction.
        mask = (validity > 0.5).float()    # (B, N)
        edge_index, edge_type = build_edges(
            nodes=node_feats,
            mask=mask,
            cfg=self._cfg,
        )

        # Initial node embeddings + h_t conditioning bias.
        h_node = self.node_embed(node_feats)              # (B, N+1, hidden)
        h_bias = self.h_proj(h_t).unsqueeze(1)            # (B, 1,   hidden)
        h_node = h_node + h_bias                          # broadcast to all nodes

        # Stacked GAT layers.
        for layer in self.gat_layers:
            h_node = layer(h_node, edge_index, edge_type)  # (B, N+1, hidden)

        # Readout: validity-weighted mean-pool over ped nodes (indices 1..N).
        ped_h = h_node[:, 1:, :]         # (B, N, hidden)
        w = mask.unsqueeze(-1)           # (B, N, 1) — suppress padded nodes
        w_sum = w.sum(1).clamp(min=1.0)  # (B, 1)
        pooled = (ped_h * w).sum(1) / w_sum   # (B, hidden)

        return self.readout(pooled)      # (B, out_dim) = c_t
