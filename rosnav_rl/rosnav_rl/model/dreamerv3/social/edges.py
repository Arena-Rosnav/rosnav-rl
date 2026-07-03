"""Graph edge construction for the Social-RSSM (C1, M3.1).

Builds the adjacency used by the heterogeneous GAT from pedestrian node
positions/velocities and a validity mask. Two relation types are produced:

    EDGE_ROBOT2PED  = 0   robot → each valid pedestrian
    EDGE_PED2PED    = 1   pedestrian → pedestrian (proximity OR converging)

An edge exists when EITHER the pair is within ``radius_m`` OR the pair is
velocity-converging (relative-heading angle < ``converge_angle_deg``).

Convention: node 0 is the robot; nodes 1..N are pedestrians (possibly padded).
The robot node features are (0, 0, 0, 0) in robot frame — position is origin,
velocity is zero (robot-centric frame).

Real-time budget: N=8 → N² = 64 operations. All work is batched-vectorized;
no Python loops over edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from ..cfg import SocialGATCfg

# Relation-type constants (kept as plain ints — no enum to avoid import overhead).
EDGE_ROBOT2PED: int = 0
EDGE_PED2PED: int = 1


def build_edges(
    nodes: torch.Tensor,
    mask: torch.Tensor,
    cfg: "SocialGATCfg",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build typed adjacency for the heterogeneous GAT.

    Args:
        nodes: ``(B, N+1, F)`` float32.  Node 0 = robot (pos=0,vel=0).
               Nodes 1..N = pedestrians (robot-frame positions + velocities).
               F ≥ 4: columns [rel_x, rel_y, vel_x, vel_y, ...].
        mask:  ``(B, N)`` bool / float32.  1 = real ped, 0 = padding.
               The robot node (index 0) has no mask entry — always valid.
        cfg:   GAT config supplying ``radius_m`` and ``converge_angle_deg``.

    Returns:
        edge_index: ``(B, 2, E_max)`` int64 — source/dest node indices.
                    E_max = N+1 (robot→ped) + N² (ped→ped). Padded with -1
                    for inactive edges so downstream code can mask them.
        edge_type:  ``(B, E_max)`` int64 — 0=robot→ped, 1=ped→ped.

    Performance note: all ops are batched matrix ops; no Python loops.
    For N=8, E_max = 9 + 64 = 73 entries per batch element — trivial.
    """
    B, Np1, F = nodes.shape
    N = Np1 - 1          # number of ped slots
    device = nodes.device
    dtype_bool = torch.bool

    # ── 1. Valid pedestrian mask ─────────────────────────────────────────────
    # mask: (B, N) → bool, shifted to node indices (robot=0, peds=1..N)
    ped_valid: torch.Tensor = mask.bool()  # (B, N)

    # ── 2. Robot → ped edges (relation 0) ───────────────────────────────────
    # One edge per valid ped: src=0 (robot), dst=1..N.
    r2p_src = torch.zeros(B, N, dtype=torch.int64, device=device)          # (B, N)
    r2p_dst = torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
    r2p_type = torch.full((B, N), EDGE_ROBOT2PED, dtype=torch.int64, device=device)
    # Mask out padded peds with sentinel -1.
    r2p_src  = torch.where(ped_valid, r2p_src,  torch.full_like(r2p_src, -1))
    r2p_dst  = torch.where(ped_valid, r2p_dst,  torch.full_like(r2p_dst, -1))
    r2p_type = torch.where(ped_valid, r2p_type, torch.full_like(r2p_type, -1))

    # ── 3. Ped → ped edges (relation 1) ─────────────────────────────────────
    # Criterion: within radius_m OR velocity-converging (heading < converge_angle_deg).
    ped_nodes = nodes[:, 1:, :]   # (B, N, F)
    pos = ped_nodes[:, :, :2]     # (B, N, 2)  relative positions in robot frame
    vel = ped_nodes[:, :, 2:4]    # (B, N, 2)  relative velocities

    # Pairwise distance: (B, N, N)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N, N, 2)
    dist = torch.norm(diff, dim=-1)              # (B, N, N)

    proximity = dist < cfg.radius_m              # (B, N, N) bool

    # Velocity-converging criterion:
    #   The angle between (displacement vector i→j) and velocity of j < threshold.
    #   A ped j is converging toward i if it is heading roughly toward i.
    cos_thresh = torch.cos(
        torch.tensor(cfg.converge_angle_deg * torch.pi / 180.0, device=device)
    )
    # Relative displacement from j to i: -diff
    displace_j2i = -diff                                    # (B, N, N, 2)
    vel_j = vel.unsqueeze(1).expand(-1, N, -1, -1)         # (B, N, N, 2)
    vel_norm = vel_j / (torch.norm(vel_j, dim=-1, keepdim=True) + 1e-6)
    disp_norm = displace_j2i / (torch.norm(displace_j2i, dim=-1, keepdim=True) + 1e-6)
    cos_angle = (vel_norm * disp_norm).sum(-1)              # (B, N, N)
    converging = cos_angle > cos_thresh                     # (B, N, N)

    # Combined adjacency (no self-loops).
    eye = torch.eye(N, dtype=dtype_bool, device=device).unsqueeze(0)
    adj = (proximity | converging) & ~eye   # (B, N, N)

    # Mask out edges involving padded peds.
    # valid_pair[b, i, j] = ped_valid[b, i] & ped_valid[b, j]
    v_i = ped_valid.unsqueeze(2)   # (B, N, 1)
    v_j = ped_valid.unsqueeze(1)   # (B, 1, N)
    adj = adj & v_i & v_j          # (B, N, N)

    # Flatten to edge list (i, j in ped-local index), shift to graph node index (+1).
    i_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, N).reshape(-1)  # N²
    j_idx = torch.arange(N, device=device).unsqueeze(0).expand(N, N).reshape(-1)  # N²
    # (B, N²)
    p2p_src_flat = (i_idx + 1).unsqueeze(0).expand(B, -1)  # ped → graph node (+1)
    p2p_dst_flat = (j_idx + 1).unsqueeze(0).expand(B, -1)
    adj_flat = adj.reshape(B, -1)                           # (B, N²)
    p2p_type_flat = torch.full((B, N * N), EDGE_PED2PED, dtype=torch.int64, device=device)

    # Mask inactive edges with -1.
    p2p_src  = torch.where(adj_flat, p2p_src_flat, torch.full_like(p2p_src_flat, -1))
    p2p_dst  = torch.where(adj_flat, p2p_dst_flat, torch.full_like(p2p_dst_flat, -1))
    p2p_type = torch.where(adj_flat, p2p_type_flat, torch.full_like(p2p_type_flat, -1))

    # ── 4. Concatenate robot→ped and ped→ped ─────────────────────────────────
    src  = torch.cat([r2p_src,  p2p_src],  dim=1)   # (B, N + N²)
    dst  = torch.cat([r2p_dst,  p2p_dst],  dim=1)
    etype = torch.cat([r2p_type, p2p_type], dim=1)

    edge_index = torch.stack([src, dst], dim=1)     # (B, 2, N + N²)
    return edge_index, etype
