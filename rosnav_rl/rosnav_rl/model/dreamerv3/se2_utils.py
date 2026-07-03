"""SE(2)-equivariant frame canonicalization utilities for Social-Dreamer.

Frame Averaging Approach
------------------------
The decoded pedestrian positions produced by the ped-decoder head live implicitly
in the **anchor** robot body frame (the frame of the last ingested real observation).
The GAT/DALI social modules, however, expect positions in the **current** robot body
frame (frame t+k during imagination, or frame t during training).

Rather than modifying the opaque GRU latent (which is non-geometric, approximate, and
touches the torch.compile'd img_step path), we apply an **exact SE(2) correction at
the decoder output level**:

    peds_current = se2_transform_points(se2_inverse(P_k), peds_anchor)

where P_k = accumulated pose of the robot at step k expressed in the anchor frame.

This is exact SE(2) equivariance because the geometric quantity being transformed
(decoded ped positions in R^2) admits a clean SE(2) action — unlike a flat dense
latent vector. The approach is sometimes called "Frame Averaging" or "frame
canonicalization": decode in a canonical (anchor) frame, then rotate out to the
current frame before any geometry-sensitive computation.

Limitation: the kinematic integration in `integrate_se2` uses Euler-arc integration
(not exact arc integration for non-zero omega). Error per step ∝ omega^2 * dt^2,
which is negligible at dt=0.1 s for typical indoor velocities.

Conventions
-----------
- T = (..., 3) tensor = [x, y, theta] — SE(2) transform in homogeneous coordinates
- se2_transform_points(T, p): R(theta) @ p + [x, y] — maps points FROM the T-local
  frame INTO T's parent frame.  Example: T = robot-pose-in-world, p in robot frame →
  result in world frame.
- se2_inverse: standard group inverse, such that
  se2_compose(T, se2_inverse(T)) = identity.
- se2_between(Ta, Tb) = se2_compose(se2_inverse(Ta), Tb) = relative pose of Tb
  expressed in Ta's frame.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core SE(2) group operations
# ---------------------------------------------------------------------------

def se2_compose(T1: Tensor, T2: Tensor) -> Tensor:
    """Compose two SE(2) transforms: apply T1 first, then T2.

    T_composed(p) = T2(T1(p)) = R(θ2) @ (R(θ1) @ p + t1) + t2

    Args:
        T1: (..., 3) = [x1, y1, theta1]
        T2: (..., 3) = [x2, y2, theta2]

    Returns:
        (..., 3) composed transform
    """
    x1, y1, th1 = T1[..., 0], T1[..., 1], T1[..., 2]
    x2, y2, th2 = T2[..., 0], T2[..., 1], T2[..., 2]
    c2 = torch.cos(th2)
    s2 = torch.sin(th2)
    xc = c2 * x1 - s2 * y1 + x2
    yc = s2 * x1 + c2 * y1 + y2
    thc = th1 + th2
    # Wrap to [-pi, pi] to prevent accumulation drift
    thc = torch.atan2(torch.sin(thc), torch.cos(thc))
    return torch.stack([xc, yc, thc], dim=-1)


def se2_inverse(T: Tensor) -> Tensor:
    """SE(2) group inverse.

    If T(p) = R(θ) @ p + t, then T^{-1}(q) = R(-θ) @ (q - t) = R(-θ) @ q - R(-θ) @ t.

    Args:
        T: (..., 3) = [x, y, theta]

    Returns:
        (..., 3) inverse transform
    """
    x, y, th = T[..., 0], T[..., 1], T[..., 2]
    c = torch.cos(th)
    s = torch.sin(th)
    xi = -(c * x + s * y)
    yi = s * x - c * y
    thi = -th
    return torch.stack([xi, yi, thi], dim=-1)


def se2_between(Ta: Tensor, Tb: Tensor) -> Tensor:
    """Relative pose of Tb expressed in Ta's frame.

    se2_between(Ta, Tb) = se2_compose(se2_inverse(Ta), Tb)

    Useful for computing P_k = relative pose of robot at step k from anchor:
        P_k = se2_between(world_pose_anchor, world_pose_k)

    Args:
        Ta: (..., 3) anchor frame pose (in world)
        Tb: (..., 3) target pose (in world)

    Returns:
        (..., 3) pose of Tb expressed in Ta's frame
    """
    return se2_compose(se2_inverse(Ta), Tb)


# ---------------------------------------------------------------------------
# Geometric operations on points / vectors
# ---------------------------------------------------------------------------

def se2_transform_points(T: Tensor, pts: Tensor) -> Tensor:
    """Apply SE(2) transform T to 2D points.

    Maps points FROM T's local frame INTO T's parent frame:
        pts_parent = R(theta) @ pts_local + t

    To transform decoded peds from anchor frame into current frame:
        pts_current = se2_transform_points(se2_inverse(P_k), pts_anchor)

    Args:
        T:   (..., 3) = [x, y, theta]
        pts: (..., N, 2) — 2D points in T's local frame

    Returns:
        (..., N, 2) — points in T's parent frame
    """
    x = T[..., 0:1].unsqueeze(-2)    # (..., 1, 1)
    y = T[..., 1:2].unsqueeze(-2)
    th = T[..., 2:3].unsqueeze(-2)
    c = torch.cos(th)
    s = torch.sin(th)
    px = pts[..., 0:1]  # (..., N, 1)
    py = pts[..., 1:2]
    new_x = c * px - s * py + x
    new_y = s * px + c * py + y
    return torch.cat([new_x, new_y], dim=-1)


def se2_rotate_vectors(T: Tensor, vecs: Tensor) -> Tensor:
    """Rotate 2D vectors by T's theta (no translation — pure rotation).

    Use for velocity vectors (relative velocity is frame-invariant up to rotation).

    Args:
        T:    (..., 3) = [x, y, theta] — only theta is used
        vecs: (..., N, 2)

    Returns:
        (..., N, 2) rotated vectors
    """
    th = T[..., 2:3].unsqueeze(-2)   # (..., 1, 1)
    c = torch.cos(th)
    s = torch.sin(th)
    vx = vecs[..., 0:1]
    vy = vecs[..., 1:2]
    new_vx = c * vx - s * vy
    new_vy = s * vx + c * vy
    return torch.cat([new_vx, new_vy], dim=-1)


# ---------------------------------------------------------------------------
# SE(2) data augmentation (for training)
# ---------------------------------------------------------------------------

def apply_se2_to_peds_flat(
    T: Tensor,
    peds_flat: Tensor,
    max_peds: int,
    node_feat_dim: int,
) -> Tensor:
    """Apply SE(2) transform T to flat ped feature tensor.

    The flat layout is (B, N*(F+1)) where each node = [x, y, vx, vy, ..., validity].
    x,y (dims 0-1) are position — transformed by full SE(2).
    vx,vy (dims 2-3) are velocity — rotated only (no translation).
    Remaining dims (including validity) are left unchanged.

    Args:
        T:          (B, 3) SE(2) transform
        peds_flat:  (B, N*(F+1))
        max_peds:   N
        node_feat_dim: F (must be ≥ 4 for vel support)

    Returns:
        (B, N*(F+1)) transformed
    """
    B = peds_flat.shape[0]
    N, FP1 = max_peds, node_feat_dim + 1
    peds = peds_flat.view(B, N, FP1)  # (B, N, F+1)

    # Position: dims 0-1
    # se2_transform_points expects T: (B, 3) and pts: (B, N, 2); it adds the N-dim
    # broadcast internally via unsqueeze(-2) on T's components.
    xy = peds[..., :2]  # (B, N, 2)
    xy_t = se2_transform_points(T, xy)  # (B, N, 2)

    out = peds.clone()
    out[..., :2] = xy_t

    # Velocity: dims 2-3 (if present)
    if node_feat_dim >= 4:
        vv = peds[..., 2:4]  # (B, N, 2)
        vv_t = se2_rotate_vectors(T, vv)
        out[..., 2:4] = vv_t

    return out.view(B, N * FP1)


def augment_se2(
    batch: dict,
    max_peds: int,
    node_feat_dim: int,
    max_translation: float = 2.0,
    p: float = 0.5,
) -> dict:
    """Apply a random SE(2) augmentation to training batch.

    Transforms:
      - data["PedestrianNodeSetSpace"] x, y positions and vx, vy velocities
      - data["RobotPoseSpace"] world-frame robot poses (so computed P_k stays correct)
    Does NOT transform:
      - StackedLaserMapSpace (already egocentric per scan)
      - DIST_ANGLE_TO_SUBGOAL (already robot-frame, recomputed from pose)
      - is_first, reward, action (frame-invariant)

    Args:
        batch:            {key: Tensor (B, T, ...)} training batch dict
        max_peds:         N
        node_feat_dim:    F
        max_translation:  uniform translation range (meters)
        p:                augmentation probability per batch

    Returns:
        augmented batch (same dict, values may be modified in-place clones)
    """
    if "RobotPoseSpace" not in batch or "PedestrianNodeSetSpace" not in batch:
        return batch

    B, T = batch["RobotPoseSpace"].shape[:2]
    device = batch["RobotPoseSpace"].device

    # Sample per-batch-element mask
    mask = torch.rand(B, device=device) < p  # (B,)
    if not mask.any():
        return batch

    # Sample random SE(2) per batch element
    tx = (torch.rand(B, device=device) * 2 - 1) * max_translation
    ty = (torch.rand(B, device=device) * 2 - 1) * max_translation
    theta = (torch.rand(B, device=device) * 2 - 1) * math.pi
    T_aug = torch.stack([tx, ty, theta], dim=-1)  # (B, 3)

    # Apply to robot poses: new_pose = se2_compose(T_aug, world_pose)
    robot_poses = batch["RobotPoseSpace"].clone()  # (B, T, 3)
    T_aug_exp = T_aug.unsqueeze(1).expand(-1, T, -1)  # (B, T, 3)
    new_robot_poses = se2_compose(T_aug_exp, robot_poses)
    # Apply mask (only augment selected batch elements)
    mask_exp = mask.unsqueeze(1).unsqueeze(2).expand_as(robot_poses)
    robot_poses = torch.where(mask_exp, new_robot_poses, robot_poses)

    # Apply to ped observations (B, T, N*(F+1))
    ped_obs = batch["PedestrianNodeSetSpace"].clone()  # (B, T, N*(F+1))
    mask_bt = mask.unsqueeze(1).expand(-1, T)  # (B, T)
    for t in range(T):
        active = mask_bt[:, t]  # (B,)
        if not active.any():
            continue
        T_t = T_aug  # same transform for all timesteps (global frame aug)
        new_peds_t = apply_se2_to_peds_flat(T_t, ped_obs[:, t], max_peds, node_feat_dim)
        ped_obs[:, t] = torch.where(
            active.unsqueeze(-1).expand_as(ped_obs[:, t]),
            new_peds_t,
            ped_obs[:, t],
        )

    batch = dict(batch)  # shallow copy to avoid mutating caller's dict
    batch["RobotPoseSpace"] = robot_poses
    batch["PedestrianNodeSetSpace"] = ped_obs
    return batch


# ---------------------------------------------------------------------------
# Kinematic integration
# ---------------------------------------------------------------------------

def integrate_se2(
    pose_accum: Tensor,
    action: Tensor,
    action_scale: Tensor,
    dt: float = 0.1,
    holonomic: bool = False,
) -> Tensor:
    """Integrate one control step via Euler arc approximation.

    Denormalizes the policy action from [-1, 1] to physical units using
    action_scale, then integrates diff-drive or omni kinematics.

    Args:
        pose_accum:   (B, 3) accumulated SE(2) pose in anchor frame
        action:       (B, act_dim) policy action in [-1, 1]
        action_scale: (2,) or (3,) = [v_max, omega_max] for diff-drive,
                      [vx_max, vy_max, omega_max] for omni. Physical units.
        dt:           control period (seconds, default 0.1 = 10 Hz)
        holonomic:    if True, treat as omni: [vx, vy, omega]; else diff-drive [v, omega]

    Returns:
        (B, 3) updated pose_accum after one step
    """
    if holonomic:
        # [vx, vy, omega] all scaled from action dims 0,1,2
        vx_n = action[..., 0]
        vy_n = action[..., 1]
        om_n = action[..., 2]
        v_scale = action_scale[0]
        vy_scale = action_scale[1]
        om_scale = action_scale[2]
        vx = vx_n * v_scale
        vy = vy_n * vy_scale
        omega = om_n * om_scale
    else:
        # Differential drive: action[0] = linear, action[-1] = angular
        v = action[..., 0] * action_scale[0]
        omega = action[..., -1] * action_scale[-1]
        vx = v
        vy = torch.zeros_like(v)

    # Delta in body frame of current step: (vx*dt, vy*dt, omega*dt)
    dth = omega * dt
    dx_body = vx * dt
    dy_body = vy * dt
    delta_body = torch.stack([dx_body, dy_body, dth], dim=-1)  # (B, 3)

    # Compose: pose_accum is robot-in-anchor, delta_body is in current body frame
    # se2_compose(pose_accum, delta_body) maps body-frame delta into anchor frame
    return se2_compose(pose_accum, delta_body)
