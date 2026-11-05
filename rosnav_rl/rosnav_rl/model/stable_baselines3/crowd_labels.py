"""Training labels for the crowd predictor, built from the pedestrian states recorded in a rollout buffer."""

from __future__ import annotations

import numpy as np

HORIZON = 10
DT = 0.1




def build_labels(crowd: np.ndarray, pose: np.ndarray, episode_starts: np.ndarray, horizon: int = HORIZON, dt: float = DT):
    """crowd (T, N, K, 6) = [x, y, vx, vy, mask, id] in the robot frame of its own step, pose (T, N, 3) = [x, y, yaw] in
    the map, episode_starts (T, N). Returns target (T, N, K, 4), valid (T, N, K) and robot_vel (T, N, 2).

    The target of a pedestrian is its [x, y, vx, vy] `horizon` steps later, matched by id, in the robot frame of the
    earlier step. A step has no label when the window leaves the buffer or crosses an episode start.
    """
    steps, envs, k, _ = crowd.shape
    target = np.zeros((steps, envs, k, 4), dtype=np.float32)
    valid = np.zeros((steps, envs, k), dtype=bool)
    robot_vel = np.zeros((steps, envs, 2), dtype=np.float32)

    for t in range(steps - horizon):
        t2 = t + horizon
        crossed = episode_starts[t + 1 : t2 + 1].any(axis=0)  # (N,)
        for n in range(envs):
            if crossed[n]:
                continue
            x0, y0, yaw0 = pose[t, n]
            x1, y1, yaw1 = pose[t2, n]
            c0, s0 = np.cos(yaw0), np.sin(yaw0)
            dyaw = yaw1 - yaw0
            cd, sd = np.cos(dyaw), np.sin(dyaw)
            # robot-frame-of-t2 -> map -> robot-frame-of-t
            dx, dy = x1 - x0, y1 - y0
            tx, ty = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
            robot_vel[t, n] = (tx / (horizon * dt), ty / (horizon * dt))

            ids_late = {int(r[5]): r for r in crowd[t2, n] if r[4] > 0.5}
            for j in range(k):
                row = crowd[t, n, j]
                if row[4] < 0.5:
                    continue
                late = ids_late.get(int(row[5]))
                if late is None:
                    continue
                px = cd * late[0] - sd * late[1] + tx
                py = sd * late[0] + cd * late[1] + ty
                vx = cd * late[2] - sd * late[3]
                vy = sd * late[2] + cd * late[3]
                target[t, n, j] = (px, py, vx, vy)
                valid[t, n, j] = True
    return target, valid, robot_vel


def map_targets_to_feature_crowd(
    feature_crowd: np.ndarray,
    raw_crowd: np.ndarray,
    raw_target: np.ndarray,
    raw_valid: np.ndarray,
    *,
    roi_in_m: float,
    map_height: int,
    map_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer ID-matched raw-track targets onto AGENT_5's map-reconstructed set.

    Args:
        feature_crowd: ``(T,N,K,5)`` physical ``[x,y,vx,vy,mask]`` values reconstructed
            from the *same feature maps used at inference*.
        raw_crowd: ``(T,N,K_raw,6)`` raw ``[x,y,vx,vy,mask,id]`` tracks from transition info.
        raw_target/raw_valid: outputs from :func:`build_labels`, indexed by raw track slot.

    The feature maps quantise positions into cells and may collapse multiple pedestrians in
    a cell. We first match by the same grid cell, then by velocity (the velocity maps were
    written from the same pedestrian stream). Unmatched/ambiguous missing tracks are masked
    rather than assigned a fabricated target.
    """
    feature_crowd = np.asarray(feature_crowd, dtype=np.float32)
    raw_crowd = np.asarray(raw_crowd, dtype=np.float32)
    raw_target = np.asarray(raw_target, dtype=np.float32)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    if feature_crowd.ndim != 4 or feature_crowd.shape[-1] != 5:
        raise ValueError(f"feature_crowd must have shape (T,N,K,5), got {feature_crowd.shape}")
    if raw_crowd.ndim != 4 or raw_crowd.shape[-1] < 6:
        raise ValueError(f"raw_crowd must have shape (T,N,K_raw,6), got {raw_crowd.shape}")
    expected = raw_crowd.shape[:3]
    if raw_target.shape != (*expected, 4) or raw_valid.shape != expected:
        raise ValueError("raw target/mask shapes do not match raw crowd tracks")
    if feature_crowd.shape[:2] != raw_crowd.shape[:2]:
        raise ValueError("feature_crowd and raw_crowd must share time/environment dimensions")
    if not np.isfinite(roi_in_m) or roi_in_m <= 0 or map_height <= 0 or map_width <= 0:
        raise ValueError("ROI and map dimensions must be positive")

    out_target = np.zeros((*feature_crowd.shape[:3], 4), dtype=np.float32)
    out_valid = np.zeros(feature_crowd.shape[:3], dtype=bool)
    cell_x = roi_in_m / map_width
    cell_y = roi_in_m / map_height
    center_x = map_width // 2
    center_y = map_height // 2

    for t in range(feature_crowd.shape[0]):
        for n in range(feature_crowd.shape[1]):
            raw = raw_crowd[t, n]
            raw_live = np.flatnonzero(raw[:, 4] > 0.5)
            if raw_live.size == 0:
                continue
            # Use exactly the same floor-and-center convention as BaseFeatureMapSpace.
            raw_ix = np.floor(raw[raw_live, 0] / cell_x + center_x).astype(np.int64)
            raw_iy = np.floor(raw[raw_live, 1] / cell_y + center_y).astype(np.int64)
            used: set[int] = set()
            for k in np.flatnonzero(feature_crowd[t, n, :, 4] > 0.5):
                row = feature_crowd[t, n, k]
                ix = int(np.floor(row[0] / cell_x + center_x + 1e-7))
                iy = int(np.floor(row[1] / cell_y + center_y + 1e-7))
                candidates = np.flatnonzero((raw_ix == ix) & (raw_iy == iy))
                if candidates.size == 0:
                    continue
                candidates = candidates[[int(raw_live[j]) not in used for j in candidates]]
                if candidates.size == 0:
                    continue
                source_indices = raw_live[candidates]
                vel_error = np.linalg.norm(raw[source_indices, 2:4] - row[2:4], axis=-1)
                # Position is a tie-breaker: row's position is the cell centre, not exact track x/y.
                pos_error = np.linalg.norm(raw[source_indices, :2] - row[:2], axis=-1)
                chosen = int(source_indices[np.argmin(vel_error + 1e-4 * pos_error)])
                used.add(chosen)
                if raw_valid[t, n, chosen]:
                    out_target[t, n, k] = raw_target[t, n, chosen]
                    out_valid[t, n, k] = True
    return out_target, out_valid
