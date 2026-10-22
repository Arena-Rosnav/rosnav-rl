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
