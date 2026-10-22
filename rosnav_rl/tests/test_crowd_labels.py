from __future__ import annotations

import numpy as np

from rosnav_rl.model.stable_baselines3.crowd_labels import build_labels


def _buffers(steps=12):
    crowd = np.zeros((steps, 1, 2, 6), dtype=np.float32)
    pose = np.zeros((steps, 1, 3), dtype=np.float32)
    starts = np.zeros((steps, 1), dtype=np.float32)
    return crowd, pose, starts


def test_static_robot_matches_the_same_pedestrian_by_id():
    crowd, pose, starts = _buffers()
    crowd[0, 0, 0] = [1, 0, 0.5, 0, 1, 7]
    crowd[10, 0, 1] = [1.5, 0, 0.5, 0, 1, 7]  # same id sits in another slot later

    target, valid, vel = build_labels(crowd, pose, starts)

    assert valid[0, 0, 0] and not valid[0, 0, 1]
    np.testing.assert_allclose(target[0, 0, 0], [1.5, 0, 0.5, 0])
    np.testing.assert_allclose(vel[0, 0], [0, 0])


def test_moving_robot_shifts_the_frame_and_gives_its_velocity():
    crowd, pose, starts = _buffers()
    pose[10, 0] = [1.0, 0.0, 0.0]  # robot advanced 1 m in 1 s
    crowd[0, 0, 0] = [3, 0, 0, 0, 1, 1]
    crowd[10, 0, 0] = [2, 0, 0, 0, 1, 1]  # a still pedestrian seen 1 m closer

    target, valid, vel = build_labels(crowd, pose, starts)

    np.testing.assert_allclose(target[0, 0, 0, :2], [3, 0], atol=1e-6)
    np.testing.assert_allclose(vel[0, 0], [1, 0], atol=1e-6)


def test_rotation_is_undone():
    crowd, pose, starts = _buffers()
    pose[10, 0] = [0.0, 0.0, np.pi / 2]
    crowd[0, 0, 0] = [0, 0, 0, 0, 1, 1]
    crowd[10, 0, 0] = [1, 0, 0, 0, 1, 1]  # ahead of the turned robot = +y in the old frame

    target, _, _ = build_labels(crowd, pose, starts)

    np.testing.assert_allclose(target[0, 0, 0, :2], [0, 1], atol=1e-6)


def test_episode_start_inside_the_window_removes_the_label():
    crowd, pose, starts = _buffers()
    crowd[0, 0, 0] = [1, 0, 0, 0, 1, 1]
    crowd[10, 0, 0] = [1, 0, 0, 0, 1, 1]
    starts[5, 0] = 1.0

    _, valid, _ = build_labels(crowd, pose, starts)

    assert not valid[0].any()


def test_steps_near_the_end_have_no_label():
    crowd, pose, starts = _buffers()
    crowd[:, 0, 0] = [1, 0, 0, 0, 1, 1]

    _, valid, _ = build_labels(crowd, pose, starts)

    assert not valid[-10:].any() and valid[0, 0, 0]
