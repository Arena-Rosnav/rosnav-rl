from __future__ import annotations

import numpy as np

from rosnav_rl.model.stable_baselines3.crowd_labels import build_labels, map_targets_to_feature_crowd


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


def test_map_targets_follow_the_feature_map_track_not_raw_slot_order():
    # 24 m ROI / 80 cells gives 0.3 m cells. x=1.45 maps to centre 1.35.
    raw = np.zeros((12, 1, 3, 6), dtype=np.float32)
    raw[:, 0, 0] = [1.45, 0.15, 0.5, 0.0, 1.0, 7.0]
    raw[:, 0, 1] = [1.48, 0.15, -0.25, 0.0, 1.0, 8.0]  # same cell, but a different velocity
    target = np.zeros((12, 1, 3, 4), dtype=np.float32)
    valid = np.zeros((12, 1, 3), dtype=bool)
    target[0, 0, 0] = [1.95, 0.15, 0.5, 0.0]
    target[0, 0, 1] = [1.25, 0.15, -0.25, 0.0]
    valid[0, 0, :2] = True
    feature = np.zeros((12, 1, 2, 5), dtype=np.float32)
    feature[..., 4] = 0
    feature[0, 0, 0] = [1.35, 0.15, -0.25, 0.0, 1.0]  # matches track ID 8 by velocity
    mapped_target, mapped_valid = map_targets_to_feature_crowd(
        feature, raw, target, valid, roi_in_m=24.0, map_height=80, map_width=80
    )
    assert mapped_valid[0, 0, 0]
    np.testing.assert_allclose(mapped_target[0, 0, 0], target[0, 0, 1])
    assert not mapped_valid[0, 0, 1]
