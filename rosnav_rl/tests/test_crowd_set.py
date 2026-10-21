"""Tests for the padded crowd set: generator, space and mirror rule."""

from __future__ import annotations

from pathlib import Path

import arena_people_msgs.msg as m
import numpy as np
import yaml

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.data_sources.generators import CrowdSetGenerator
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdSetSpace
from rosnav_rl.spaces.space_manager.mirror_space_manager import _negate

PARAMS = AgentParameters(robot_radius=0.4)
POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0}


def _ped(i, x, y, vx, vy):
    p = m.Pedestrian()
    p.id = i
    p.pose.position.x = x
    p.pose.position.y = y
    p.twist.linear.x = vx
    p.twist.linear.y = vy
    return p


def _msg(*peds):
    msg = m.Pedestrians()
    msg.pedestrians = list(peds)
    return msg


def _gen(msg, pose=POSE, **kwargs):
    return CrowdSetGenerator(name="crowd_set", **kwargs)._generate(pose, msg, PARAMS)


def test_rows_are_ordered_by_id_padded_and_out_of_range_dropped():
    out = _gen(_msg(_ped(7, 3, 0, -1, 0), _ped(2, 1, 1, 0, 0.5), _ped(9, 50, 0, 0, 0)), max_pedestrians=4)

    assert out.shape == (4, 5)
    np.testing.assert_allclose(out[0], [1, 1, 0, 0.5, 1])
    np.testing.assert_allclose(out[1], [3, 0, -1, 0, 1])
    assert not out[2:].any()


def test_empty_crowd_is_all_padding():
    assert not _gen(m.Pedestrians(), max_pedestrians=4).any()


def test_order_does_not_depend_on_message_order():
    a = _gen(_msg(_ped(1, 1, 0, 0, 0), _ped(2, 2, 0, 0, 0)), max_pedestrians=3)
    b = _gen(_msg(_ped(2, 2, 0, 0, 0), _ped(1, 1, 0, 0, 0)), max_pedestrians=3)

    np.testing.assert_allclose(a, b)


def test_more_than_k_keeps_the_nearest_then_orders_by_id():
    peds = [_ped(i, float(i), 0, 0, 0) for i in (5, 4, 3, 2, 1)]
    out = _gen(_msg(*peds), max_pedestrians=3)

    np.testing.assert_allclose(out[:, 0], [1, 2, 3])


def test_robot_yaw_rotates_positions_and_velocities():
    out = _gen(_msg(_ped(1, 1, 1, 0, 0.5)), pose={"x": 0.0, "y": 0.0, "yaw": np.pi / 2}, max_pedestrians=2)

    np.testing.assert_allclose(out[0], [1, -1, 0.5, 0, 1], atol=1e-6)


def test_space_clips_to_physical_bounds():
    sp = CrowdSetSpace(max_pedestrians=2)
    x = np.zeros((2, 5), dtype=np.float32)
    x[0] = [20, 1, 5, 0, 1]

    assert sp.space.shape == (2, 5)
    np.testing.assert_allclose(sp.encode_observation(x)[0], [12, 1, 3, 0, 1])


def test_mirror_negates_y_and_vy_only():
    x = np.array([[1.0, 2.0, 0.5, -0.3, 1.0]], dtype=np.float32)

    np.testing.assert_allclose(_negate((1, 3))(x), [[1, -2, 0.5, 0.3, 1]])


def test_datasource_is_registered_in_the_package_yaml():
    path = Path(__file__).resolve().parents[1] / "rosnav_rl" / "observations" / "observations.yaml"

    assert yaml.safe_load(path.read_text())["datasources"]["crowd_set"]["type"] == "CrowdSetGenerator"
