"""Tests for the reachability-tube lookup: loading, interpolation, out-of-table fallbacks."""

from __future__ import annotations

import importlib.resources
import pathlib

import numpy as np
import pytest

from rosnav_rl.observations.utils.reachability_tube import ReachabilityTube, straight_line_closest_approach

DATA_DIR = pathlib.Path(str(importlib.resources.files("rosnav_rl") / "observations" / "utils" / "data"))
JACKAL_NPZ = DATA_DIR / "jackal_brt_v1.npz"
JACKAL_JSON = DATA_DIR / "jackal_brt_v1.json"


@pytest.fixture
def tube() -> ReachabilityTube:
    return ReachabilityTube.from_npz(JACKAL_NPZ, JACKAL_JSON)


def test_from_npz_loads_the_jackal_table_metadata(tube: ReachabilityTube):
    assert tube.robot_rated_speed_mps == pytest.approx(0.5)
    assert tube.robot_max_yaw_rate_radps == pytest.approx(1.5)
    assert tube.horizon_s == pytest.approx(3.0)


def test_from_npz_rejects_a_tampered_array(tmp_path: pathlib.Path):
    npz = tmp_path / "jackal_brt_v1.npz"
    npz.write_bytes(JACKAL_NPZ.read_bytes() + b"\x00")
    json_path = tmp_path / "jackal_brt_v1.json"
    json_path.write_text(JACKAL_JSON.read_text())
    with pytest.raises(ValueError, match="sha256"):
        ReachabilityTube.from_npz(npz, json_path)


def test_matches_robot_accepts_the_generating_params_and_rejects_others(tube: ReachabilityTube):
    assert tube.matches_robot(0.5, 1.5)
    assert not tube.matches_robot(0.6, 1.5)
    assert not tube.matches_robot(0.5, 1.0)


def test_query_at_a_grid_node_matches_the_dequantized_stored_value(tube: ReachabilityTube):
    raw = np.load(JACKAL_NPZ)
    vi, xi, yi, ti = 5, 60, 40, 10
    expected = raw["values"][vi, xi, yi, ti] / np.iinfo(np.uint16).max * tube.oob_cap
    got = tube.query_batch(
        np.array([raw["x"][xi]]),
        np.array([raw["y"][yi]]),
        np.array([raw["theta"][ti]]),
        np.array([raw["pedestrian_speed"][vi]]),
    )
    np.testing.assert_allclose(got, [expected], atol=1e-5)


def test_a_state_far_outside_the_box_reads_the_saturation_cap(tube: ReachabilityTube):
    got = tube.query_batch(np.array([100.0]), np.array([100.0]), np.array([0.0]), np.array([1.0]))
    np.testing.assert_allclose(got, [tube.oob_cap])


def test_a_pedestrian_speed_beyond_the_table_falls_back_to_the_straight_line_bound_not_a_false_safe_cap(
    tube: ReachabilityTube,
):
    value = tube.value(np.array([0.3]), np.array([0.0]), np.array([np.pi]), np.array([5.0]), contact_radius=0.567)
    assert value[0] < 0


def test_the_periodic_theta_axis_is_continuous_across_the_seam(tube: ReachabilityTube):
    near_minus_pi = tube.query_batch(np.array([1.0]), np.array([0.5]), np.array([-np.pi + 1e-3]), np.array([1.0]))
    near_plus_pi = tube.query_batch(np.array([1.0]), np.array([0.5]), np.array([np.pi - 1e-3]), np.array([1.0]))
    np.testing.assert_allclose(near_minus_pi, near_plus_pi, atol=0.02)


def test_a_pedestrian_closer_ahead_reads_less_safe_than_one_further_ahead(tube: ReachabilityTube):
    close = tube.value(np.array([0.3]), np.array([0.0]), np.array([np.pi]), np.array([0.0]), contact_radius=0.567)
    far = tube.value(np.array([3.5]), np.array([0.0]), np.array([np.pi]), np.array([0.0]), contact_radius=0.567)
    assert close[0] < far[0]


def test_query_batch_handles_n_zero():
    tube = ReachabilityTube.from_npz(JACKAL_NPZ, JACKAL_JSON)
    got = tube.query_batch(np.array([]), np.array([]), np.array([]), np.array([]))
    assert got.shape == (0,)


def test_query_batch_rejects_mismatched_shapes(tube: ReachabilityTube):
    with pytest.raises(ValueError, match="share one shape"):
        tube.query_batch(np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_straight_line_closest_approach_matches_a_hand_computed_head_on_case():
    # theta=0 points the robot's velocity (v_r*cos(theta), v_r*sin(theta)) at +x, i.e. straight at
    # a pedestrian standing at x=5; reaching it exactly at t=horizon=x/v_r=10.
    distance = straight_line_closest_approach(
        np.array([5.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), v_r=0.5, horizon=10.0
    )
    np.testing.assert_allclose(distance, [0.0], atol=1e-6)


def test_straight_line_closest_approach_reads_the_initial_distance_when_moving_away():
    distance = straight_line_closest_approach(
        np.array([5.0]), np.array([0.0]), np.array([np.pi]), np.array([0.0]), v_r=0.5, horizon=10.0
    )
    np.testing.assert_allclose(distance, [5.0], atol=1e-6)


def test_reachability_tube_module_does_not_import_jax():
    import sys

    assert "jax" not in sys.modules
    assert "hj_reachability" not in sys.modules
