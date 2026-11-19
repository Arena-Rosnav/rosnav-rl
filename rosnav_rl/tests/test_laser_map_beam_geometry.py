import numpy as np
import pytest

from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import (
    LaserCartesianMapSpace,
    StackedLaserMapSpace,
)

JACKAL_BEAMS = 720
JACKAL_INCREMENT = 2 * np.pi / JACKAL_BEAMS


def _jackal_scan() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.2, 30.0, JACKAL_BEAMS).astype(np.float32)


def _reference_stacked_frame(scan: np.ndarray, stack: int, cols: int, max_range: float) -> np.ndarray:
    per_col = len(scan) // cols
    columns = scan[: cols * per_col].reshape(cols, per_col)
    flat = np.empty(2 * cols, dtype=np.float32)
    flat[:cols] = columns.min(axis=1)
    flat[cols:] = columns.mean(axis=1)
    return flat * np.float32(2.0 / max_range) - 1.0


def _cartesian_reference(scan, angles, size=80, roi=20.0, max_range=30.0):
    grid = np.zeros((size, size), dtype=np.float32)
    hit = scan < max_range * 0.99
    x = np.cos(angles[hit]) * scan[hit]
    y = np.sin(angles[hit]) * scan[hit]
    gx = np.floor(x * size / roi + size // 2).astype(int)
    gy = np.floor(y * size / roi + size // 2).astype(int)
    ok = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
    grid[gy[ok], gx[ok]] = 1.0 - scan[hit][ok] / max_range
    return grid


@pytest.mark.parametrize("encoding", [1, 2])
def test_stacked_laser_map_keeps_the_jackal_720_beam_encoding(encoding):
    space = StackedLaserMapSpace(
        laser_stack_size=1, feature_map_size=80, laser_max_range=30.0, laser_num_beams=720, encoding_version=encoding
    )
    scan = _jackal_scan()

    result = space.encode_observation(scan, False)

    expected = np.tile(_reference_stacked_frame(scan, 1, 80, 30.0), 40)
    np.testing.assert_array_equal(result.ravel(), expected)


def test_stacked_laser_map_encoding_1_drops_the_beams_past_the_last_full_column():
    space = StackedLaserMapSpace(laser_stack_size=1, feature_map_size=80, laser_max_range=10.0, laser_num_beams=541)
    scan = np.full(541, 5.0, dtype=np.float32)

    reference = space.encode_observation(scan, False).ravel()[:160]
    scan[540] = 1.0

    np.testing.assert_array_equal(space.encode_observation(scan, False).ravel()[:160], reference)
    np.testing.assert_array_equal(reference, _reference_stacked_frame(scan, 1, 80, 10.0))


def test_stacked_laser_map_bins_the_beams_past_the_last_full_column():
    space = StackedLaserMapSpace(
        laser_stack_size=1, feature_map_size=80, laser_max_range=10.0, laser_num_beams=541, encoding_version=2
    )
    scan = np.full(541, 5.0, dtype=np.float32)
    scan[540] = 1.0

    minimums = space.encode_observation(scan, False).ravel()[:80]

    expected = np.zeros(80, dtype=np.float32)
    expected[79] = 1.0 * 0.2 - 1.0
    np.testing.assert_allclose(minimums, expected, atol=1e-6)


def test_stacked_laser_map_spreads_odd_beam_counts_over_every_column():
    space = StackedLaserMapSpace(
        laser_stack_size=1, feature_map_size=80, laser_max_range=10.0, laser_num_beams=541, encoding_version=2
    )

    for beam in (0, 270, 540):
        scan = np.full(541, 5.0, dtype=np.float32)
        scan[beam] = 1.0
        minimums = space.encode_observation(scan, False).ravel()[:80]
        assert np.count_nonzero(minimums) == 1
        assert np.flatnonzero(minimums)[0] == np.searchsorted(np.arange(81) * 541 // 80, beam, side="right") - 1


def test_stacked_laser_map_refuses_fewer_beams_than_columns():
    with pytest.raises(ValueError, match="cannot fill"):
        StackedLaserMapSpace(laser_stack_size=1, feature_map_size=80, laser_num_beams=40, encoding_version=2)


@pytest.mark.parametrize("encoding", [1, 2])
def test_laser_cartesian_map_keeps_the_jackal_full_circle_angles(encoding):
    scan = _jackal_scan()
    scan[::3] = 30.0
    legacy_angles = np.linspace(-np.pi, np.pi, JACKAL_BEAMS, endpoint=False, dtype=np.float32)

    default = LaserCartesianMapSpace(laser_num_beams=JACKAL_BEAMS, encoding_version=encoding).encode_observation(scan)
    explicit = LaserCartesianMapSpace(
        laser_num_beams=JACKAL_BEAMS, laser_angle_increment=JACKAL_INCREMENT, encoding_version=encoding
    ).encode_observation(scan)

    np.testing.assert_array_equal(default, _cartesian_reference(scan, legacy_angles))
    np.testing.assert_array_equal(explicit, default)


def test_laser_cartesian_map_encoding_1_spreads_any_scan_over_a_full_circle():
    kwargs = {"laser_num_beams": 541, "laser_angle_min": -0.75 * np.pi, "laser_max_range": 30.0}
    scan = _jackal_scan()[:541]

    with_increment = LaserCartesianMapSpace(**kwargs, laser_angle_increment=np.deg2rad(0.5)).encode_observation(scan)
    without = LaserCartesianMapSpace(**kwargs).encode_observation(scan)

    np.testing.assert_array_equal(with_increment, without)


def test_laser_cartesian_map_places_the_last_beam_of_a_270_degree_scan():
    space = LaserCartesianMapSpace(
        laser_num_beams=541,
        laser_angle_min=-0.75 * np.pi,
        laser_angle_increment=np.deg2rad(0.5),
        laser_max_range=30.0,
        encoding_version=2,
    )
    scan = np.full(541, 30.0, dtype=np.float32)
    scan[540] = 2.0

    result = space.encode_observation(scan)

    rows, cols = np.nonzero(result)
    assert list(zip(rows, cols, strict=True)) == [(45, 34)]
    assert result[45, 34] == pytest.approx(1.0 - 2.0 / 30.0, rel=1e-6)
