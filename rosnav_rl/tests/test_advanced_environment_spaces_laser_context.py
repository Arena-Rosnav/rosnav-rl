"""Environment context, spatial awareness and obstacle proximity from constructed laser scans."""

import numpy as np
import pytest

from rosnav_rl.spaces.observation_space.spaces.environment.advanced_environment_spaces import (
    EnvironmentContextSpace,
    ObstacleProximitySpace,
    SpatialAwarenessSpace,
)


def open_scan(**ranges):
    scan = np.full(360, 20.0)
    for index, value in ranges.items():
        scan[int(index[1:])] = value
    return scan


def test_environment_context_dims_follow_sector_count():
    space = EnvironmentContextSpace(density_sectors=5).space

    assert space.shape == (8,)
    assert (space.low.min(), space.high.max()) == (0.0, 1.0)


def test_environment_context_sector_densities_and_clearances():
    space = EnvironmentContextSpace(max_range=10.0, density_sectors=4, obstacle_threshold=0.5)
    scan = np.array([0.2, 3.0, 12.0, 0.0, 0.4, 0.3, 5.0, 10.0])

    np.testing.assert_allclose(space.encode_observation(scan), [0.5, 0.0, 1.0, 0.0, 0.375, 0.02, 0.5], atol=1e-6)


def test_environment_context_last_sector_takes_the_remainder_beams():
    space = EnvironmentContextSpace(max_range=10.0, density_sectors=4, obstacle_threshold=0.5)
    scan = np.array([5.0] * 8 + [0.1])

    result = space.encode_observation(scan)

    np.testing.assert_allclose(result[:5], [0.0, 0.0, 0.0, 1 / 3, 1 / 12], atol=1e-6)


def test_environment_context_without_valid_readings_reports_full_clearance():
    space = EnvironmentContextSpace(max_range=10.0, density_sectors=2)

    np.testing.assert_array_equal(space.encode_observation(np.array([0.0, 10.0, 20.0])), [0, 0, 0, 1, 1])


def test_spatial_awareness_is_a_seven_dim_unit_box():
    assert SpatialAwarenessSpace().space.shape == (7,)


def test_spatial_awareness_right_leaning_corridor_and_clearances():
    space = SpatialAwarenessSpace(max_range=10.0, safety_distance=1.0, corridor_width_threshold=1.5)

    result = space.encode_observation(open_scan(i270=0.8, i90=1.2))

    np.testing.assert_allclose(result, [1.0, 0.8, 1.0, 1.0, 1.0, 0.25, 1 - (0.8 * 0.6 + 0.95 * 0.4)], atol=1e-6)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2.0, 1.0, (1.0, 0.75)), (1.0, 1.0, (1.0, 0.5)), (0.7, 0.7, (0.0, 0.5)), (4.0, 4.0, (0.0, 0.5))],
)
def test_spatial_awareness_corridor_width_window_and_direction(left, right, expected):
    space = SpatialAwarenessSpace(max_range=10.0, corridor_width_threshold=1.5)

    result = space.encode_observation(open_scan(i260=left, i100=right))

    assert tuple(result[4:6]) == pytest.approx(expected)


def test_spatial_awareness_needs_both_walls_for_a_corridor():
    space = SpatialAwarenessSpace(max_range=10.0)

    assert tuple(space.encode_observation(open_scan(i270=1.0))[4:6]) == (0.0, 0.5)


@pytest.mark.parametrize("index", [0, 5, 345, 359])
def test_spatial_awareness_back_sector_wraps_around_the_scan_ends(index):
    space = SpatialAwarenessSpace(max_range=10.0, safety_distance=1.0)

    result = space.encode_observation(open_scan(**{f"i{index}": 0.3}))

    assert result[3] == pytest.approx(0.3)
    assert result[0] == 1.0


def test_spatial_awareness_front_sector_is_centered_on_the_middle_beam():
    space = SpatialAwarenessSpace(max_range=10.0, safety_distance=2.0)

    result = space.encode_observation(open_scan(i180=0.5))

    np.testing.assert_allclose(result[:4], [0.25, 1.0, 1.0, 1.0])


def test_spatial_awareness_sector_indices_wrap():
    indices = SpatialAwarenessSpace()._get_sector_indices(8, 150, 210)

    assert indices.tolist() == [7, 0]


def test_obstacle_proximity_zone_occupancy_closest_and_risk():
    space = ObstacleProximitySpace(proximity_zones=[0.5, 1.5, 3.0], max_range=10.0, risk_threshold=1.0)

    result = space.encode_observation(np.array([0.4, 1.0, 2.0, 12.0]))

    np.testing.assert_allclose(result, [0.25, 0.5, 0.75, 0.04, 0.6], atol=1e-6)


def test_obstacle_proximity_far_obstacles_carry_no_risk():
    space = ObstacleProximitySpace(max_range=10.0, risk_threshold=1.0)

    np.testing.assert_allclose(space.encode_observation(np.array([5.0, 6.0])), [0, 0, 0, 0.5, 0], atol=1e-6)


def test_obstacle_proximity_without_valid_readings_is_clear():
    space = ObstacleProximitySpace(proximity_zones=[1.0], max_range=10.0)

    assert space.space.shape == (3,)
    np.testing.assert_array_equal(space.encode_observation(np.array([0.0, 10.0])), [0.0, 1.0, 0.0])
