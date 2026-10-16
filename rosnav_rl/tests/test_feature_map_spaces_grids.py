"""Pedestrian and laser feature maps placed on known grid cells."""

import numpy as np
import pytest
from gymnasium import spaces

from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import (
    BaseFeatureMapSpace,
    LaserCartesianMapSpace,
    PedestrianLocationSpace,
    PedestrianSocialStateSpace,
    PedestrianTypeSpace,
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)

LOCATIONS = np.array([[1.0, -0.5], [3.0, 0.0], [-1.2, 2.2]])


class AbstractGymSpaceMap(BaseFeatureMapSpace):
    name = "AbstractGymSpaceMap"

    def get_gym_space(self):
        return super().get_gym_space()

    def encode_observation(self, *args, **kwargs):
        return np.zeros(1)


class AbstractEncodeMap(BaseFeatureMapSpace):
    name = "AbstractEncodeMap"

    def get_gym_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=self._get_feature_map_shape())

    def encode_observation(self, *args, **kwargs):
        return super().encode_observation(*args, **kwargs)


def nonzero_cells(feature_map, background=0):
    rows, cols = np.nonzero(feature_map != background)
    return {(int(r), int(c)): feature_map[r, c].item() for r, c in zip(rows, cols, strict=True)}


def test_grid_geometry_is_derived_from_size_and_roi():
    space = PedestrianLocationSpace(feature_map_size=10, roi_in_m=5.0)

    assert space.feature_map_size == 10
    assert (space._grid_resolution, space._grid_center) == (0.5, 5)
    assert space._get_map_index((1.0, -0.5)) == (7, 4)


@pytest.mark.parametrize(
    ("space_class", "kwargs", "key"),
    [
        (PedestrianVelXSpace, {"ped_min_speed_x": -2.0, "ped_max_speed_x": 2.0}, "pedestrian_vel_x"),
        (PedestrianVelYSpace, {"ped_min_speed_y": -2.0, "ped_max_speed_y": 2.0}, "pedestrian_vel_y"),
    ],
)
def test_velocity_maps_write_speeds_into_pedestrian_cells_inside_the_roi(space_class, kwargs, key):
    space = space_class(feature_map_size=10, roi_in_m=5.0, **kwargs)

    feature_map = space.encode_observation(LOCATIONS, np.array([0.7, 1.9, -1.5]))

    assert space.space.shape == (10, 10)
    assert (space.space.low.min(), space.space.high.max()) == (-2.0, 2.0)
    assert nonzero_cells(feature_map) == {(4, 7): pytest.approx(0.7), (9, 2): pytest.approx(-1.5)}
    assert key in space.requires


@pytest.mark.parametrize(
    "space",
    [
        PedestrianVelXSpace(-1.0, 1.0, feature_map_size=10, roi_in_m=5.0, flatten=True),
        PedestrianVelYSpace(-1.0, 1.0, feature_map_size=10, roi_in_m=5.0, flatten=True),
    ],
)
def test_flattened_velocity_maps_are_row_major_vectors(space):
    feature_map = space.encode_observation(LOCATIONS[:1], np.array([0.5]))

    assert space.space.shape == (100,)
    assert feature_map.shape == (100,)
    assert np.flatnonzero(feature_map).tolist() == [47]


@pytest.mark.parametrize("space_class", [PedestrianVelXSpace, PedestrianVelYSpace])
def test_velocity_maps_are_empty_without_pedestrians(space_class):
    space = space_class(-1.0, 1.0, feature_map_size=6, roi_in_m=3.0)

    assert not space.encode_observation(np.empty((0, 2)), np.array([])).any()


def test_location_map_marks_occupied_cells_and_clears_between_steps():
    space = PedestrianLocationSpace(feature_map_size=10, roi_in_m=5.0)

    occupied = nonzero_cells(space.encode_observation(LOCATIONS))
    empty = space.encode_observation(np.empty((0, 2)))

    assert occupied == {(4, 7): 1.0, (9, 2): 1.0}
    assert not empty.any()
    assert space.space.shape == (10, 10)


def test_flattened_location_map_matches_the_space_shape():
    space = PedestrianLocationSpace(feature_map_size=10, roi_in_m=5.0, flatten=True)

    assert space.encode_observation(LOCATIONS).shape == space.space.shape == (100,)


def test_location_just_outside_the_negative_roi_edge_is_dropped():
    space = PedestrianLocationSpace(feature_map_size=10, roi_in_m=5.0)

    assert not space.encode_observation(np.array([[-2.6, 0.0], [0.0, -2.7]])).any()


def test_social_state_map_writes_states_as_integers():
    space = PedestrianSocialStateSpace(ped_social_state_num=4, feature_map_size=10, roi_in_m=5.0)

    feature_map = space.encode_observation(LOCATIONS, np.array([2, 3, 4]))

    assert feature_map.dtype == np.int32
    assert nonzero_cells(feature_map) == {(4, 7): 2, (9, 2): 4}
    assert (space.space.low.min(), space.space.high.max(), space.space.dtype) == (0, 4, np.int32)
    assert not space.encode_observation(np.empty((0, 2)), np.array([])).any()


def test_flattened_social_state_map():
    space = PedestrianSocialStateSpace(ped_social_state_num=4, feature_map_size=10, roi_in_m=5.0, flatten=True)

    assert space.encode_observation(LOCATIONS, np.array([2, 3, 4])).shape == (100,)


def test_type_map_uses_minus_one_as_background():
    space = PedestrianTypeSpace(ped_num_types=3, feature_map_size=10, roi_in_m=5.0)

    feature_map = space.encode_observation(LOCATIONS, np.array([0, 1, 2]))

    assert nonzero_cells(feature_map, background=-1) == {(4, 7): 0, (9, 2): 2}
    assert (space.space.low.min(), space.space.high.max()) == (-1, 3)
    assert (space.encode_observation(np.empty((0, 2)), np.array([])) == -1).all()


def test_flattened_type_map():
    space = PedestrianTypeSpace(ped_num_types=3, feature_map_size=10, roi_in_m=5.0, flatten=True)

    feature_map = space.encode_observation(LOCATIONS, np.array([0, 1, 2]))

    assert feature_map.shape == (100,)
    assert feature_map[47] == 0


def test_laser_cartesian_map_marks_hits_with_proximity_and_skips_max_range_and_far_hits():
    space = LaserCartesianMapSpace(
        laser_num_beams=4, laser_max_range=10.0, laser_angle_min=np.pi / 4, feature_map_size=10, roi_in_m=5.0
    )
    diagonal = np.hypot(1.25, 1.25)

    expected = pytest.approx(1.0 - diagonal / 10.0, rel=1e-6)

    assert nonzero_cells(space.encode_observation(np.array([diagonal, 5.0, np.nan, diagonal]))) == {
        (7, 7): expected,
        (2, 7): expected,
    }
    assert not space.encode_observation([np.inf, 10.0, 9.95, 30.0]).any()


def test_laser_cartesian_map_clips_negative_ranges_to_the_robot_cell():
    space = LaserCartesianMapSpace(
        laser_num_beams=4, laser_max_range=10.0, laser_angle_min=np.pi / 4, feature_map_size=10, roi_in_m=5.0
    )

    assert nonzero_cells(space.encode_observation(np.array([-1.0, 30.0, 30.0, 30.0]))) == {(5, 5): 1.0}


@pytest.mark.parametrize(
    ("space", "first", "second"),
    [
        (PedestrianLocationSpace(feature_map_size=10, roi_in_m=5.0), LOCATIONS, np.empty((0, 2))),
        (
            LaserCartesianMapSpace(laser_num_beams=4, laser_max_range=10.0, feature_map_size=10, roi_in_m=5.0),
            np.full(4, 1.0),
            np.full(4, 30.0),
        ),
    ],
)
def test_returned_map_survives_the_next_encode(space, first, second):
    earlier = space.encode_observation(first)
    snapshot = earlier.copy()

    space.encode_observation(second)

    np.testing.assert_array_equal(earlier, snapshot)


def test_flattened_laser_cartesian_map():
    space = LaserCartesianMapSpace(laser_num_beams=4, feature_map_size=10, roi_in_m=5.0, flatten=True)

    assert space.encode_observation(np.full(4, 30.0)).shape == space.space.shape == (100,)
    assert (space.space.low.min(), space.space.high.max()) == (0.0, 1.0)


def stacked(**kwargs):
    params = {"laser_stack_size": 2, "feature_map_size": 4, "laser_max_range": 10.0, "laser_num_beams": 8}
    params.update(kwargs)
    return StackedLaserMapSpace(**params)


SCAN = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0], dtype=np.float32)


def test_stacked_laser_map_summarizes_min_and_mean_per_column_oldest_first():
    space = stacked()

    first = space.encode_observation(SCAN, False)
    second = space.encode_observation(np.full(8, 10.0, dtype=np.float32), False)

    assert first.shape == space.space.shape == (1, 4, 4)
    np.testing.assert_allclose(first[0, :2], -1.0)
    np.testing.assert_allclose(first[0, 2:], [[-1.0, -0.2, 0.6, -0.8], [-0.8, 0.0, 0.8, -0.6]], atol=1e-6)
    np.testing.assert_allclose(second[0, :2], [[-1.0, -0.2, 0.6, -0.8], [-0.8, 0.0, 0.8, -0.6]], atol=1e-6)
    np.testing.assert_allclose(second[0, 2:], 1.0)


def test_stacked_laser_map_terminal_step_clears_the_history():
    space = stacked()
    space.encode_observation(np.full(8, 10.0, dtype=np.float32), False)

    restarted = space.encode_observation(SCAN, True)

    np.testing.assert_allclose(restarted, stacked().encode_observation(SCAN, False))


def test_stacked_laser_map_reset_clears_the_history():
    space = stacked()
    space.encode_observation(np.full(8, 10.0, dtype=np.float32), False)

    space.reset()

    np.testing.assert_allclose(space.encode_observation(SCAN, False), stacked().encode_observation(SCAN, False))


def test_stacked_laser_map_reallocates_for_a_new_beam_count():
    space = stacked()
    space.encode_observation(SCAN, False)

    result = space.encode_observation(np.arange(12, dtype=np.float64), False)

    assert space._ring_buffer.shape == (2, 12)
    assert space._beams_per_col == 3
    np.testing.assert_allclose(result[0, 3], [-0.8, -0.2, 0.4, 1.0], atol=1e-6)


@pytest.mark.parametrize("scan", [np.array([], dtype=np.float32), [1.0] * 8])
def test_stacked_laser_map_treats_empty_or_non_array_scans_as_zero_range(scan):
    np.testing.assert_allclose(stacked().encode_observation(scan, False), -1.0)


def test_stacked_laser_map_tiles_a_short_summary_to_fill_the_map():
    space = stacked(laser_stack_size=1, flatten=True)

    result = space.encode_observation(SCAN, False)

    assert result.shape == space.space.shape == (16,)
    np.testing.assert_allclose(result[:8], result[8:])
    np.testing.assert_allclose(result[:8], [-1.0, -0.2, 0.6, -0.8, -0.8, 0.0, 0.8, -0.6], atol=1e-6)


def test_feature_map_base_gym_space_is_abstract():
    with pytest.raises(NotImplementedError):
        AbstractGymSpaceMap(feature_map_size=4, roi_in_m=2.0)


def test_feature_map_base_encode_is_abstract():
    space = AbstractEncodeMap(feature_map_size=4, roi_in_m=2.0, flatten=True)

    assert space.space.shape == (16,)
    with pytest.raises(NotImplementedError):
        space.encode_observation()
