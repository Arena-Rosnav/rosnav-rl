"""Regression tests for environment/feature_map_spaces.py.

Pins exact numeric output for PedestrianVelXSpace, PedestrianVelYSpace,
PedestrianSocialStateSpace, PedestrianTypeSpace, and StackedLaserMapSpace
before/after the P3.1/P3.2 buffer-reuse refactor (audit 2026-07-04): these
classes historically allocated a fresh np.zeros/np.full array (or used
np.roll/np.tile) on every encode_observation() call. The refactor must not
change any encoded value — only avoid the per-tick allocation.
"""

import numpy as np

from rosnav_rl.spaces.observation_space.spaces.environment.feature_map_spaces import (
    PedestrianSocialStateSpace,
    PedestrianTypeSpace,
    PedestrianVelXSpace,
    PedestrianVelYSpace,
    StackedLaserMapSpace,
)

LOCS = np.array([[1.0, 2.0], [-3.0, 0.5], [10.0, 10.0]], dtype=np.float32)
EMPTY_LOCS = np.zeros((0, 2), dtype=np.float32)


class TestPedestrianVelXSpace:
    def test_encode_matches_pinned_output(self):
        space = PedestrianVelXSpace(
            ped_min_speed_x=-2.0, ped_max_speed_x=2.0, feature_map_size=8, roi_in_m=8.0
        )
        vx = np.array([1.5, -0.5, 99.0], dtype=np.float32)
        out = space.encode_observation(
            pedestrian_relative_locations=LOCS, pedestrian_vel_x=vx
        )

        expected = np.zeros((8, 8), dtype=np.float32)
        expected[4, 1] = -0.5
        expected[6, 5] = 1.5

        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, expected)

    def test_buffer_reuse_does_not_leak_between_calls(self):
        space = PedestrianVelXSpace(
            ped_min_speed_x=-2.0, ped_max_speed_x=2.0, feature_map_size=8, roi_in_m=8.0
        )
        space.encode_observation(
            pedestrian_relative_locations=LOCS,
            pedestrian_vel_x=np.array([1.5, -0.5, 99.0], dtype=np.float32),
        )
        out = space.encode_observation(
            pedestrian_relative_locations=EMPTY_LOCS,
            pedestrian_vel_x=np.zeros((0,), dtype=np.float32),
        )
        np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.float32))


class TestPedestrianVelYSpace:
    def test_encode_matches_pinned_output(self):
        space = PedestrianVelYSpace(
            ped_min_speed_y=-2.0, ped_max_speed_y=2.0, feature_map_size=8, roi_in_m=8.0
        )
        vy = np.array([0.3, -1.1, 55.0], dtype=np.float32)
        out = space.encode_observation(
            pedestrian_relative_locations=LOCS, pedestrian_vel_y=vy
        )

        expected = np.zeros((8, 8), dtype=np.float32)
        expected[4, 1] = np.float32(-1.1)
        expected[6, 5] = np.float32(0.3)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_buffer_reuse_does_not_leak_between_calls(self):
        space = PedestrianVelYSpace(
            ped_min_speed_y=-2.0, ped_max_speed_y=2.0, feature_map_size=8, roi_in_m=8.0
        )
        space.encode_observation(
            pedestrian_relative_locations=LOCS,
            pedestrian_vel_y=np.array([0.3, -1.1, 55.0], dtype=np.float32),
        )
        out = space.encode_observation(
            pedestrian_relative_locations=EMPTY_LOCS,
            pedestrian_vel_y=np.zeros((0,), dtype=np.float32),
        )
        np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.float32))


class TestPedestrianSocialStateSpace:
    def test_encode_matches_pinned_output(self):
        space = PedestrianSocialStateSpace(
            ped_social_state_num=5, feature_map_size=8, roi_in_m=8.0
        )
        ss = np.array([2, 1, 9], dtype=np.int64)
        out = space.encode_observation(
            pedestrian_relative_locations=LOCS, pedestrian_social_states=ss
        )

        expected = np.zeros((8, 8), dtype=np.int32)
        expected[4, 1] = 1
        expected[6, 5] = 2

        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, expected)

    def test_buffer_reuse_does_not_leak_between_calls(self):
        space = PedestrianSocialStateSpace(
            ped_social_state_num=5, feature_map_size=8, roi_in_m=8.0
        )
        space.encode_observation(
            pedestrian_relative_locations=LOCS,
            pedestrian_social_states=np.array([2, 1, 9], dtype=np.int64),
        )
        out = space.encode_observation(
            pedestrian_relative_locations=EMPTY_LOCS,
            pedestrian_social_states=np.zeros((0,), dtype=np.int64),
        )
        np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.int32))


class TestPedestrianTypeSpace:
    def test_encode_matches_pinned_output(self):
        space = PedestrianTypeSpace(ped_num_types=4, feature_map_size=8, roi_in_m=8.0)
        tt = np.array([0, 3, 9], dtype=np.int64)
        out = space.encode_observation(
            pedestrian_relative_locations=LOCS, pedestrian_types=tt
        )

        expected = np.full((8, 8), -1, dtype=np.int32)
        expected[4, 1] = 3
        expected[6, 5] = 0

        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, expected)

    def test_buffer_reuse_does_not_leak_between_calls(self):
        space = PedestrianTypeSpace(ped_num_types=4, feature_map_size=8, roi_in_m=8.0)
        space.encode_observation(
            pedestrian_relative_locations=LOCS,
            pedestrian_types=np.array([0, 3, 9], dtype=np.int64),
        )
        out = space.encode_observation(
            pedestrian_relative_locations=EMPTY_LOCS,
            pedestrian_types=np.zeros((0,), dtype=np.int64),
        )
        # Background value is -1, not 0 — must still be fully reset.
        np.testing.assert_array_equal(out, np.full((8, 8), -1, dtype=np.int32))


class TestStackedLaserMapSpace:
    def _make_space(self):
        return StackedLaserMapSpace(
            laser_stack_size=3,
            feature_map_size=4,
            laser_max_range=10.0,
            laser_num_beams=8,
        )

    def test_encode_matches_pinned_output_over_three_steps(self):
        space = self._make_space()
        scans = [
            np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
            np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32),
            np.array([9, 8, 7, 6, 5, 4, 3, 2], dtype=np.float32),
        ]

        expected = [
            np.full((1, 4, 4), -1.0, dtype=np.float32),
            np.array(
                [[[-1.0, -1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0, -1.0],
                  [-0.8, -0.4, 0.0, 0.4],
                  [-0.7, -0.3, 0.1, 0.5]]],
                dtype=np.float32,
            ),
            np.array(
                [[[-0.8, -0.4, 0.0, 0.4],
                  [-0.7, -0.3, 0.1, 0.5],
                  [-0.6, -0.6, -0.6, -0.6],
                  [-0.6, -0.6, -0.6, -0.6]]],
                dtype=np.float32,
            ),
        ]

        for scan, exp in zip(scans, expected):
            out = space.encode_observation(front_laser=scan, is_terminal=False)
            assert out.dtype == np.float32
            assert out.shape == (1, 4, 4)
            np.testing.assert_allclose(out, exp, rtol=0, atol=1e-6)

    def test_beam_count_change_reallocates_without_crashing(self):
        space = self._make_space()
        space.encode_observation(
            front_laser=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
            is_terminal=False,
        )
        # Different beam count triggers the ring/ordered-buffer reallocation
        # branch (P3.2 refactor touched this).
        out = space.encode_observation(
            front_laser=np.array([1, 2, 3, 4], dtype=np.float32), is_terminal=False
        )
        assert out.shape == (1, 4, 4)
        assert np.all(np.isfinite(out))

    def test_reset_clears_ring_buffer(self):
        space = self._make_space()
        space.encode_observation(
            front_laser=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
            is_terminal=False,
        )
        space.reset()

        out = space.encode_observation(
            front_laser=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
            is_terminal=False,
        )
        np.testing.assert_allclose(
            out, np.full((1, 4, 4), -1.0, dtype=np.float32), rtol=0, atol=1e-6
        )
