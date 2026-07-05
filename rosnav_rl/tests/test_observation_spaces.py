"""Comprehensive tests for observation spaces.

Tests cover:
- LaserScanSpace / ReducedLaserScanSpace
- DistAngleToGoalSpace / DistAngleToSubgoalSpace
- LastActionSpace / SubgoalInRobotFrameSpace
- IsFirstStepSpace / IsTerminalStepSpace / EpisodeStepSpace
- MotionStateSpace / KinematicStateSpace / TrajectoryStateSpace
- RobustGoalSpace / MultiScaleGoalSpace
- Normalization
- reset() behavior
- get_gym_space() shapes
- safe_encode_observation() error handling
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from conftest import SimulationStateContainerStub

from rosnav_rl.observations.utils.pose import Pose2DType

# ---------- perception/laser ----------
from rosnav_rl.spaces.observation_space.spaces.perception.laser.basic_laser_spaces import (
    LaserScanSpace,
    ReducedLaserScanSpace,
)

# ---------- navigation ----------
from rosnav_rl.spaces.observation_space.spaces.navigation.basic_navigation_spaces import (
    DistAngleToGoalSpace,
    DistAngleToSubgoalSpace,
)
from rosnav_rl.spaces.observation_space.spaces.navigation.advanced_navigation_spaces import (
    RobustGoalSpace,
    MultiScaleGoalSpace,
)

# ---------- dynamics ----------
from rosnav_rl.spaces.observation_space.spaces.dynamics.basic_dynamics_spaces import (
    LastActionSpace,
    SubgoalInRobotFrameSpace,
)
from rosnav_rl.spaces.observation_space.spaces.dynamics.advanced_dynamics_spaces import (
    MotionStateSpace,
    KinematicStateSpace,
)

# ---------- meta ----------
from rosnav_rl.spaces.observation_space.spaces.meta.basic_meta_spaces import (
    IsFirstStepSpace,
    IsTerminalStepSpace,
    EpisodeStepSpace,
)


# =====================================================================
#  Helper
# =====================================================================

def make_pose(x, y, yaw):
    p = np.zeros(1, dtype=Pose2DType)[0]
    p["x"] = x
    p["y"] = y
    p["yaw"] = yaw
    return p


# =====================================================================
#  LaserScanSpace
# =====================================================================

class TestLaserScanSpace:
    def test_shape(self):
        space = LaserScanSpace(laser_num_beams=360, laser_max_range=10.0)
        assert space.shape == (360,)

    def test_encode_observation_clips(self):
        space = LaserScanSpace(laser_num_beams=5, laser_max_range=3.5)
        laser = np.array([1.0, 2.0, 5.0, 0.5, 3.5], dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        assert result.shape == (5,)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.5, 0.5, 3.5])

    def test_no_mutation_of_input(self):
        space = LaserScanSpace(laser_num_beams=3, laser_max_range=2.0)
        laser = np.array([5.0, 1.0, 3.0], dtype=np.float32)
        original = laser.copy()

        space.encode_observation(front_laser=laser)

        np.testing.assert_array_equal(laser, original)

    def test_all_within_range(self):
        space = LaserScanSpace(laser_num_beams=3, laser_max_range=10.0)
        laser = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        np.testing.assert_array_equal(result, laser)

    def test_zeros(self):
        space = LaserScanSpace(laser_num_beams=3, laser_max_range=5.0)
        laser = np.zeros(3, dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_gym_space_bounds(self):
        space = LaserScanSpace(laser_num_beams=100, laser_max_range=8.0)
        gym_space = space.get_gym_space()

        assert gym_space.shape == (100,)
        assert gym_space.low[0] == 0.0
        assert gym_space.high[0] == 8.0

    def test_with_normalization(self):
        space = LaserScanSpace(laser_num_beams=3, laser_max_range=10.0,
                               normalize=True, normalizer="max_abs")
        laser = np.array([0.0, 5.0, 10.0], dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        # max_abs: (2 * (x - low)) / (high - low) - 1
        # For low=0, high=10: (2 * x / 10) - 1
        expected = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)


# =====================================================================
#  ReducedLaserScanSpace
# =====================================================================

class TestReducedLaserScanSpace:
    def test_shape(self):
        space = ReducedLaserScanSpace(
            laser_num_beams=360, laser_max_range=10.0, reduced_num_beams=60
        )
        assert space.shape == (60,)

    def test_reduction(self):
        space = ReducedLaserScanSpace(
            laser_num_beams=10, laser_max_range=10.0, reduced_num_beams=5
        )
        laser = np.arange(10, dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        assert result.shape == (5,)
        # Indices should be evenly spaced: 0, 2, 4, 6, 8
        np.testing.assert_array_equal(result, [0, 2, 4, 6, 8])

    def test_reduction_with_clamping(self):
        space = ReducedLaserScanSpace(
            laser_num_beams=10, laser_max_range=5.0, reduced_num_beams=3
        )
        laser = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        assert result.shape == (3,)
        # Values > 5.0 should be clamped to 5.0
        assert np.all(result <= 5.0)

    def test_cached_indices(self):
        space = ReducedLaserScanSpace(
            laser_num_beams=360, laser_max_range=10.0, reduced_num_beams=60
        )

        idx1 = space.get_indices()
        idx2 = space.get_indices()

        assert idx1 is idx2  # Same object from cache

    def test_invalid_reduced_size(self):
        space = ReducedLaserScanSpace(
            laser_num_beams=5, laser_max_range=10.0, reduced_num_beams=10
        )
        with pytest.raises(ValueError, match="Cannot reduce"):
            space.get_indices()

    def test_same_as_full(self):
        """When reduced == full, output is same as full."""
        space = ReducedLaserScanSpace(
            laser_num_beams=5, laser_max_range=10.0, reduced_num_beams=5
        )
        laser = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        result = space.encode_observation(front_laser=laser)

        np.testing.assert_array_equal(result, laser)


# =====================================================================
#  DistAngleToGoalSpace
# =====================================================================

class TestDistAngleToGoalSpace:
    def test_shape(self):
        space = DistAngleToGoalSpace(goal_max_dist=30.0)
        assert space.shape == (2,)

    def test_passthrough(self):
        space = DistAngleToGoalSpace(goal_max_dist=30.0)
        da = np.array([5.0, 0.5], dtype=np.float32)

        result = space.encode_observation(dist_angle_to_goal=da)

        np.testing.assert_array_equal(result, da)

    def test_gym_space_bounds(self):
        space = DistAngleToGoalSpace(goal_max_dist=20.0)
        gs = space.get_gym_space()

        assert gs.low[0] == 0.0
        assert gs.high[0] == 20.0
        assert gs.low[1] == pytest.approx(-np.pi)
        assert gs.high[1] == pytest.approx(np.pi)


# =====================================================================
#  DistAngleToSubgoalSpace
# =====================================================================

class TestDistAngleToSubgoalSpace:
    def test_passthrough(self):
        space = DistAngleToSubgoalSpace(subgoal_max_dist=15.0)
        da = np.array([3.0, -0.5], dtype=np.float32)

        result = space.encode_observation(dist_angle_to_subgoal=da)

        np.testing.assert_array_equal(result, da)


# =====================================================================
#  LastActionSpace
# =====================================================================

class TestLastActionSpace:
    def test_differential_drive_shape(self):
        space = LastActionSpace(
            min_linear_vel=-0.5, max_linear_vel=1.0,
            min_angular_vel=-1.0, max_angular_vel=1.0,
        )
        assert space.shape == (2,)

    def test_holonomic_shape(self):
        space = LastActionSpace(
            min_linear_vel=-0.5, max_linear_vel=1.0,
            min_angular_vel=-1.0, max_angular_vel=1.0,
            min_translational_vel=-0.5, max_translational_vel=0.5,
        )
        assert space.shape == (3,)

    def test_passthrough(self):
        space = LastActionSpace(
            min_linear_vel=-1.0, max_linear_vel=1.0,
            min_angular_vel=-1.0, max_angular_vel=1.0,
        )
        action = np.array([0.5, 0.3], dtype=np.float32)

        result = space.encode_observation(last_action=action)

        np.testing.assert_array_equal(result, action)


# =====================================================================
#  SubgoalInRobotFrameSpace
# =====================================================================

class TestSubgoalInRobotFrameSpace:
    def test_shape(self):
        space = SubgoalInRobotFrameSpace(subgoal_max_dist=5.0)
        assert space.shape == (2,)

    def test_passthrough(self):
        space = SubgoalInRobotFrameSpace(subgoal_max_dist=5.0)
        subgoal = np.array([2.5, -1.3], dtype=np.float32)

        result = space.encode_observation(subgoal_in_robot_frame=subgoal)

        np.testing.assert_array_equal(result, subgoal)


# =====================================================================
#  IsFirstStepSpace / IsTerminalStepSpace / EpisodeStepSpace
# =====================================================================

class TestMetaSpaces:
    def test_is_first_space(self):
        space = IsFirstStepSpace()
        assert space.encode_observation(is_first=1) == 1
        assert space.encode_observation(is_first=0) == 0

    def test_is_terminal_space(self):
        space = IsTerminalStepSpace()
        assert space.encode_observation(is_terminal=1) == 1
        assert space.encode_observation(is_terminal=0) == 0

    def test_episode_step_space(self):
        space = EpisodeStepSpace(max_episode_steps=500)
        result = space.encode_observation(episode_step=42)

        np.testing.assert_array_equal(result, [42])

    def test_episode_step_default(self):
        space = EpisodeStepSpace(max_episode_steps=500)
        result = space.encode_observation()

        np.testing.assert_array_equal(result, [0])

    def test_episode_step_gym_space(self):
        space = EpisodeStepSpace(max_episode_steps=1000)
        gs = space.get_gym_space()

        assert gs.shape == (1,)
        assert gs.high[0] == 1000


# =====================================================================
#  MotionStateSpace
# =====================================================================

class TestMotionStateSpace:
    def test_shape_without_acceleration(self):
        space = MotionStateSpace(max_velocity=2.0)
        assert space.shape == (3,)

    def test_shape_with_acceleration(self):
        space = MotionStateSpace(max_velocity=2.0, include_acceleration=True)
        assert space.shape == (4,)

    def test_zero_velocity(self):
        space = MotionStateSpace(max_velocity=2.0)
        action = np.array([0.0, 0.0, 0.0])

        result = space.encode_observation(last_action=action)

        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)  # magnitude
        assert result[1] == pytest.approx(0.0)  # direction

    def test_forward_motion(self):
        space = MotionStateSpace(max_velocity=2.0)
        action = np.array([1.0, 0.0, 0.0])

        result = space.encode_observation(last_action=action)

        # normalized_magnitude = tanh(1.0 / 2.0) = tanh(0.5)
        assert result[0] == pytest.approx(np.tanh(0.5))
        # direction = arctan2(0, 1) / pi = 0
        assert result[1] == pytest.approx(0.0)

    def test_stability_improves_over_time(self):
        space = MotionStateSpace(max_velocity=2.0, stability_window=5)
        action = np.array([0.5, 0.0, 0.0])

        results = []
        for _ in range(5):
            r = space.encode_observation(last_action=action)
            results.append(r[2])  # stability metric

        # With constant velocity, stability should increase (variance → 0)
        assert results[-1] >= results[1]

    def test_reset_clears_history(self):
        space = MotionStateSpace(max_velocity=2.0, stability_window=5)

        for _ in range(3):
            space.encode_observation(last_action=np.array([0.5, 0.0, 0.2]))

        assert len(space.velocity_history) > 0

        space.reset()

        assert len(space.velocity_history) == 0
        assert space.last_velocity is None

    def test_acceleration_computed(self):
        space = MotionStateSpace(max_velocity=2.0, include_acceleration=True)

        # First action
        result1 = space.encode_observation(last_action=np.array([0.0, 0.0, 0.0]))
        assert result1[3] == pytest.approx(0.0)  # No acceleration initially

        # Second action with change
        result2 = space.encode_observation(last_action=np.array([1.0, 0.0, 0.0]))
        assert result2[3] > 0.0  # Acceleration should be > 0


# =====================================================================
#  KinematicStateSpace
# =====================================================================

class TestKinematicStateSpace:
    def test_shape_with_alignment(self):
        space = KinematicStateSpace(max_velocity=2.0, include_motion_alignment=True)
        assert space.shape == (7,)

    def test_shape_without_alignment(self):
        space = KinematicStateSpace(max_velocity=2.0, include_motion_alignment=False)
        assert space.shape == (6,)

    def test_origin_zero_velocity(self):
        space = KinematicStateSpace(max_velocity=2.0, include_motion_alignment=True)
        pose = make_pose(0.0, 0.0, 0.0)
        action = np.array([0.0, 0.0, 0.0])

        result = space.encode_observation(robot_pose=pose, last_action=action)

        assert result.shape == (7,)
        assert result[0] == pytest.approx(0.0)  # x
        assert result[1] == pytest.approx(0.0)  # y
        assert result[2] == pytest.approx(1.0)  # cos(0)
        assert result[3] == pytest.approx(0.0)  # sin(0)
        assert result[4] == pytest.approx(0.0)  # linear_vel
        assert result[5] == pytest.approx(0.0)  # angular_vel
        assert result[6] == pytest.approx(0.0)  # motion_alignment (no motion)

    def test_forward_motion_alignment(self):
        space = KinematicStateSpace(max_velocity=2.0, include_motion_alignment=True)
        pose = make_pose(0.0, 0.0, 0.0)
        action = np.array([1.0, 0.0, 0.0])

        result = space.encode_observation(robot_pose=pose, last_action=action)

        # Pure forward: alignment = cos(arctan2(0, 1)) = cos(0) = 1.0
        assert result[6] == pytest.approx(1.0)

    def test_backward_motion_alignment(self):
        space = KinematicStateSpace(max_velocity=2.0, include_motion_alignment=True)
        pose = make_pose(0.0, 0.0, 0.0)
        action = np.array([-1.0, 0.0, 0.0])

        result = space.encode_observation(robot_pose=pose, last_action=action)

        # Pure backward: alignment = cos(arctan2(0, -1)) = cos(pi) = -1.0
        assert result[6] == pytest.approx(-1.0)

    def test_position_normalization(self):
        space = KinematicStateSpace(max_velocity=2.0, position_scale=10.0)
        pose = make_pose(10.0, -5.0, np.pi / 4)
        action = np.array([0.0, 0.0, 0.0])

        result = space.encode_observation(robot_pose=pose, last_action=action)

        assert result[0] == pytest.approx(np.tanh(1.0))  # tanh(10/10)
        assert result[1] == pytest.approx(np.tanh(-0.5))  # tanh(-5/10)

    def test_velocity_clipping(self):
        space = KinematicStateSpace(max_velocity=1.0)
        pose = make_pose(0.0, 0.0, 0.0)
        action = np.array([5.0, 0.0, 5.0])  # Far exceeds max

        result = space.encode_observation(robot_pose=pose, last_action=action)

        # Should be clipped to 1.0
        assert result[4] == pytest.approx(1.0)
        assert result[5] == pytest.approx(1.0)


# =====================================================================
#  RobustGoalSpace
# =====================================================================

class TestRobustGoalSpace:
    def test_shape_with_progress(self):
        space = RobustGoalSpace(include_progress=True)
        assert space.shape == (3,)

    def test_shape_without_progress(self):
        space = RobustGoalSpace(include_progress=False)
        assert space.shape == (2,)

    def test_tanh_distance_normalization(self):
        space = RobustGoalSpace(goal_max_dist=50.0, distance_scaling="tanh",
                                include_progress=False)
        da = np.array([25.0, 0.0])

        result = space.encode_observation(dist_angle_to_goal=da)

        assert result[0] == pytest.approx(np.tanh(25.0 / 50.0))
        assert result[1] == pytest.approx(0.0)

    def test_linear_distance_normalization(self):
        space = RobustGoalSpace(goal_max_dist=10.0, distance_scaling="linear",
                                include_progress=False)
        da = np.array([5.0, np.pi / 2])

        result = space.encode_observation(dist_angle_to_goal=da)

        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)  # pi/2 / pi

    def test_log_distance_normalization(self):
        space = RobustGoalSpace(goal_max_dist=10.0, distance_scaling="log",
                                include_progress=False)
        da = np.array([10.0, 0.0])

        result = space.encode_observation(dist_angle_to_goal=da)

        # log(1+10) / log(1+10) = 1.0
        assert result[0] == pytest.approx(1.0)

    def test_progress_tracking(self):
        space = RobustGoalSpace(goal_max_dist=50.0, include_progress=True)

        # First observation — no progress (baseline)
        result1 = space.encode_observation(dist_angle_to_goal=np.array([10.0, 0.0]))
        assert result1[2] == pytest.approx(0.0)

        # Closer — progress > 0
        result2 = space.encode_observation(dist_angle_to_goal=np.array([5.0, 0.0]))
        assert result2[2] > 0.0

    def test_progress_moving_away_is_zero(self):
        space = RobustGoalSpace(goal_max_dist=50.0, include_progress=True)

        space.encode_observation(dist_angle_to_goal=np.array([5.0, 0.0]))
        result = space.encode_observation(dist_angle_to_goal=np.array([10.0, 0.0]))

        # Moving away => progress = 0 (max(0, negative) = 0)
        assert result[2] == pytest.approx(0.0)

    def test_reset(self):
        space = RobustGoalSpace(include_progress=True)
        space.encode_observation(dist_angle_to_goal=np.array([5.0, 0.0]))

        assert space.last_distance is not None

        space.reset()

        assert space.last_distance is None


# =====================================================================
#  MultiScaleGoalSpace
# =====================================================================

class TestMultiScaleGoalSpace:
    def test_shape(self):
        space = MultiScaleGoalSpace(distance_scales=[0.5, 1.0, 2.0])
        assert space.shape == (6,)

    def test_default_scales(self):
        space = MultiScaleGoalSpace()
        assert len(space.distance_scales) == 3
        assert space.shape == (6,)


# =====================================================================
#  Base observation space: safe_encode_observation
# =====================================================================

class TestSafeEncodeObservation:
    def test_returns_zeros_on_missing_key(self):
        space = DistAngleToGoalSpace(goal_max_dist=30.0)
        # Missing the required key
        result = space.safe_encode_observation()

        # Should return null observation
        assert result is not None
        np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))

    def test_returns_correct_data_on_success(self):
        space = DistAngleToGoalSpace(goal_max_dist=30.0)
        da = np.array([5.0, 0.3], dtype=np.float32)

        result = space.safe_encode_observation(dist_angle_to_goal=da)

        np.testing.assert_array_equal(result, da)

    def test_increments_encode_failure_count_on_missing_key(self):
        """Regression test for P2.2 (audit 2026-07-04)."""
        space = DistAngleToGoalSpace(goal_max_dist=30.0)
        assert space.encode_failure_count == 0

        space.safe_encode_observation()
        space.safe_encode_observation()

        assert space.encode_failure_count == 2

    def test_strict_mode_reraises_instead_of_returning_null(self):
        """Regression test for P2.2 (audit 2026-07-04): ``strict=True`` must
        re-raise encoding errors instead of silently falling back to a
        zero-filled observation, so tests/CI can catch encoding bugs.

        ``dist_angle_to_goal`` is a required positional argument with no
        default, so calling with no kwargs raises ``TypeError`` (missing
        argument) — caught by ``safe_encode_observation``'s second except
        clause, not the ``KeyError`` clause.
        """
        space = DistAngleToGoalSpace(goal_max_dist=30.0, strict=True)

        with pytest.raises(TypeError):
            space.safe_encode_observation()

        # The failure must still be counted even though it re-raised.
        assert space.encode_failure_count == 1

    def test_strict_mode_reraises_on_none_result(self):
        """Regression test for P2.2 (audit 2026-07-04): the "encode_observation
        returned None" branch must also re-raise under strict mode, not just
        the exception-catching branches.
        """
        space = DistAngleToGoalSpace(goal_max_dist=30.0, strict=True)
        with patch.object(space, "encode_observation", return_value=None):
            with pytest.raises(ValueError, match="returned None"):
                space.safe_encode_observation(dist_angle_to_goal=np.array([1.0, 0.0]))

    def test_non_strict_mode_unaffected(self):
        """``strict=False`` (the default) must keep returning a null
        observation on failure, not raise — the real-time loop must keep
        running.
        """
        space = DistAngleToGoalSpace(goal_max_dist=30.0, strict=False)

        result = space.safe_encode_observation()

        np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))


# =====================================================================
#  Normalization integration
# =====================================================================

class TestNormalizationIntegration:
    def test_min_max_scaler(self):
        space = DistAngleToGoalSpace(goal_max_dist=10.0,
                                      normalize=True, normalizer="min_max")
        da = np.array([5.0, 0.0], dtype=np.float32)

        result = space.encode_observation(dist_angle_to_goal=da)

        # min_max: (x - low) / (high - low)
        # distance: (5 - 0) / (10 - 0) = 0.5
        # angle: (0 - (-pi)) / (pi - (-pi)) = 0.5
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)

    def test_identity_normalizer(self):
        space = DistAngleToGoalSpace(goal_max_dist=10.0,
                                      normalize=True, normalizer="identity")
        da = np.array([7.0, 1.5], dtype=np.float32)

        result = space.encode_observation(dist_angle_to_goal=da)

        np.testing.assert_array_equal(result, da)

    def test_no_normalization_by_default(self):
        space = DistAngleToGoalSpace(goal_max_dist=10.0)
        da = np.array([7.0, 1.5], dtype=np.float32)

        result = space.encode_observation(dist_angle_to_goal=da)

        np.testing.assert_array_equal(result, da)


# =====================================================================
#  apply_limit static method
# =====================================================================

class TestApplyLimit:
    def test_basic_clamping(self):
        result = LaserScanSpace.apply_limit(
            np.array([1.0, 5.0, 10.0]), max_range=3.0
        )
        np.testing.assert_array_equal(result, [1.0, 3.0, 3.0])

    def test_no_clamping_needed(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = LaserScanSpace.apply_limit(arr, max_range=10.0)
        np.testing.assert_array_equal(result, arr)

    def test_returns_new_array(self):
        arr = np.array([5.0, 10.0])
        result = LaserScanSpace.apply_limit(arr, max_range=3.0)

        # Must not share memory
        assert not np.shares_memory(result, arr)


# =====================================================================
#  ObservationSpaceManager.reset_spaces(): error-flush cadence
# =====================================================================


class TestObservationSpaceManagerResetFlush:
    """``flush_and_log_errors()`` — the only function that drains the global
    error/warning collector into actual log output — used to have zero
    call sites anywhere in production code, so every warning/error reported
    through ``ErrorReportingMixin``/``_report_error``/``_report_warning``
    was silently buffered forever and never surfaced. It is now called from
    ``ObservationSpaceManager.reset_spaces()``, the one choke point every
    episode-reset path (training, inference, action server) already goes
    through.
    """

    def test_reset_spaces_flushes_error_collector(self):
        from rosnav_rl.spaces.observation_space.observation_space_manager import (
            ObservationSpaceManager,
        )

        manager = ObservationSpaceManager()

        with patch(
            "rosnav_rl.spaces.observation_space.observation_space_manager."
            "flush_and_log_errors"
        ) as mock_flush:
            manager.reset_spaces()

        mock_flush.assert_called_once()
