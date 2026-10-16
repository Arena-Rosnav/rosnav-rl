"""Mission, performance and safety context vectors from known goal and laser sequences."""

import numpy as np
import pytest

from rosnav_rl.spaces.observation_space.spaces.meta.advanced_meta_spaces import (
    MissionContextSpace,
    PerformanceContextSpace,
    SafetyContextSpace,
)


def goal(distance, angle=0.0):
    return np.array([distance, angle])


def action(linear, angular=0.0):
    return np.array([linear, 0.0, angular])


@pytest.mark.parametrize("space_class", [MissionContextSpace, PerformanceContextSpace, SafetyContextSpace])
def test_context_spaces_are_four_unit_boxes(space_class):
    space = space_class().space

    assert space.shape == (4,)
    np.testing.assert_array_equal(space.low, np.zeros(4))
    np.testing.assert_array_equal(space.high, np.ones(4))


def test_mission_first_step_has_no_progress_and_phase_from_thresholds():
    space = MissionContextSpace(max_mission_time=1.0, progress_smoothing=0.5)

    np.testing.assert_allclose(space.encode_observation(goal(10.0)), [0.0, 0.0, 1 / 3, 0.0], atol=1e-6)
    assert space.initial_distance == 10.0
    assert space.mission_start_time == 1


def test_mission_progress_is_exponentially_smoothed():
    space = MissionContextSpace(max_mission_time=1.0, progress_smoothing=0.5)
    space.encode_observation(goal(10.0))

    second = space.encode_observation(goal(5.0))
    third = space.encode_observation(goal(2.0))

    np.testing.assert_allclose(second, [0.25, 0.1, 1 / 3, 0.0], atol=1e-6)
    np.testing.assert_allclose(third, [0.5 * 0.25 + 0.5 * 0.8, 0.2, 2 / 3, 0.0], atol=1e-6)


@pytest.mark.parametrize(("distance", "phase"), [(15.0, 0.0), (10.5, 0.0), (3.0, 2 / 3), (1.0, 1.0), (0.2, 1.0)])
def test_mission_phase_buckets_distance_by_descending_thresholds(distance, phase):
    space = MissionContextSpace(phase_distance_thresholds=[1.0, 10.0, 3.0])

    assert space.encode_observation(goal(distance))[2] == pytest.approx(phase)


def test_mission_urgency_grows_when_progress_lags_behind_time_and_saturates():
    space = MissionContextSpace(max_mission_time=1.0, progress_smoothing=0.5)

    urgencies = [space.encode_observation(goal(10.0))[3] for _ in range(12)]

    assert urgencies[5] == pytest.approx(min(1.0, 0.8 * 0.5 * 2.0))
    assert urgencies[11] == 1.0
    assert space.encode_observation(goal(10.0))[1] == 1.0


def test_mission_initial_distance_is_floored_at_one_meter():
    space = MissionContextSpace(progress_smoothing=0.0)

    space.encode_observation(goal(0.25))

    assert space.initial_distance == 1.0
    assert space.encode_observation(goal(0.5))[0] == pytest.approx(0.5)


def test_mission_progress_before_initialization_is_zero():
    assert MissionContextSpace()._compute_progress(3.0) == 0.0


def test_mission_reset_restarts_the_mission():
    space = MissionContextSpace(max_mission_time=1.0, progress_smoothing=0.5)
    for distance in (10.0, 6.0, 4.0):
        space.encode_observation(goal(distance))

    space.reset()

    np.testing.assert_allclose(space.encode_observation(goal(4.0)), [0.0, 0.0, 1 / 3, 0.0], atol=1e-6)
    assert space.step_count == 1


def test_performance_tracks_path_energy_and_smoothness_over_the_window():
    space = PerformanceContextSpace(efficiency_window=3, max_velocity=2.0)

    first = space.encode_observation(goal(10.0), action(1.0))
    second = space.encode_observation(goal(9.0), action(1.0))
    third = space.encode_observation(goal(9.5), action(0.0))
    fourth = space.encode_observation(goal(8.0), action(0.0))

    np.testing.assert_allclose(first, [0.5, 0.5, 1.0, 0.6], atol=1e-6)
    np.testing.assert_allclose(second, [1.0, 0.5, 1.0, 0.85], atol=1e-6)
    np.testing.assert_allclose(third, [0.5, 2 / 3, 0.5, 0.25 + 0.2 + 0.1], atol=1e-6)
    np.testing.assert_allclose(fourth, [0.5, 5 / 6, 0.5, 0.25 + 0.25 + 0.1], atol=1e-6)
    assert space.distance_history == [9.0, 9.5, 8.0]


def test_performance_energy_uses_linear_and_angular_magnitude_and_clips_at_zero():
    space = PerformanceContextSpace(max_velocity=2.0)

    slow = space.encode_observation(goal(5.0), action(0.6, 0.8))
    space.reset()
    fast = space.encode_observation(goal(5.0), action(4.0))

    assert slow[1] == pytest.approx(0.5)
    assert fast[1] == 0.0


def test_performance_without_velocity_history_is_fully_energy_efficient():
    assert PerformanceContextSpace()._compute_energy_efficiency() == 1.0


def test_performance_reset_clears_histories():
    space = PerformanceContextSpace()
    space.encode_observation(goal(5.0), action(1.0))
    space.encode_observation(goal(4.0), action(1.0))

    space.reset()

    assert space.distance_history == []
    assert space.velocity_history == []
    np.testing.assert_allclose(space.encode_observation(goal(4.0), action(0.0)), [0.5, 1.0, 1.0, 0.75], atol=1e-6)


def test_safety_scales_risk_between_critical_and_safety_distance():
    space = SafetyContextSpace(safety_distance=1.0, critical_distance=0.5)

    result = space.encode_observation(np.array([2.0, 0.75, 0.0]), action(0.0))

    np.testing.assert_allclose(result, [0.5, 0.5, 0.0, 0.5], atol=1e-6)


def test_safety_history_violations_and_window():
    space = SafetyContextSpace(safety_distance=1.0, critical_distance=0.5, risk_history_length=2)

    space.encode_observation(np.array([2.0, 0.75]), action(0.0))
    critical = space.encode_observation(np.array([0.4, 3.0]), action(1.0))
    clear = space.encode_observation(np.array([5.0]), action(0.0))

    np.testing.assert_allclose(critical, [1.0, 0.75, 0.5, 0.25], atol=1e-6)
    np.testing.assert_allclose(clear, [0.0, 0.5, 1 / 3, 0.5], atol=1e-6)
    assert space.risk_history == [1.0, 0.0]


def test_safety_speed_amplifies_risk_up_to_one_and_a_half():
    space = SafetyContextSpace(safety_distance=1.0, critical_distance=0.0)

    moderate = space.encode_observation(np.array([0.8]), action(0.3, 0.4))[0]
    capped = space.encode_observation(np.array([0.8]), action(3.0))[0]

    assert moderate == pytest.approx(0.2 * 1.25)
    assert capped == pytest.approx(0.2 * 1.5)


def test_safety_without_valid_readings_has_no_risk():
    space = SafetyContextSpace()

    np.testing.assert_allclose(space.encode_observation(np.array([0.0, -1.0, np.nan]), action(1.0)), [0, 0, 0, 1])


def test_safety_normalized_output_maps_unit_box_to_symmetric_range():
    space = SafetyContextSpace(safety_distance=1.0, critical_distance=0.5, normalize=True, normalizer="max_abs")

    result = space.encode_observation(np.array([0.75]), action(0.0))

    np.testing.assert_allclose(result, [0.0, 0.0, -1.0, 0.0], atol=1e-6)


def test_safety_metrics_fall_back_to_current_risk_without_history():
    assert SafetyContextSpace()._compute_safety_metrics(0.4) == {
        "avg_risk": 0.4,
        "violation_rate": 0.0,
        "safety_margin": pytest.approx(0.6),
    }


def test_safety_reset_clears_tracking():
    space = SafetyContextSpace(safety_distance=1.0, critical_distance=0.5)
    space.encode_observation(np.array([0.1]), action(0.0))

    space.reset()

    assert (space.risk_history, space.violation_count, space.total_steps) == ([], 0, 0)
