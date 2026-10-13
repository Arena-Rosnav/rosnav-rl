"""BaseObservationSpace null fallbacks, validation and normalization decorators."""

import numpy as np
import pytest
from gymnasium import spaces

from rosnav_rl.observations.utils.pose import Pose2DType
from rosnav_rl.spaces.observation_space.spaces.base_observation_space import BaseObservationSpace
from rosnav_rl.spaces.observation_space.spaces.localization.advanced_localization_spaces import PoseStabilizedSpace
from rosnav_rl.spaces.observation_space.spaces.meta.advanced_meta_spaces import MissionContextSpace
from rosnav_rl.spaces.observation_space.spaces.perception.laser.basic_laser_spaces import LaserScanSpace
from rosnav_rl.utils.logging import ErrorSeverity, get_error_collector


class EchoSpace(BaseObservationSpace):
    name = "EchoSpace"

    def __init__(self, gym_space=None, **kwargs):
        self._gym_space = gym_space if gym_space is not None else spaces.Box(0.0, 2.0, shape=(2,))
        super().__init__(**kwargs)

    def get_gym_space(self):
        return self._gym_space

    def encode_observation(self, value):
        return value


class ValidatedEchoSpace(EchoSpace):
    name = "ValidatedEchoSpace"

    @BaseObservationSpace.apply_normalization
    @BaseObservationSpace.check_dtype
    def encode_observation(self, value):
        return value


class DeferringSpace(BaseObservationSpace):
    name = "DeferringSpace"

    def get_gym_space(self):
        return super().get_gym_space()

    def encode_observation(self, *args, **kwargs):
        return super().encode_observation(*args, **kwargs)


class DeferringEncodeSpace(DeferringSpace):
    name = "DeferringEncodeSpace"

    def get_gym_space(self):
        return spaces.Box(0.0, 1.0, shape=(1,))


@pytest.fixture
def collector():
    errors = get_error_collector()
    previous = errors._min_severity
    errors.set_min_severity(ErrorSeverity.DEBUG)
    errors.clear()
    yield errors
    errors.clear()
    errors.set_min_severity(previous)


def reports(errors):
    return [(msg.component_name, msg.severity, msg.error_type) for msg in errors._errors]


def test_repr_config_and_space_accessors():
    space = LaserScanSpace(laser_num_beams=3, laser_max_range=10.0, normalize=True, normalizer="min_max")

    config = space.config
    config["normalize"] = False

    assert repr(space) == "LaserScanSpace(name='LaserScanSpace')"
    assert space.config == {"normalize": True, "normalizer": "min_max"}
    assert isinstance(space.space, spaces.Box)
    assert space.shape == (3,)


def test_unknown_normalizer_warns_and_leaves_values_unscaled(collector):
    space = LaserScanSpace(laser_num_beams=3, laser_max_range=10.0, normalize=True, normalizer="bogus")

    result = space.encode_observation(np.array([5.0, 20.0, 1.0]))

    np.testing.assert_allclose(result, [5.0, 10.0, 1.0])
    assert reports(collector) == [("LaserScanSpace", ErrorSeverity.WARNING, None)]


def test_missing_key_in_the_pose_falls_back_to_zeros(collector):
    space = PoseStabilizedSpace()

    result = space.safe_encode_observation(robot_pose={"x": 1.0, "y": 2.0, "yaw": 0.0})

    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))
    assert reports(collector) == [("PoseStabilizedSpace", ErrorSeverity.ERROR, "KeyError")]


def test_unexpected_exception_falls_back_to_zeros(collector):
    space = MissionContextSpace(max_mission_time=0.0)

    result = space.safe_encode_observation(dist_angle_to_goal=np.array([3.0, 0.0]))

    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))
    assert reports(collector) == [("MissionContextSpace", ErrorSeverity.ERROR, "ZeroDivisionError")]


def test_successful_encode_passes_through_safe_encode():
    pose = np.array([(10.0, 0.0, 0.0)], dtype=Pose2DType)[0]

    np.testing.assert_allclose(PoseStabilizedSpace().safe_encode_observation(pose), [np.tanh(1.0), 0, 1, 0], atol=1e-6)


def test_none_result_is_a_warning_with_box_null(collector):
    result = EchoSpace().safe_encode_observation(None)

    np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))
    assert reports(collector) == [("EchoSpace", ErrorSeverity.WARNING, None)]


def test_discrete_space_null_is_a_single_zero_index():
    result = EchoSpace(spaces.Discrete(3)).safe_encode_observation(None)

    assert result.dtype == np.int32
    assert result.tolist() == [0]


def test_dict_space_null_matches_each_subspace():
    space = EchoSpace(
        spaces.Dict(
            {
                "scan": spaces.Box(0.0, 1.0, shape=(2, 3)),
                "mode": spaces.Discrete(4),
                "flags": spaces.MultiBinary(5),
            }
        )
    )

    result = space.safe_encode_observation(None)

    assert set(result) == {"scan", "mode", "flags"}
    assert result["scan"].shape == (2, 3)
    assert result["mode"].tolist() == [0]
    assert result["flags"].tolist() == [0.0]


def test_unsupported_space_null_warns_and_returns_a_scalar_zero(collector):
    result = EchoSpace(spaces.MultiBinary(4)).safe_encode_observation(None)

    assert result.tolist() == [0.0]
    assert [severity for _, severity, _ in reports(collector)] == [ErrorSeverity.WARNING, ErrorSeverity.WARNING]


@pytest.mark.parametrize("value", [np.array([np.nan, 1.0]), np.array([1.0, np.inf]), np.array([0.5, 1 + 2j])])
def test_non_finite_or_complex_observations_become_zeros(collector, value):
    result = ValidatedEchoSpace().encode_observation(value)

    np.testing.assert_array_equal(result, np.zeros(2))
    assert reports(collector) == [("ValidatedEchoSpace", ErrorSeverity.WARNING, None)]


def test_normalization_uses_the_box_bounds():
    space = ValidatedEchoSpace(spaces.Box(0.0, 2.0, shape=(2,)), normalize=True, normalizer="min_max")

    np.testing.assert_allclose(space.encode_observation(np.array([1.0, 2.0])), [0.5, 1.0])


def test_normalization_is_skipped_for_non_box_spaces():
    space = ValidatedEchoSpace(spaces.Discrete(3), normalize=True, normalizer="min_max")

    assert space.encode_observation(np.array([2.0])).tolist() == [2.0]
    assert space._norm_low is None


def test_base_gym_space_is_abstract():
    with pytest.raises(NotImplementedError, match="get_gym_space"):
        DeferringSpace()


def test_base_encode_is_abstract():
    with pytest.raises(NotImplementedError, match="encode_observation"):
        DeferringEncodeSpace().encode_observation()
