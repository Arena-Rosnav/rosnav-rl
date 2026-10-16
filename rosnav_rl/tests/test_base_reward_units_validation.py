"""RewardUnit base validation of schemas, reward values and info payloads."""

import numpy as np
import pytest

from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.reward.reward_units.base_reward_units import RewardUnit
from rosnav_rl.reward.reward_units.reward_units import RewardSafeDistance
from rosnav_rl.utils.logging import ErrorSeverity, get_error_collector


class ListRequiresUnit(RewardUnit):
    requires = ["laser"]

    def __call__(self, **kwargs):
        self.add_reward(1.0)


class IntKeyRequiresUnit(RewardUnit):
    requires = {"laser": float, 3: float}

    def __call__(self, **kwargs):
        self.add_reward(1.0)


class DeferringUnit(RewardUnit):
    requires = {}

    def __call__(self, **kwargs):
        super().__call__(**kwargs)


@pytest.fixture
def holder():
    return RewardFunction({})


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


def test_non_dict_requires_is_critical_and_leaves_no_cached_keys(holder, collector):
    unit = ListRequiresUnit(holder)

    assert unit._cached_requirements is None
    assert reports(collector) == [("ListRequiresUnit", ErrorSeverity.CRITICAL, None)]


def test_non_string_requirement_key_is_an_error_but_keys_are_cached(holder, collector):
    unit = IntKeyRequiresUnit(holder)

    assert unit._cached_requirements == {"laser", 3}
    assert reports(collector) == [("IntKeyRequiresUnit", ErrorSeverity.ERROR, None)]


@pytest.mark.parametrize(
    ("value", "error_type"),
    [("1.0", "TypeError"), (None, "TypeError"), (float("nan"), "ValueError"), (-np.inf, "ValueError")],
)
def test_invalid_reward_values_are_rejected_without_touching_the_total(holder, collector, value, error_type):
    unit = RewardSafeDistance(holder, reward=-0.1)

    unit.add_reward(value)

    assert holder.state.current_reward == 0.0
    assert reports(collector) == [("RewardSafeDistance", ErrorSeverity.ERROR, error_type)]


@pytest.mark.parametrize("value", [2, np.float32(0.5), np.int64(-3)])
def test_numpy_and_int_rewards_are_accumulated(holder, value):
    unit = RewardSafeDistance(holder, reward=-0.1)

    unit.add_reward(value)

    assert holder.state.current_reward == pytest.approx(float(value))
    assert holder.state.reward_overview == {"RewardSafeDistance": value}


def test_non_dict_info_is_rejected(holder, collector):
    unit = RewardSafeDistance(holder, reward=-0.1)

    unit.add_info([("is_done", True)])

    assert holder.state.info == {}
    assert reports(collector) == [("RewardSafeDistance", ErrorSeverity.ERROR, "TypeError")]


def test_base_reset_keeps_unit_state(holder):
    unit = RewardSafeDistance(holder, reward=-0.4)

    unit.reset()
    unit(laser_safety_violation=True)

    assert holder.state.current_reward == pytest.approx(-0.4)


def test_base_call_is_abstract(holder):
    unit = DeferringUnit(holder)

    with pytest.raises(NotImplementedError, match="DeferringUnit"):
        unit()
