"""RewardUnitFactory registration and lookup."""

import pytest

from rosnav_rl.reward.reward_units.reward_unit_factory import RewardUnitFactory
from rosnav_rl.reward.reward_units.reward_units import RewardGoalReached, RewardSafeDistance


def test_factory_rejects_a_name_that_is_already_registered():
    with pytest.raises(ValueError, match="goal_reached"):
        RewardUnitFactory.register("goal_reached")(RewardGoalReached)


def test_factory_rejects_classes_that_are_not_reward_units():
    with pytest.raises(TypeError, match="dict"):
        RewardUnitFactory.register("unregistered_plain_dict_unit")(dict)

    assert "unregistered_plain_dict_unit" not in RewardUnitFactory.registry


def test_factory_returns_the_registered_class():
    assert RewardUnitFactory.instantiate("safe_distance") is RewardSafeDistance


def test_factory_raises_for_unknown_names():
    with pytest.raises(KeyError, match="no_such_reward_unit"):
        RewardUnitFactory.instantiate("no_such_reward_unit")
