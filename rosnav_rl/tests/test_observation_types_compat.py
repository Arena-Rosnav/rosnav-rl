"""Regression tests for the observations.utils.types compat shim (P4.4, audit 2026-07-04).

DataSpec and the per-field Annotated[...] type aliases moved from
rosnav_rl.observations.utils.types to rosnav_rl.utils.observation_types so
spaces/reward code doesn't need to depend on the observations package.
The old location must keep re-exporting every name (including the ones
missing from __all__) so existing importers are unaffected.
"""

import rosnav_rl.observations.utils.types as old_location
import rosnav_rl.utils.observation_types as new_location


def test_all_list_is_identical():
    assert old_location.__all__ == new_location.__all__


def test_all_listed_names_are_same_object():
    for name in new_location.__all__:
        assert getattr(old_location, name) is getattr(new_location, name)


def test_names_missing_from_all_are_still_reexported():
    # These are used by real importers (reward_units/goal.py,
    # observations/data_sources/generators.py,
    # spaces/.../meta/basic_meta_spaces.py) despite not being in __all__.
    for name in (
        "ArenaPedestrianStates",
        "GoalRelativePosition",
        "IsFirst",
        "PedestrianDistances",
    ):
        assert getattr(old_location, name) is getattr(new_location, name)


def test_data_spec_is_shared():
    assert old_location.DataSpec is new_location.DataSpec
