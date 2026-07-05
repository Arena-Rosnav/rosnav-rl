"""Regression tests for ObservationSpaceManager encoding (P3.4, audit 2026-07-04).

Pins exact encode_observation() output before/after the P3.4 refactor, which
replaces the per-call dict comprehension in _extract_space_args() and the
per-call OrderedDict() build in _encode_sequential() with reusable mutable
containers (same trick already used by RewardFunction._prepare_execution_kwargs).
No behavior change allowed.
"""

from rosnav_rl.spaces.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)


def _make_manager():
    manager = ObservationSpaceManager(validate_observations=False)
    manager.load_configuration(
        {
            "IsFirstStepSpace": {},
            "IsTerminalStepSpace": {},
        }
    )
    return manager


def test_encode_observation_matches_pinned_output_multi_space():
    manager = _make_manager()

    out1 = manager.encode_observation({"is_first": 1, "is_terminal": 0})
    assert out1 == {"IsFirstStepSpace": 1, "IsTerminalStepSpace": 0}

    out2 = manager.encode_observation({"is_first": 0, "is_terminal": 1})
    assert out2 == {"IsFirstStepSpace": 0, "IsTerminalStepSpace": 1}


def test_encode_observation_single_space_returns_raw_value():
    manager = ObservationSpaceManager(validate_observations=False)
    manager.load_configuration({"IsFirstStepSpace": {}})

    out = manager.encode_observation({"is_first": 1})
    assert out == 1


def test_encode_observation_repeated_calls_do_not_leak_between_spaces():
    manager = _make_manager()

    manager.encode_observation({"is_first": 1, "is_terminal": 1})
    out = manager.encode_observation({"is_first": 0, "is_terminal": 0})

    # A reused-but-not-cleared buffer would leak stale values from the
    # previous call into this one.
    assert out == {"IsFirstStepSpace": 0, "IsTerminalStepSpace": 0}


def test_encode_observation_returned_dict_is_independent_snapshot():
    manager = _make_manager()

    out1 = manager.encode_observation({"is_first": 1, "is_terminal": 0})
    manager.encode_observation({"is_first": 0, "is_terminal": 1})

    # Mutating internal reusable buffers on later calls must not retroactively
    # change a dict already handed back to a previous caller.
    assert out1 == {"IsFirstStepSpace": 1, "IsTerminalStepSpace": 0}
