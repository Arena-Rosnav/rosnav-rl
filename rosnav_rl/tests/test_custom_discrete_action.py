from __future__ import annotations

import pytest

from rosnav_rl.utils.action_space.custom_discrete_action import generate_discrete_action_dict


@pytest.mark.parametrize(
    ("translational_range", "num_translational", "expected"),
    [((-0.2, 0.2), 3, [-0.2, 0.0, 0.2]), ((0.0, 0.0), 1, [0.0])],
)
def test_every_translational_value_is_paired_with_each_motion(translational_range, num_translational, expected):
    actions = generate_discrete_action_dict((0.0, 0.4), (-1.0, 1.0), 2, 3, translational_range, num_translational)

    motions = {(a["linear"], a["angular"]) for a in actions}
    assert len(actions) == len(motions) * num_translational
    assert sorted({round(a["translational"], 2) for a in actions}) == expected


def test_no_translational_range_yields_none():
    actions = generate_discrete_action_dict((0.0, 0.4), (-1.0, 1.0), 2, 3)

    assert {a["translational"] for a in actions} == {None}
