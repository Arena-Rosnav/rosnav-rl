"""Regression tests for ObservationManager's simulation_state_container handling
(P2.6, audit 2026-07-04): a missing container used to be silently replaced
with a default, unconfigured AgentParameters() (only a log warning) — a
safety hazard since real robot parameters (e.g. safe distance) would
silently be replaced with placeholder values. This is now opt-in via
allow_default_params, defaulting to a hard ValueError.
"""

from unittest.mock import MagicMock

import pytest

from rosnav_rl.observations.core.manager import ObservationManager


class TestObservationManagerDefaultParams:
    def test_raises_by_default_when_container_is_none(self):
        with pytest.raises(ValueError, match="simulation_state_container"):
            ObservationManager(
                node=MagicMock(),
                ns="",
                data_sources={},
                simulation_state_container=None,
            )

    def test_falls_back_to_defaults_when_opted_in(self):
        manager = ObservationManager(
            node=MagicMock(),
            ns="",
            data_sources={},
            simulation_state_container=None,
            allow_default_params=True,
        )

        assert manager._simulation_state_container is not None
