"""Regression test for guarded debug logging in ObservationManager (P3.5,
audit 2026-07-04).

_observation_callback / _synchronized_callback (nested closures inside
ObservationManager._setup_collectors) used to build an f-string debug
message unconditionally on every message, even when the DEBUG level was
disabled and the message would never be emitted. The fix guards the
f-string construction with self._logger.is_enabled_for(LoggingSeverity.DEBUG).

This must not change what gets logged when DEBUG *is* enabled, and must
skip building the message (never touching collector.name/update_count)
when DEBUG is disabled.
"""

from unittest.mock import MagicMock, patch

from rclpy.logging import LoggingSeverity

from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.observations.core.manager import ObservationManager
from rosnav_rl.observations.data_sources.base import Collector


class _FakeMsg:
    pass


class _FakeCollector(Collector[_FakeMsg, int]):
    topic = "fake_topic"

    def _preprocess(self, msg):
        return 0


def _make_manager(logger):
    node = MagicMock()
    node.get_logger.return_value = logger

    collector = _FakeCollector(name="fake", topic="fake_topic", node=node)

    with patch("rosnav_rl.observations.core.manager.SubscriptionManager") as sm_cls:
        manager = ObservationManager(
            node=node,
            ns="test_ns",
            data_sources={"fake": collector},
            simulation_state_container=AgentParameters(),
            validate_generators=False,
        )
        setup_call = sm_cls.return_value.setup_collectors.call_args
    observation_callback = setup_call.args[1]
    return manager, collector, observation_callback


def test_debug_message_still_emitted_when_debug_enabled():
    logger = MagicMock()
    logger.is_enabled_for.return_value = True
    _, collector, observation_callback = _make_manager(logger)

    observation_callback(_FakeMsg(), collector=collector)

    logger.is_enabled_for.assert_called_with(LoggingSeverity.DEBUG)
    logger.debug.assert_called_once()
    assert "fake" in logger.debug.call_args.args[0]
    assert collector.update_count == 1


def test_debug_message_not_built_when_debug_disabled():
    logger = MagicMock()
    logger.is_enabled_for.return_value = False
    _, collector, observation_callback = _make_manager(logger)

    observation_callback(_FakeMsg(), collector=collector)

    logger.is_enabled_for.assert_called_with(LoggingSeverity.DEBUG)
    logger.debug.assert_not_called()
    # The update itself must still happen — only the log message is skipped.
    assert collector.update_count == 1
