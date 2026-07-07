"""Tests for the deploy-time velocity-attenuation wiring (Item 4d, §2.4 stretch).

Exercises `ArenaInferenceNode._safety_gamma` / `_on_scenario_reset` directly against a
bare instance (`object.__new__`) with a stubbed `agent.model.last_step_info` — building a
full rclpy Node is unnecessary since the safety layer only reads instance attributes and
never touches `self.node`.
"""

import types

from rosnav_rl.inference.arena_inference_node import ArenaInferenceNode
from rosnav_rl.model.dreamerv3.safety import SafetyCalibration


def _make_node(calib, lam=1.0, gamma_min=0.3, kl_surprise=None, enabled=True):
    node = object.__new__(ArenaInferenceNode)
    node._safety_enabled = enabled
    node._safety_calib = calib
    node._safety_lam = lam
    node._safety_gamma_min = gamma_min
    node._u_ema = None
    node._last_speed = 0.0
    last_step_info = {} if kl_surprise is None else {"kl_surprise": kl_surprise}
    node.agent = types.SimpleNamespace(
        model=types.SimpleNamespace(last_step_info=last_step_info),
        reset=lambda: None,
    )
    return node


def _default_calib(ema_beta=0.6, deployed=1.0):
    return SafetyCalibration(per_driver={}, deployed=deployed, alpha=0.05, ema_beta=ema_beta)


class TestSafetyGamma:
    def test_disabled_returns_one_even_with_high_surprise(self):
        node = _make_node(calib=_default_calib(), kl_surprise=100.0, enabled=False)
        assert node._safety_gamma() == 1.0

    def test_missing_kl_surprise_returns_one(self):
        node = _make_node(calib=_default_calib(), kl_surprise=None)
        assert node._safety_gamma() == 1.0

    def test_below_threshold_no_attenuation(self):
        node = _make_node(calib=_default_calib(deployed=1.0), kl_surprise=0.5)
        assert node._safety_gamma() == 1.0

    def test_above_threshold_attenuates_between_gamma_min_and_one(self):
        node = _make_node(calib=_default_calib(deployed=1.0, ema_beta=1.0), kl_surprise=3.0, gamma_min=0.3)
        gamma = node._safety_gamma()
        assert 0.3 <= gamma < 1.0

    def test_ema_state_persists_and_climbs_toward_new_surprise(self):
        node = _make_node(calib=_default_calib(deployed=1.0, ema_beta=0.5), kl_surprise=1.0)
        node._safety_gamma()
        first_ema = node._u_ema

        node.agent.model.last_step_info = {"kl_surprise": 5.0}
        node._safety_gamma()
        second_ema = node._u_ema

        assert first_ema == 1.0
        assert first_ema < second_ema < 5.0

    def test_reset_clears_ema_state(self):
        node = _make_node(calib=_default_calib(), kl_surprise=5.0)
        node._safety_gamma()
        assert node._u_ema is not None

        node._on_scenario_reset(None)

        assert node._u_ema is None
