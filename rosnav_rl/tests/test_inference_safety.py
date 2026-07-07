"""Tests for the deploy-time velocity-attenuation wiring (Item 4d, §2.4 stretch).

Exercises `ArenaInferenceNode._safety_gamma` / `_load_safety_calibration` /
`_on_scenario_reset` directly against a bare instance (`object.__new__`) with a stubbed
`agent.model` — building a full rclpy Node is unnecessary since these methods only read
instance attributes (plus, for `_load_safety_calibration`'s path lookup, `self.node`).

`get_safety_signal()` is the capability method backends override to expose a safety
signal (see `RL_Model.get_safety_signal`, `DreamerV3Model.get_safety_signal`). Backends
that never override it (all SB3 backends) must make `_load_safety_calibration` fail
fast rather than silently produce gamma=1.0 forever once enabled.
"""

import types

import pytest

from rosnav_rl.inference.arena_inference_node import ArenaInferenceNode
from rosnav_rl.model.dreamerv3.safety import SafetyCalibration
from rosnav_rl.model.model import RL_Model


class _UnsupportedModel:
    """Mimics a backend that never overrides get_safety_signal (e.g. any SB3 model)."""

    get_safety_signal = RL_Model.get_safety_signal


class _SupportedModel:
    """Mimics DreamerV3Model: overrides get_safety_signal with a mutable signal."""

    def __init__(self, kl_surprise=None):
        self.kl_surprise = kl_surprise

    def get_safety_signal(self):
        return self.kl_surprise


def _make_node(model, calib=None, lam=1.0, gamma_min=0.3, enabled=True):
    node = object.__new__(ArenaInferenceNode)
    node._safety_enabled = enabled
    node._safety_calib = calib
    node._safety_lam = lam
    node._safety_gamma_min = gamma_min
    node._u_ema = None
    node._last_speed = 0.0
    node.agent = types.SimpleNamespace(model=model, reset=lambda: None)
    return node


def _stub_string_param(node, value):
    node.node = types.SimpleNamespace(
        get_parameter=lambda name: types.SimpleNamespace(
            get_parameter_value=lambda: types.SimpleNamespace(string_value=value)
        )
    )


def _default_calib(ema_beta=0.6, deployed=1.0):
    return SafetyCalibration(per_driver={}, deployed=deployed, alpha=0.05, ema_beta=ema_beta)


class TestLoadSafetyCalibration:
    def test_raises_for_backend_that_never_overrides_get_safety_signal(self):
        node = _make_node(model=_UnsupportedModel())
        with pytest.raises(RuntimeError, match="does not override get_safety_signal"):
            node._load_safety_calibration()

    def test_raises_when_supported_backend_missing_calibration_path(self):
        node = _make_node(model=_SupportedModel())
        _stub_string_param(node, "")
        with pytest.raises(RuntimeError, match="safety_calibration_path"):
            node._load_safety_calibration()

    def test_loads_calibration_json_for_supported_backend(self, tmp_path):
        calib = _default_calib(deployed=1.5)
        path = str(tmp_path / "safety_calibration.json")
        calib.to_json(path)

        node = _make_node(model=_SupportedModel())
        node.logger = types.SimpleNamespace(info=lambda *a, **kw: None)
        _stub_string_param(node, path)

        loaded = node._load_safety_calibration()
        assert loaded == calib


class TestSafetyGamma:
    def test_disabled_returns_one_even_with_high_surprise(self):
        node = _make_node(model=_SupportedModel(100.0), calib=_default_calib(), enabled=False)
        assert node._safety_gamma() == 1.0

    def test_missing_signal_returns_one(self):
        node = _make_node(model=_SupportedModel(None), calib=_default_calib())
        assert node._safety_gamma() == 1.0

    def test_below_threshold_no_attenuation(self):
        node = _make_node(model=_SupportedModel(0.5), calib=_default_calib(deployed=1.0))
        assert node._safety_gamma() == 1.0

    def test_above_threshold_attenuates_between_gamma_min_and_one(self):
        node = _make_node(
            model=_SupportedModel(3.0), calib=_default_calib(deployed=1.0, ema_beta=1.0), gamma_min=0.3
        )
        gamma = node._safety_gamma()
        assert 0.3 <= gamma < 1.0

    def test_ema_state_persists_and_climbs_toward_new_surprise(self):
        model = _SupportedModel(1.0)
        node = _make_node(model=model, calib=_default_calib(deployed=1.0, ema_beta=0.5))
        node._safety_gamma()
        first_ema = node._u_ema

        model.kl_surprise = 5.0
        node._safety_gamma()
        second_ema = node._u_ema

        assert first_ema == 1.0
        assert first_ema < second_ema < 5.0

    def test_reset_clears_ema_state(self):
        node = _make_node(model=_SupportedModel(5.0), calib=_default_calib())
        node._safety_gamma()
        assert node._u_ema is not None

        node._on_scenario_reset(None)

        assert node._u_ema is None
