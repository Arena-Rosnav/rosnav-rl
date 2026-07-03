"""Tests for the Social-Dreamer config block and the augmented-feature dimensional contract (M0).

The key invariant ("firebreak") is that with ``social.enabled = False`` the augmented feature
width equals the baseline DreamerV3 feature width, so the baseline network stays byte-identical.
"""

import pytest

from rosnav_rl.model.dreamerv3 import cfg as dcfg
from rosnav_rl.model.dreamerv3.social import dims


@pytest.fixture
def model_cfg():
    """A default DreamerV3 model config (discrete dynamics, social disabled)."""
    return dcfg.ModelCfg()


class _Wrap:
    """Minimal stand-in exposing ``.model`` for the dims helpers."""

    def __init__(self, model):
        self.model = model


class TestSocialCfgDefaults:
    def test_disabled_by_default(self, model_cfg):
        assert model_cfg.social.enabled is False
        assert model_cfg.social.dali.enabled is False
        assert model_cfg.social.curriculum.enabled is False

    def test_core_dims(self, model_cfg):
        assert model_cfg.social.max_peds == 8
        assert model_cfg.social.node_feat_dim == 5
        assert model_cfg.social.gat.out_dim == 64
        assert model_cfg.social.dali.out_dim == 64

    def test_dali_window_fixed_at_2s(self, model_cfg):
        # K = 40 steps = 2 s at 20 Hz, sized to cover an SFM/HSFM avoidance cycle.
        assert model_cfg.social.dali.k_steps == 40

    def test_imagination_backprop_cap(self, model_cfg):
        assert model_cfg.social.dali.imag_backprop_steps == 5

    def test_curriculum_gates(self, model_cfg):
        assert model_cfg.social.curriculum.recon_mse_gate == pytest.approx(0.10)
        assert model_cfg.social.curriculum.probe_acc_gate == pytest.approx(0.70)


class TestDimsContract:
    def test_base_matches_get_feat_formula(self, model_cfg):
        m = model_cfg
        expected = (
            m.dyn_stoch * m.dyn_discrete + m.dyn_deter
            if m.dyn_discrete
            else m.dyn_stoch + m.dyn_deter
        )
        assert dims.base_feat_size(_Wrap(m)) == expected

    def test_firebreak_disabled_equals_base(self, model_cfg):
        w = _Wrap(model_cfg)
        assert dims.augmented_feat_size(w) == dims.base_feat_size(w)
        assert dims.social_context_size(w) == 0
        assert dims.dynamics_context_size(w) == 0

    def test_gat_only_adds_context(self, model_cfg):
        w = _Wrap(model_cfg)
        base = dims.base_feat_size(w)
        model_cfg.social.enabled = True
        assert dims.social_context_size(w) == 64
        assert dims.dynamics_context_size(w) == 0  # DALI still off
        assert dims.augmented_feat_size(w) == base + 64

    def test_gat_plus_dali_adds_both(self, model_cfg):
        w = _Wrap(model_cfg)
        base = dims.base_feat_size(w)
        model_cfg.social.enabled = True
        model_cfg.social.dali.enabled = True
        assert dims.augmented_feat_size(w) == base + 64 + 64

    def test_dali_requires_master_switch(self, model_cfg):
        # DALI contributes nothing unless the social master switch is also on.
        w = _Wrap(model_cfg)
        model_cfg.social.dali.enabled = True
        assert dims.dynamics_context_size(w) == 0
        assert dims.augmented_feat_size(w) == dims.base_feat_size(w)
