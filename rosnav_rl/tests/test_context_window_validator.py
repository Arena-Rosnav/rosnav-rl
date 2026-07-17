"""cSRSSM window/batch_length cross-field validation (DreamerV3Cfg).

Guards the silent-disable failure mode: pred_scale>0 but window>=batch_length means
context_pred_loss has no out-of-window targets and never fires (models.py `T_c > K_c`
gate), collapsing b. The validator turns that into a config error.
"""

import pytest
from pydantic import ValidationError

from rosnav_rl.model.dreamerv3.cfg import (
    DreamerV3Cfg,
    ModelCfg,
    SocialCfg,
    SocialContextCfg,
    TrainingCfg,
)


def _cfg(window: int, batch_length: int, pred_scale: float, enabled: bool = True):
    return DreamerV3Cfg(
        model=ModelCfg(
            social=SocialCfg(
                enabled=True,
                context=SocialContextCfg(
                    enabled=enabled, window=window, pred_scale=pred_scale
                ),
            )
        ),
        training=TrainingCfg(batch_length=batch_length),
    )


def test_window_ge_batch_length_with_pred_raises():
    with pytest.raises(ValidationError, match="out-of-window"):
        _cfg(window=16, batch_length=16, pred_scale=0.1)


def test_window_lt_batch_length_ok():
    cfg = _cfg(window=16, batch_length=64, pred_scale=0.1)
    assert cfg.model.social.context.window == 16


def test_pred_scale_zero_disables_check():
    # No prediction loss requested -> window==batch_length is harmless.
    cfg = _cfg(window=16, batch_length=16, pred_scale=0.0)
    assert cfg.training.batch_length == 16


def test_context_disabled_disables_check():
    cfg = _cfg(window=16, batch_length=16, pred_scale=0.1, enabled=False)
    assert cfg.model.social.context.enabled is False
