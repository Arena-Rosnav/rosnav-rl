"""An agent's trained control rate is the lockstep step period under lockstep, else control_hz."""

from __future__ import annotations

import pytest
from arena_training.arena_rosnav_rl.cfg.arena_cfg import GeneralCfg

from rosnav_rl.inference.arena_inference_node import trained_control_rate


def test_free_running_training_ticked_at_control_hz():
    assert trained_control_rate(GeneralCfg(control_hz=25.0, lockstep=False, lockstep_step_seconds=0.5)) == 25.0


def test_lockstep_training_ticked_at_the_step_period():
    rate = trained_control_rate(GeneralCfg(control_hz=25.0, lockstep=True, lockstep_step_seconds=0.05))

    assert rate == pytest.approx(20.0)
