"""Tests for the threshold stop callbacks gated on the curriculum's last stage."""

from __future__ import annotations

import gymnasium as gym
import pytest
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rosnav_rl.utils.stable_baselines3.callbacks import (
    StopTrainingOnRewardThreshold,
    StopTrainingOnSuccessThreshold,
)


def _eval_callback(stop_cb) -> EvalCallback:
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    return EvalCallback(env, callback_on_new_best=stop_cb, verbose=0)


@pytest.mark.parametrize(
    ("success_rate", "last_stage", "continues"),
    [
        (0.0, True, True),
        (0.0, False, True),
        (0.95, False, True),
        (0.95, True, False),
    ],
)
def test_success_threshold_stops_only_above_threshold_on_last_stage(success_rate, last_stage, continues):
    stop_cb = StopTrainingOnSuccessThreshold(success_threshold=0.9, is_last_state_getter=lambda: last_stage)
    eval_cb = _eval_callback(stop_cb)
    eval_cb.last_success_rate = success_rate

    assert stop_cb._on_step() is continues


@pytest.mark.parametrize(
    ("best_reward", "last_stage", "continues"),
    [
        (-5.0, True, True),
        (25.0, False, True),
        (25.0, True, False),
    ],
)
def test_reward_threshold_stops_only_above_threshold_on_last_stage(best_reward, last_stage, continues):
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=20.0, is_last_state_getter=lambda: last_stage)
    eval_cb = _eval_callback(stop_cb)
    eval_cb.best_mean_reward = best_reward

    assert stop_cb._on_step() is continues


def test_without_curriculum_success_threshold_alone_decides():
    stop_cb = StopTrainingOnSuccessThreshold(success_threshold=0.9)
    eval_cb = _eval_callback(stop_cb)
    eval_cb.last_success_rate = 0.0

    assert stop_cb._on_step() is True
