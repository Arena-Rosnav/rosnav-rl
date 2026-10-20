from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from rosnav_rl.model.stable_baselines3.policy.recurrent import (
    PackedObservations,
    PackedRecurrentDictRolloutBuffer,
    PackedRecurrentPPO,
)

N_ENVS = 4
N_STEPS = 128
BATCH_SIZE = 256


class ShortEpisodesEnv(gym.Env):
    """Dict observations with mostly short episodes and the odd long one."""

    def __init__(self, seed: int, image_shape: tuple[int, ...], image_dtype: type = np.float32):
        self.rng = np.random.default_rng(seed)
        high = 255 if image_dtype is np.uint8 else 1.0
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, high, image_shape, dtype=image_dtype),
                "vector": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.left = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {key: space.sample() for key, space in self.observation_space.spaces.items()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.observation_space.seed(int(self.rng.integers(1 << 31)))
        self.left = 300 if self.rng.random() < 0.1 else int(self.rng.integers(2, 12))
        return self._obs(), {}

    def step(self, action):
        self.left -= 1
        return self._obs(), float(self.rng.random()), self.left <= 0, False, {}


class RowCountingExtractor(CombinedExtractor):
    """CombinedExtractor that records the batch size of every forward pass in training mode."""

    rows: list[int]

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        self.rows = []

    def forward(self, observations):
        if self.training:
            self.rows.append(next(iter(observations.values())).shape[0])
        return super().forward(observations)


def _model(algorithm: type[RecurrentPPO], image_shape: tuple[int, ...], **policy_kwargs):
    env = DummyVecEnv([lambda i=i: ShortEpisodesEnv(i, image_shape) for i in range(N_ENVS)])
    return algorithm(
        "MultiInputLstmPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=2,
        seed=0,
        device="cpu",
        policy_kwargs=dict(lstm_hidden_size=16, n_lstm_layers=2, enable_critic_lstm=True, **policy_kwargs),
    )


def test_packed_updates_extract_features_on_batch_size_rows_at_most():
    packed = _model(PackedRecurrentPPO, (1, 16, 16), features_extractor_class=RowCountingExtractor)
    padded = _model(RecurrentPPO, (1, 16, 16), features_extractor_class=RowCountingExtractor)

    packed.learn(3 * N_ENVS * N_STEPS)
    padded.learn(3 * N_ENVS * N_STEPS)

    packed_rows = packed.policy.features_extractor.rows
    padded_rows = padded.policy.features_extractor.rows
    assert isinstance(packed.rollout_buffer, PackedRecurrentDictRolloutBuffer)
    assert len(packed_rows) == 3 * 2 * (N_ENVS * N_STEPS // BATCH_SIZE)
    assert max(packed_rows) <= BATCH_SIZE
    assert max(padded_rows) > 2 * BATCH_SIZE


def _minibatches(buffer: RecurrentDictRolloutBuffer):
    env_change = np.zeros((N_STEPS, N_ENVS))
    env_change[0, :] = 1.0
    env_change = buffer.swap_and_flatten(env_change)
    for split in (0, 37, 301):
        batch_inds = np.roll(np.arange(N_STEPS * N_ENVS), -split)[:BATCH_SIZE]
        yield (
            RecurrentDictRolloutBuffer._get_samples(buffer, batch_inds, env_change),
            buffer._get_samples(batch_inds, env_change),
        )


def test_packed_minibatch_evaluates_like_the_padded_one():
    model = _model(PackedRecurrentPPO, (1, 16, 16))
    model.learn(N_ENVS * N_STEPS)

    for padded, packed in _minibatches(model.rollout_buffer):
        assert isinstance(packed.observations, PackedObservations)
        mask = packed.mask > 1e-8
        assert int(mask.sum()) == BATCH_SIZE
        assert next(iter(packed.observations.values())).shape[0] == BATCH_SIZE
        assert next(iter(padded.observations.values())).shape[0] > BATCH_SIZE
        with th.no_grad():
            expected = model.policy.evaluate_actions(
                padded.observations, padded.actions, padded.lstm_states, padded.episode_starts
            )
            actual = model.policy.evaluate_actions(
                packed.observations, packed.actions, packed.lstm_states, packed.episode_starts
            )
        for want, got in zip(expected, actual, strict=True):
            th.testing.assert_close(got.flatten()[mask], want.flatten()[mask])


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs a CUDA device")
def test_packed_update_peak_memory_stays_bounded_across_updates():
    peaks: list[int] = []

    class PeakRecordingPPO(PackedRecurrentPPO):
        def train(self) -> None:
            th.cuda.synchronize()
            th.cuda.reset_peak_memory_stats()
            super().train()
            th.cuda.synchronize()
            peaks.append(th.cuda.max_memory_allocated())

    env = DummyVecEnv([lambda i=i: ShortEpisodesEnv(i, (3, 80, 80), np.uint8) for i in range(N_ENVS)])
    model = PeakRecordingPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=2,
        seed=0,
        device="cuda",
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=2, enable_critic_lstm=True),
    )
    model.learn(6 * N_ENVS * N_STEPS)

    assert len(peaks) == 6
    assert max(peaks) <= 1.2 * min(peaks)
