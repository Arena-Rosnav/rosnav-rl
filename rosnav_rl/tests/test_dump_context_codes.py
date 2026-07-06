"""Tests for scripts/dump_context_codes.py (Item 2 dump/window-reconstruction path).

No checkpoint needed: uses a random-init SocialContextEncoder and synthetic episode dicts
round-tripped through the real tools.save_episodes/load_episodes npz format. The
checkpoint-loading path (load_social_context_encoder) is exercised only by argument
plumbing elsewhere -- an actual checkpoint run is deferred to the Phase-2 A100 phase.
"""

import numpy as np
import pytest

from rosnav_rl.model.dreamerv3 import tools
from rosnav_rl.model.dreamerv3.cfg import SocialContextCfg
from rosnav_rl.model.dreamerv3.social.context import SocialContextEncoder
from scripts.dump_context_codes import (
    DRIVER_ID_KEY,
    dump_episode_dir,
    encode_b_mean,
    reconstruct_window,
)


def _make_encoder(b_dim=4, hidden=16, node_feat_dim=3, window=6):
    cfg = SocialContextCfg(enabled=True, b_dim=b_dim, hidden=hidden, window=window)
    return SocialContextEncoder(cfg=cfg, node_feat_dim=node_feat_dim, device="cpu")


def _make_episode(T=10, N=5, F=3, with_driver=True, driver="sfm"):
    peds = np.random.default_rng(0).normal(size=(T, N * (F + 1))).astype(np.float32)
    episode = {"reward": np.zeros(T, dtype=np.float32), "PedestrianNodeSetSpace": peds}
    if with_driver:
        episode[DRIVER_ID_KEY] = np.array([driver] * T)
    return episode


class TestReconstructWindow:
    def test_shapes(self):
        episode = _make_episode(T=10, N=5, F=3)
        feat, valid = reconstruct_window(episode, K=6, N=5, F=3)
        assert feat.shape == (6, 5, 3)
        assert valid.shape == (6, 5)

    def test_clips_to_episode_length_when_shorter_than_window(self):
        episode = _make_episode(T=4, N=5, F=3)
        feat, valid = reconstruct_window(episode, K=6, N=5, F=3)
        assert feat.shape == (4, 5, 3)

    def test_matches_models_py_slicing(self):
        # Feature/validity split is the last channel, first K steps -- same as
        # models.py:391-395 (`_peds_bt[:, :K_c].view(...)`, `[..., :F_c]`, `[..., -1]`).
        N, F = 2, 3
        peds = np.zeros((5, N * (F + 1)), dtype=np.float32)
        peds_view = peds.reshape(5, N, F + 1)
        peds_view[..., :F] = 1.0
        peds_view[..., -1] = 2.0
        episode = {"reward": np.zeros(5), "PedestrianNodeSetSpace": peds}

        feat, valid = reconstruct_window(episode, K=3, N=N, F=F)
        assert np.all(feat == 1.0)
        assert np.all(valid == 2.0)


class TestEncodeBMean:
    def test_returns_b_dim_vector(self):
        enc = _make_encoder(b_dim=4, node_feat_dim=3)
        feat = np.random.default_rng(0).normal(size=(6, 5, 3)).astype(np.float32)
        valid = np.ones((6, 5), dtype=np.float32)
        b_mean = encode_b_mean(enc, feat, valid)
        assert b_mean.shape == (4,)


class TestDumpEpisodeDir:
    def test_raises_when_driver_id_missing(self, tmp_path):
        episode = _make_episode(with_driver=False)
        tools.save_episodes(tmp_path, {"ep0": episode})
        enc = _make_encoder(b_dim=4, node_feat_dim=3)

        with pytest.raises(RuntimeError, match="driver_id"):
            dump_episode_dir(tmp_path, enc, N=5, F=3, K=6, out_path=str(tmp_path / "out.npz"))

    def test_writes_one_row_per_episode(self, tmp_path):
        episodes = {
            "ep0": _make_episode(driver="sfm"),
            "ep1": _make_episode(driver="hsfm"),
        }
        tools.save_episodes(tmp_path, episodes)
        enc = _make_encoder(b_dim=4, node_feat_dim=3)

        out = dump_episode_dir(tmp_path, enc, N=5, F=3, K=6, out_path=str(tmp_path / "out.npz"))
        data = np.load(out, allow_pickle=True)

        assert data["b"].shape == (2, 4)
        assert set(data["driver_id"].tolist()) == {"sfm", "hsfm"}
        assert len(data["episode_id"]) == 2
