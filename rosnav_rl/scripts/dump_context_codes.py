#!/usr/bin/env python3
"""Dump per-episode cSRSSM context codes ``b`` for the linear-probe diagnostic (Item 2).

For each episode in an eval-rollout directory, reconstructs the K-step pedestrian window
(same slicing as ``models.py``'s training-time window, see ``reconstruct_window``), runs
it through a trained ``SocialContextEncoder`` to get ``b_mean``, and writes one row per
episode to an ``.npz`` file with columns ``[b, driver_id, episode_id]``.

Prerequisite: episodes must carry a ``driver_id`` key (the active humansim local-planner
name, e.g. "orca"/"sfm"/"hsfm") -- as of this writing that is NOT logged anywhere in the
DreamerV3 episode store (``tools.py``'s ``save_episodes``/``load_episodes`` only carry
observation/action/reward keys). Wiring that through is a separate prerequisite task
(trace the active humansim local-planner through the env wrapper into the episode dict);
this script deliberately raises rather than silently fabricating a driver label.

Usage:
    python scripts/dump_context_codes.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --episodes-dir /path/to/eval_episodes \\
        --config configs/social_csrssm_config.yaml \\
        --out context_codes.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rosnav_rl.model.dreamerv3 import tools  # noqa: E402
from rosnav_rl.model.dreamerv3.cfg import SocialContextCfg  # noqa: E402
from rosnav_rl.model.dreamerv3.social.context import SocialContextEncoder  # noqa: E402

DRIVER_ID_KEY = "driver_id"
CHECKPOINT_PREFIX = "_wm._social_context."


def reconstruct_window(
    episode: dict, K: int, N: int, F: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the (feat, valid) ped window from one episode's raw log.

    Mirrors ``models.py``'s training-time slicing exactly (``_peds_bt[:, :K_c]`` split into
    feature/validity via the last channel): the first ``K`` steps of the episode's
    ``PedestrianNodeSetSpace`` log, clipped to the episode length if shorter.

    Returns:
        ``feat``: ``(K, N, F)``, ``valid``: ``(K, N)``.
    """
    peds = episode["PedestrianNodeSetSpace"]  # (T, N*(F+1))
    K = min(K, peds.shape[0])
    peds_win = peds[:K].reshape(K, N, F + 1)
    return peds_win[..., :F], peds_win[..., -1]


def encode_b_mean(
    encoder: SocialContextEncoder, feat: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Run one episode's window through the encoder, return ``b_mean`` as a ``(b_dim,)`` array."""
    feat_t = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
    valid_t = torch.as_tensor(valid, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        b_mean, _ = encoder(feat_t, valid_t)
    return b_mean.squeeze(0).cpu().numpy()


def load_social_context_encoder(
    checkpoint_path: str, cfg: SocialContextCfg, node_feat_dim: int, device: str = "cpu"
) -> SocialContextEncoder:
    """Build a ``SocialContextEncoder`` and load its weights out of a full Dreamer checkpoint.

    Checkpoints are saved as ``{"agent_state_dict": <full Dreamer state_dict>, ...}``
    (``dreamerv3_model.py``'s ``save``/``load``); this pulls out just the
    ``_wm._social_context.*`` keys rather than reconstructing the whole agent.
    """
    encoder = SocialContextEncoder(cfg=cfg, node_feat_dim=node_feat_dim, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {
        k[len(CHECKPOINT_PREFIX) :]: v
        for k, v in checkpoint["agent_state_dict"].items()
        if k.startswith(CHECKPOINT_PREFIX)
    }
    if not state:
        raise RuntimeError(
            f"No '{CHECKPOINT_PREFIX}*' keys found in checkpoint {checkpoint_path} -- "
            "was model.social.context.enabled=true during training?"
        )
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def dump_episode_dir(
    episode_dir: str,
    encoder: SocialContextEncoder,
    N: int,
    F: int,
    K: int,
    out_path: str,
) -> str:
    """Write ``[b, driver_id, episode_id]`` for every episode in ``episode_dir`` to ``out_path``."""
    episodes = tools.load_episodes(episode_dir)
    if not episodes:
        raise RuntimeError(f"No episodes found under {episode_dir}")

    b_rows, driver_rows, id_rows = [], [], []
    for episode_id, episode in episodes.items():
        if DRIVER_ID_KEY not in episode:
            raise RuntimeError(
                f"Episode '{episode_id}' has no '{DRIVER_ID_KEY}' key -- driver identity "
                "isn't logged into episodes yet. Prerequisite: wire the active humansim "
                "local-planner name into the episode dict during eval rollouts before "
                "running this script (see this script's module docstring)."
            )
        feat, valid = reconstruct_window(episode, K, N, F)
        b_rows.append(encode_b_mean(encoder, feat, valid))
        driver_rows.append(np.asarray(episode[DRIVER_ID_KEY]).reshape(-1)[0])
        id_rows.append(episode_id)

    np.savez(
        out_path,
        b=np.stack(b_rows),
        driver_id=np.array(driver_rows),
        episode_id=np.array(id_rows),
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--config", required=True, help="training config social.* block source")
    ap.add_argument("--out", default="context_codes.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import yaml

    social = yaml.safe_load(open(args.config))["model"]["social"]
    ctx_cfg = SocialContextCfg(
        enabled=True,
        b_dim=social["context"]["b_dim"],
        hidden=social["context"]["hidden"],
        window=social["context"]["window"],
    )
    encoder = load_social_context_encoder(
        args.checkpoint, ctx_cfg, social["node_feat_dim"], device=args.device
    )
    out = dump_episode_dir(
        args.episodes_dir,
        encoder,
        N=social["max_peds"],
        F=social["node_feat_dim"],
        K=social["context"]["window"],
        out_path=args.out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
