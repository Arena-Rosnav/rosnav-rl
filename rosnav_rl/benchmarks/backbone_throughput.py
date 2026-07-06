#!/usr/bin/env python3
"""World-model backbone throughput benchmark: GRU vs TransformerCell vs TSSM.

Measures wall-clock observe() forward+backward throughput — the quantity behind the
proposal's parallel-training claim for the TSSM backbone (§4.3 "Backbone throughput").
Imagination is sequential for every backbone, so observe() is where the backbones differ.

Usage:
    python benchmarks/backbone_throughput.py                 # production dims (A100)
    python benchmarks/backbone_throughput.py --smoke         # tiny dims (CPU sanity run)
    python benchmarks/backbone_throughput.py --config path/to/config.yaml

Reports, per backbone: observe() steps/s (batch_size * batch_length / iteration time),
iteration latency, and peak CUDA memory when a GPU is available.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rosnav_rl.model.dreamerv3.networks import RSSM, TSSM  # noqa: E402

BACKBONES = ("gru", "transformer", "tssm")


def build_backbone(kind: str, dims: dict, device: str):
    """Build one dynamics backbone at the given dims. Mirrors models.py's selection."""
    common = dict(
        stoch=dims["stoch"],
        deter=dims["deter"],
        hidden=dims["hidden"],
        discrete=dims["discrete"],
        num_actions=dims["num_actions"],
        embed=dims["embed"],
        device=device,
        transformer_ctx_len=dims["ctx_len"],
        transformer_num_heads=dims["num_heads"],
    )
    if kind == "tssm":
        return TSSM(**common, tssm_num_layers=dims["tssm_num_layers"]).to(device)
    return RSSM(**common, cell_type=kind).to(device)


def load_dims(args) -> dict:
    if args.smoke:
        return dict(
            stoch=4, deter=32, hidden=16, discrete=4, num_actions=3, embed=16,
            ctx_len=8, num_heads=2, tssm_num_layers=1,
            batch_size=4, batch_length=16,
        )
    import yaml

    cfg = yaml.safe_load(open(args.config))
    model = cfg["model"]
    social = model.get("social", {})
    return dict(
        stoch=model["dyn_stoch"],
        deter=model["dyn_deter"],
        hidden=model["dyn_hidden"],
        discrete=model["dyn_discrete"],
        num_actions=args.num_actions,
        embed=args.embed,
        ctx_len=social.get("transformer_ctx_len", 64),
        num_heads=social.get("transformer_num_heads", 4),
        tssm_num_layers=social.get("tssm_num_layers", 2),
        batch_size=cfg["training"]["batch_size"] if "training" in cfg else 16,
        batch_length=cfg["training"]["batch_length"] if "training" in cfg else 64,
    )


def bench_one(kind: str, dims: dict, device: str, iters: int, warmup: int) -> dict:
    torch.manual_seed(0)
    dyn = build_backbone(kind, dims, device)
    B, T = dims["batch_size"], dims["batch_length"]
    embed = torch.randn(B, T, dims["embed"], device=device)
    action = torch.randn(B, T, dims["num_actions"], device=device)
    is_first = torch.zeros(B, T, device=device)
    is_first[:, 0] = 1.0

    opt = torch.optim.SGD(dyn.parameters(), lr=0.0)

    def step():
        opt.zero_grad(set_to_none=True)
        post, prior = dyn.observe(embed, action, is_first)
        loss, _, _, _ = dyn.kl_loss(post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1)
        loss.mean().backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    out = dict(backbone=kind, iter_ms=dt * 1e3, steps_per_s=B * T / dt)
    if device == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny dims, CPU sanity run")
    ap.add_argument(
        "--config",
        default=str(
            pathlib.Path(__file__).resolve().parents[3]
            / "configs" / "social_csrssm_config.yaml"
        ),
        help="training config to read production dims from",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-actions", type=int, default=2, dest="num_actions")
    ap.add_argument("--embed", type=int, default=512)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dims = load_dims(args)
    if args.smoke:
        args.iters, args.warmup = min(args.iters, 5), 1

    print(f"device={device}  B={dims['batch_size']} T={dims['batch_length']}  "
          f"stoch={dims['stoch']}x{dims['discrete']} deter={dims['deter']}")
    for kind in args.backbones:
        r = bench_one(kind, dims, device, args.iters, args.warmup)
        mem = f"  peak_mem={r['peak_mem_mb']:.0f}MB" if "peak_mem_mb" in r else ""
        print(f"{r['backbone']:<12} {r['iter_ms']:8.1f} ms/iter  "
              f"{r['steps_per_s']:10.0f} steps/s{mem}")


if __name__ == "__main__":
    main()
