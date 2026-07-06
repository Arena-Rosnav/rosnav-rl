#!/usr/bin/env python3
"""Linear probe: does cSRSSM's ``b`` carry driver-regime identity? (Item 2, §4.3 diagnostic (i))

Multinomial logistic regression (a single ``torch.nn.Linear`` + cross-entropy -- sklearn is
not installed in ``/opt/venv``, so this avoids adding it as a dependency) trained to predict
the active pedestrian driver from ``b``, evaluated with 5-fold cross-validation. Phase-1 gate:
probe accuracy well above chance (``1 / num_drivers``) means ``b`` has not collapsed to the
prior; accuracy near chance is the collapse failure mode the persistence-floor gap
(``context_pred_floor``, Item 1) is meant to catch earlier and cheaper.

Usage:
    python scripts/probe_context.py --codes context_codes.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kfold_splits(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def _train_val_split(idx: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_val = max(1, int(len(idx) * val_frac))
    return idx[n_val:], idx[:n_val]


def train_probe(
    b_train: np.ndarray,
    y_train: np.ndarray,
    b_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 300,
    lr: float = 1e-2,
    patience: int = 20,
) -> tuple[nn.Linear, float]:
    """Train a single-linear-layer probe with early stopping on a held-out validation split."""
    X_train = torch.as_tensor(b_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    X_val = torch.as_tensor(b_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)

    probe = nn.Linear(X_train.shape[1], num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc, best_state, stall = -1.0, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(X_train), y_train_t)
        loss.backward()
        opt.step()

        with torch.no_grad():
            val_acc = (probe(X_val).argmax(-1) == y_val_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    probe.load_state_dict(best_state)
    return probe, best_val_acc


def probe_accuracy(
    b: np.ndarray, labels: np.ndarray, num_classes: int, k_folds: int = 5, seed: int = 0
) -> np.ndarray:
    """5-fold CV test accuracy of the b -> driver-ID linear probe. Returns one accuracy per fold."""
    accs = []
    for fold, (train_idx, test_idx) in enumerate(_kfold_splits(len(b), k_folds, seed=seed)):
        inner_train, inner_val = _train_val_split(train_idx, seed=seed + fold)
        probe, _ = train_probe(
            b[inner_train], labels[inner_train], b[inner_val], labels[inner_val], num_classes
        )
        with torch.no_grad():
            pred = probe(torch.as_tensor(b[test_idx], dtype=torch.float32)).argmax(-1).numpy()
        accs.append(float((pred == labels[test_idx]).mean()))
    return np.array(accs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True, help="output of scripts/dump_context_codes.py")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.codes, allow_pickle=True)
    b, driver_id = data["b"], data["driver_id"]
    classes, labels = np.unique(driver_id, return_inverse=True)

    accs = probe_accuracy(b, labels, num_classes=len(classes), k_folds=args.k_folds, seed=args.seed)
    chance = 1.0 / len(classes)
    print(f"drivers: {list(classes)}  chance level: {chance:.3f}")
    print(f"probe accuracy: {accs.mean():.3f} +/- {accs.std():.3f}  (folds: {accs.round(3).tolist()})")


if __name__ == "__main__":
    main()
