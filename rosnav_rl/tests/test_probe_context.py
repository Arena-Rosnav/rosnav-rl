"""Tests for the b -> driver-ID linear probe (Item 2, §4.3 diagnostic (i))."""

import numpy as np
import pytest
import torch

from scripts.probe_context import probe_accuracy, train_probe


def _make_blobs(n_per_class=40, num_classes=3, b_dim=4, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(num_classes, b_dim))
    b = np.concatenate(
        [centers[c] + rng.normal(scale=0.5, size=(n_per_class, b_dim)) for c in range(num_classes)]
    ).astype(np.float32)
    labels = np.repeat(np.arange(num_classes), n_per_class)
    return b, labels


class TestTrainProbe:
    def test_separable_blobs_fit_near_perfectly(self):
        torch.manual_seed(0)
        b, labels = _make_blobs()
        probe, val_acc = train_probe(b, labels, b, labels, num_classes=3)
        assert val_acc > 0.95


class TestProbeAccuracy:
    def test_informative_b_beats_chance(self):
        torch.manual_seed(0)
        num_classes = 3
        b, labels = _make_blobs(num_classes=num_classes)
        accs = probe_accuracy(b, labels, num_classes=num_classes, k_folds=5, seed=0)

        chance = 1.0 / num_classes
        assert accs.mean() > 2 * chance, f"probe should clearly beat chance ({chance}), got {accs.mean()}"

    def test_label_shuffled_b_is_near_chance(self):
        # Same b distribution, but labels shuffled -> no real b/driver relationship left.
        torch.manual_seed(0)
        num_classes = 3
        b, labels = _make_blobs(num_classes=num_classes)
        shuffled = np.random.default_rng(1).permutation(labels)

        accs = probe_accuracy(b, shuffled, num_classes=num_classes, k_folds=5, seed=0)

        chance = 1.0 / num_classes
        assert accs.mean() < chance + 0.2, (
            f"shuffled labels should sit near chance ({chance}), got {accs.mean()}"
        )
