"""Tests for the split-conformal safety-layer math (Item 4b, §2.4 stretch)."""

import numpy as np
import pytest

from rosnav_rl.model.dreamerv3.safety import (
    SafetyCalibration,
    attenuation_factor,
    fit_conformal_threshold,
)


class TestFitConformalThreshold:
    def test_recovers_true_quantile_on_large_gaussian_sample(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=1000)
        q_hat = fit_conformal_threshold(scores, alpha=0.05)
        true_quantile = 1.6449  # scipy.stats.norm.ppf(0.95)
        assert abs(q_hat - true_quantile) < 0.2

    def test_finite_sample_correction_exact_for_tiny_n(self):
        # n=4, alpha=0.5 -> k = ceil((4+1)*0.5) = ceil(2.5) = 3 -> 3rd smallest of 4 scores.
        scores = np.array([4.0, 1.0, 3.0, 2.0])
        q_hat = fit_conformal_threshold(scores, alpha=0.5)
        assert q_hat == 3.0

    def test_k_capped_at_n_returns_sample_max(self):
        # alpha small enough that ceil((n+1)(1-alpha)) > n -> capped at n -> the max.
        scores = np.array([1.0, 2.0, 3.0])
        q_hat = fit_conformal_threshold(scores, alpha=0.01)
        assert q_hat == 3.0

    def test_raises_on_empty_scores(self):
        with pytest.raises(ValueError):
            fit_conformal_threshold(np.array([]), alpha=0.05)


class TestAttenuationFactor:
    def test_no_attenuation_at_or_below_threshold(self):
        assert attenuation_factor(u=0.5, q_hat=1.0) == 1.0
        assert attenuation_factor(u=1.0, q_hat=1.0) == 1.0

    def test_monotone_decreasing_above_threshold(self):
        q_hat = 1.0
        gammas = [attenuation_factor(u, q_hat) for u in (1.0, 1.5, 2.0, 3.0)]
        assert gammas == sorted(gammas, reverse=True)

    def test_clipped_at_gamma_min(self):
        gamma = attenuation_factor(u=1000.0, q_hat=1.0, gamma_min=0.3)
        assert gamma == 0.3

    def test_degenerate_zero_threshold_falls_back_to_gamma_min(self):
        assert attenuation_factor(u=0.0, q_hat=0.0) == 1.0
        assert attenuation_factor(u=0.1, q_hat=0.0, gamma_min=0.3) == 0.3


class TestSafetyCalibrationSerialization:
    def test_round_trips_through_json(self, tmp_path):
        calib = SafetyCalibration(
            per_driver={"orca": 1.2, "sfm": 1.5}, deployed=1.5, alpha=0.05, ema_beta=0.6
        )
        path = str(tmp_path / "safety_calibration.json")
        calib.to_json(path)
        loaded = SafetyCalibration.from_json(path)
        assert loaded == calib
