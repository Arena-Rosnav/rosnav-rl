"""Split-conformal calibration for the uncertainty-aware safety layer (§2.4, stretch).

Pure math, no ROS/torch dependency. Turns the deploy-time KL-surprise signal
``u_t = KL(q(z|o) || p(z))`` (see ``dreamer.py``'s ``_policy``, gated by
``behavior.expose_kl_surprise``) into a calibrated velocity-attenuation factor: fit a
per-driver conformal threshold on validation episodes, take the max across training
drivers as the deployed threshold (the robot cannot know the true driver at runtime),
then shrink commanded velocity smoothly once ``u_t`` exceeds it.

The calibration *run* itself (collecting scores over real validation episodes with a
trained checkpoint) is Phase-4-gated -- this module only provides the math and the
serialization format it will write to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


def fit_conformal_threshold(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal quantile: the k-th smallest calibration score, k = ceil((n+1)(1-alpha)).

    Standard split-conformal threshold (Angelopoulos & Bates, 2021): guarantees the true
    score exceeds the returned threshold with probability at most ``alpha`` on a fresh draw
    from the same distribution as ``scores``. ``k`` is capped at ``n`` (returns the sample
    max) when ``alpha`` is small enough that the finite-sample correction would ask for more
    than all available data.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("fit_conformal_threshold requires at least one calibration score")
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


@dataclass
class SafetyCalibration:
    """Serialized to ``safety_calibration.json`` next to the checkpoint (see ``scripts/calibrate_safety.py``, Phase 4)."""

    per_driver: dict[str, float]
    """Per-training-driver threshold, analysis only -- never used at deploy time."""
    deployed: float
    """max(per_driver.values()): the deployed robot can't condition on the true driver."""
    alpha: float
    ema_beta: float
    """u_t EMA smoothing factor used at inference (see dreamer.py/arena_inference_node.py)."""

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SafetyCalibration":
        with open(path) as f:
            return cls(**json.load(f))


def attenuation_factor(u: float, q_hat: float, lam: float = 1.0, gamma_min: float = 0.3) -> float:
    """Velocity scale gamma in [gamma_min, 1] from surprise u against calibrated threshold q_hat.

    gamma = 1 when u <= q_hat (within calibrated normal range); otherwise decreases linearly
    with the relative overshoot ``(u - q_hat) / q_hat``, clipped at ``gamma_min`` so the robot
    never fully stops from this signal alone.
    """
    if u <= q_hat:
        return 1.0
    if q_hat <= 0:
        # Degenerate calibration (zero-surprise training distribution): any positive
        # surprise is maximally out-of-distribution relative to it.
        return gamma_min
    gamma = 1.0 - lam * (u - q_hat) / q_hat
    return float(min(max(gamma, gamma_min), 1.0))
