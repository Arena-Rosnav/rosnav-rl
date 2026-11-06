"""NumPy-only runtime lookup for the Hamilton-Jacobi reachability tube table, with no JAX at runtime."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import pathlib

import numpy as np

_UINT16_MAX = float(np.iinfo(np.uint16).max)


# Duplicated from tools/solve_reachability_tube.py, which needs JAX and so cannot be imported at runtime.
def straight_line_closest_approach(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray, v_p: np.ndarray, v_r: float, horizon: float
) -> np.ndarray:
    """Closed-form min distance to the origin under constant relative velocity, clamped to [0, horizon]."""
    ux = v_p - v_r * np.cos(theta)
    uy = -v_r * np.sin(theta)
    u_sq = ux * ux + uy * uy
    safe_u_sq = np.where(u_sq > 1e-12, u_sq, 1.0)
    t_star = np.where(u_sq > 1e-12, np.clip(-(x * ux + y * uy) / safe_u_sq, 0.0, horizon), 0.0)
    return np.hypot(x + ux * t_star, y + uy * t_star)


class ReachabilityTube:
    """A loaded backward-reachable-tube table for one robot, queryable by (x, y, theta, pedestrian_speed)."""

    def __init__(
        self,
        values: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        pedestrian_speed: np.ndarray,
        metadata: dict,
    ):
        self._values = values.astype(np.float32) / _UINT16_MAX * metadata["oob_cap_m"]
        self._x = x
        self._y = y
        self._theta = theta
        self._v_p = pedestrian_speed
        self.metadata = metadata
        self.oob_cap = float(metadata["oob_cap_m"])
        self.horizon_s = float(metadata["horizon_s"])
        self.robot_rated_speed_mps = float(metadata["robot_rated_speed_mps"])
        self.robot_max_yaw_rate_radps = float(metadata["robot_max_yaw_rate_radps"])

    @classmethod
    def from_npz(cls, npz_path: pathlib.Path, json_path: pathlib.Path) -> ReachabilityTube:
        """Loads a table from an explicit pair of files, verifying the npz against its own sha256."""
        metadata = json.loads(json_path.read_text())
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if digest != metadata["array_sha256"]:
            raise ValueError(f"{npz_path}: sha256 {digest} does not match metadata {metadata['array_sha256']}")
        data = np.load(npz_path)
        return cls(data["values"], data["x"], data["y"], data["theta"], data["pedestrian_speed"], metadata)

    @classmethod
    def from_package(cls, robot_id: str) -> ReachabilityTube:
        """Loads `<robot_id>_brt_v1.{npz,json}` from this package's bundled data directory."""
        data_dir = importlib.resources.files("rosnav_rl") / "observations" / "utils" / "data"
        npz_path = pathlib.Path(str(data_dir / f"{robot_id}_brt_v1.npz"))
        json_path = pathlib.Path(str(data_dir / f"{robot_id}_brt_v1.json"))
        if not npz_path.is_file():
            raise FileNotFoundError(f"No reachability table for robot_id={robot_id!r} at {npz_path}")
        return cls.from_npz(npz_path, json_path)

    def matches_robot(self, v_r: float, w_max: float, tol: float = 1e-3) -> bool:
        """Whether this table was generated for the given rated speed and turn-rate bound."""
        return abs(self.robot_rated_speed_mps - v_r) <= tol and abs(self.robot_max_yaw_rate_radps - w_max) <= tol

    def _axis_weights(
        self, coord: np.ndarray, axis: np.ndarray, periodic: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (lo_index, hi_index, weight_on_hi) for linear interpolation along one grid axis."""
        n = axis.shape[0]
        spacing = axis[1] - axis[0]
        position = (coord - axis[0]) / spacing
        if periodic:
            position = np.mod(position, n)
            lo = np.floor(position).astype(np.int64) % n
            hi = (lo + 1) % n
            weight_hi = position - np.floor(position)
        else:
            lo = np.clip(np.floor(position).astype(np.int64), 0, n - 2)
            hi = lo + 1
            weight_hi = np.clip(position - lo, 0.0, 1.0)
        return lo, hi, weight_hi

    def query_batch(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, pedestrian_speed: np.ndarray) -> np.ndarray:
        """Interpolated V0 without the contact radius; outside the box it reads oob_cap, beyond the table's speeds the straight-line bound."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        theta = np.mod(np.atleast_1d(np.asarray(theta, dtype=np.float64)) + np.pi, 2 * np.pi) - np.pi
        v_p = np.atleast_1d(np.asarray(pedestrian_speed, dtype=np.float64))
        if not (x.shape == y.shape == theta.shape == v_p.shape):
            raise ValueError("x, y, theta, pedestrian_speed must share one shape")

        in_box = (x >= self._x[0]) & (x <= self._x[-1]) & (y >= self._y[0]) & (y <= self._y[-1])
        in_v_p_range = v_p <= self._v_p[-1]

        result = np.full(x.shape, self.oob_cap, dtype=np.float32)
        fallback = in_box & ~in_v_p_range
        result[fallback] = straight_line_closest_approach(
            x[fallback], y[fallback], theta[fallback], v_p[fallback], self.robot_rated_speed_mps, self.horizon_s
        )

        interpolate = in_box & in_v_p_range
        if interpolate.any():
            x_lo, x_hi, wx = self._axis_weights(x[interpolate], self._x, periodic=False)
            y_lo, y_hi, wy = self._axis_weights(y[interpolate], self._y, periodic=False)
            t_lo, t_hi, wt = self._axis_weights(theta[interpolate], self._theta, periodic=True)
            v_clamped = np.clip(v_p[interpolate], self._v_p[0], self._v_p[-1])
            v_lo, v_hi, wv = self._axis_weights(v_clamped, self._v_p, periodic=False)

            acc = np.zeros(x_lo.shape, dtype=np.float64)
            for xi, xw in ((x_lo, 1 - wx), (x_hi, wx)):
                for yi, yw in ((y_lo, 1 - wy), (y_hi, wy)):
                    for ti, tw in ((t_lo, 1 - wt), (t_hi, wt)):
                        for vi, vw in ((v_lo, 1 - wv), (v_hi, wv)):
                            acc += xw * yw * tw * vw * self._values[vi, xi, yi, ti]
            result[interpolate] = acc.astype(np.float32)

        return result

    def value(
        self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, pedestrian_speed: np.ndarray, contact_radius: float
    ) -> np.ndarray:
        """V0 minus `contact_radius`; this, not query_batch, is the signed tube value to threshold at 0."""
        return self.query_batch(x, y, theta, pedestrian_speed) - contact_radius
