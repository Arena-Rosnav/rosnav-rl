"""Offline Hamilton-Jacobi backward reachable tube solver for the crowd reachability channel."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import time

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
import yaml


@dataclasses.dataclass(frozen=True)
class RobotParams:
    """Rated speed, turn-rate bound and radius read from a robot's caps/mobile.yaml."""

    robot_id: str
    v_r: float
    w_max: float
    radius: float


def load_robot_params(caps_path: pathlib.Path) -> RobotParams:
    """Reads v_r (velocity_limits.linear max), w_max and radius from caps/mobile.yaml."""
    caps = yaml.safe_load(caps_path.read_text())
    linear = caps["velocity_limits"]["linear"]
    angular = caps["velocity_limits"]["angular"]
    return RobotParams(
        robot_id=caps_path.parent.parent.name,
        v_r=float(max(abs(linear[0]), abs(linear[1]))),
        w_max=float(max(abs(angular[0]), abs(angular[1]))),
        radius=float(caps["radius"]),
    )


def straight_line_closest_approach(x: np.ndarray, y: np.ndarray, theta: np.ndarray, v_p: float, v_r: float, horizon: float) -> np.ndarray:
    """Closed-form min distance to the origin under constant relative velocity, clamped to [0, horizon]."""
    ux = v_p - v_r * np.cos(theta)
    uy = -v_r * np.sin(theta)
    u_sq = ux * ux + uy * uy
    safe_u_sq = np.where(u_sq > 1e-12, u_sq, 1.0)
    t_star = np.where(u_sq > 1e-12, np.clip(-(x * ux + y * uy) / safe_u_sq, 0.0, horizon), 0.0)
    return np.hypot(x + ux * t_star, y + uy * t_star)


class RelativeUnicyclePedestrianSlice(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    """Robot unicycle (control omega) vs. a pedestrian walking straight at fixed speed v_p."""

    def __init__(self, v_p: float, v_r: float, w_max: float):
        self.v_p = v_p
        self.v_r = v_r
        control_space = hj.sets.Box(jnp.array([-w_max]), jnp.array([w_max]))
        disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__("max", "min", control_space, disturbance_space)

    def open_loop_dynamics(self, state: jax.Array, time: float) -> jax.Array:
        _, _, theta = state
        return jnp.array([self.v_p - self.v_r * jnp.cos(theta), -self.v_r * jnp.sin(theta), 0.0])

    def control_jacobian(self, state: jax.Array, time: float) -> jax.Array:
        return jnp.array([[0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state: jax.Array, time: float) -> jax.Array:
        return jnp.array([[0.0], [0.0], [0.0]])


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """A shared (x, y, theta) domain, sized to stay valid for every pedestrian-speed slice."""

    x_lo: float
    x_hi: float
    y_half: float
    n_theta: int
    dxy: float

    @property
    def shape(self) -> tuple[int, int, int]:
        nx = round((self.x_hi - self.x_lo) / self.dxy) + 1
        ny = round((2 * self.y_half) / self.dxy) + 1
        return nx, ny, self.n_theta

    def build(self) -> hj.Grid:
        domain = hj.sets.Box(
            np.array([self.x_lo, -self.y_half, -np.pi]),
            np.array([self.x_hi, self.y_half, np.pi]),
        )
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(domain, self.shape, periodic_dims=2)


def build_grid_spec(params: RobotParams, horizon: float, value_cap: float, pedestrian_radius_max: float, v_p_max: float, dxy: float, n_theta: int, margin: float) -> GridSpec:
    """Asymmetric domain box sized for the fastest pedestrian so every slice shares one grid."""
    oob_cap = value_cap + params.radius + pedestrian_radius_max
    return GridSpec(
        x_lo=-(oob_cap + (v_p_max + params.v_r) * horizon) - margin,
        x_hi=oob_cap + params.v_r * horizon + margin,
        y_half=oob_cap + params.v_r * horizon + margin,
        n_theta=n_theta,
        dxy=dxy,
    )


def solve_slice(dynamics: RelativeUnicyclePedestrianSlice, grid: hj.Grid, times: np.ndarray) -> np.ndarray:
    """Backward reachable tube values at each time in `times`, shape (len(times), *grid.shape)."""
    ell0 = jnp.linalg.norm(grid.states[..., :2], axis=-1)
    settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    return np.asarray(hj.solve(settings, dynamics, grid, jnp.asarray(times), ell0, progress_bar=False))


def solve_no_turn_slice(grid: hj.Grid, v_p: float, v_r: float, horizon: float) -> np.ndarray:
    """Final values of the omega_max=0 problem, solved once and shared by gates G1 and G2."""
    no_turn = RelativeUnicyclePedestrianSlice(v_p=v_p, v_r=v_r, w_max=0.0)
    return solve_slice(no_turn, grid, np.array([0.0, -horizon]))[-1]


def gate_g1_straight_line(no_turn_values: np.ndarray, grid: hj.Grid, v_p: float, v_r: float, horizon: float, tol: float, near_target_band: float, near_target_tol: float, decision_band_upper: float) -> dict:
    """The omega_max=0 solve must match the closed-form straight-line distance away from the point target."""
    x, y, theta = (np.asarray(grid.states[..., i]) for i in range(3))
    closed_form = straight_line_closest_approach(x, y, theta, v_p, v_r, horizon)
    diff = np.abs(no_turn_values - closed_form)
    near_target = closed_form < near_target_band
    decision_band = (~near_target) & (closed_form <= decision_band_upper)
    near_target_diff = float(np.max(diff[near_target])) if near_target.any() else 0.0
    decision_band_diff = float(np.max(diff[decision_band])) if decision_band.any() else 0.0
    if decision_band_diff > tol:
        raise AssertionError(f"gate G1 failed for v_p={v_p}: max diff in the decision band = {decision_band_diff} > {tol}")
    if near_target_diff > near_target_tol:
        raise AssertionError(f"gate G1 failed for v_p={v_p}: max diff near the target = {near_target_diff} > {near_target_tol}")
    return {"decision_band": decision_band_diff, "near_target": near_target_diff}


def gate_g2_turning_helps(values_with_turn: np.ndarray, no_turn_values: np.ndarray, v_p: float, tol: float) -> float:
    """Being able to turn must never lower the value versus the omega=0 case; returns the min margin."""
    margin = float(np.min(values_with_turn - no_turn_values))
    if margin < -tol:
        raise AssertionError(f"gate G2 failed for v_p={v_p}: turning lowered the value by {-margin} > {tol}")
    return margin


def gate_g3_bounds(values: np.ndarray, grid: hj.Grid, v_p: float, v_r: float, horizon: float, tol: float, near_target_band: float, near_target_tol: float, decision_band_upper: float) -> dict:
    """V must sit between the straight-line value and ell, and above ell - (v_p + v_r) * horizon."""
    x, y, theta = (np.asarray(grid.states[..., i]) for i in range(3))
    ell = np.hypot(x, y)
    straight_line = straight_line_closest_approach(x, y, theta, v_p, v_r, horizon)
    speed_bound = ell - (v_p + v_r) * horizon
    lower_bound = np.maximum(straight_line, speed_bound)
    upper_violation = float(np.max(values - ell))
    lower_violation = lower_bound - values
    near_target = straight_line < near_target_band
    decision_band = (~near_target) & (straight_line <= decision_band_upper)
    near_target_violation = float(np.max(lower_violation[near_target])) if near_target.any() else 0.0
    decision_band_violation = float(np.max(lower_violation[decision_band])) if decision_band.any() else 0.0
    if upper_violation > tol or decision_band_violation > tol:
        raise AssertionError(f"gate G3 failed for v_p={v_p}: upper violation {upper_violation}, lower violation {decision_band_violation} (tol {tol})")
    if near_target_violation > near_target_tol:
        raise AssertionError(f"gate G3 failed for v_p={v_p}: near-target lower violation {near_target_violation} > {near_target_tol}")
    return {"upper_violation": upper_violation, "decision_band_lower": decision_band_violation, "near_target_lower": near_target_violation}


def gate_g4_monotonic(values_history: np.ndarray, tol: float) -> float:
    """The tube can only grow with horizon, so V must be non-increasing along the time axis."""
    violation = float(np.max(np.diff(values_history, axis=0)))
    if violation > tol:
        raise AssertionError(f"gate G4 failed: value increased by {violation} going to a longer horizon (tol {tol})")
    return violation


def gate_g5_mirror_symmetry(values: np.ndarray, tol: float) -> float:
    """V(x, y, theta) must equal V(x, -y, -theta) for a symmetric turn-rate bound."""
    n_theta = values.shape[2]
    theta_mirror = (-np.arange(n_theta)) % n_theta
    mirrored = values[:, ::-1, :][:, :, theta_mirror]
    diff = float(np.max(np.abs(values - mirrored)))
    if diff > tol:
        raise AssertionError(f"gate G5 failed: max mirror asymmetry {diff} > {tol}")
    return diff


def run_core_gates(values_history: np.ndarray, grid: hj.Grid, v_p: float, v_r: float, horizon: float, tol: float, near_target_band: float, near_target_tol: float, decision_band_upper: float) -> dict:
    """Runs gates G1-G5 for one pedestrian-speed slice; raises on the first one that fails."""
    final_values = values_history[-1]
    no_turn_values = solve_no_turn_slice(grid, v_p, v_r, horizon)
    return {
        "g1_straight_line": gate_g1_straight_line(no_turn_values, grid, v_p, v_r, horizon, tol, near_target_band, near_target_tol, decision_band_upper),
        "g2_turning_margin": gate_g2_turning_helps(final_values, no_turn_values, v_p, tol),
        "g3_bounds": gate_g3_bounds(final_values, grid, v_p, v_r, horizon, tol, near_target_band, near_target_tol, decision_band_upper),
        "g4_monotonic_violation": gate_g4_monotonic(values_history, tol),
        "g5_mirror_diff": gate_g5_mirror_symmetry(final_values, tol),
    }


def gate_g6_air3d_crosscheck(params: RobotParams, v_p: float, horizon: float, value_cap: float, pedestrian_radius_max: float, dxy: float, n_psi: int, margin: float, tol: float) -> dict:
    """Cross-checks the pedestrian-frame solve against hj.systems.Air3d solved independently in the robot frame."""
    oob_cap = value_cap + params.radius + pedestrian_radius_max
    half = oob_cap + (v_p + params.v_r) * horizon + margin
    shape = (round(2 * half / dxy) + 1, round(2 * half / dxy) + 1, n_psi)
    robot_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-half, -half, -np.pi]), np.array([half, half, np.pi])), shape, periodic_dims=2)
    air3d = hj.systems.Air3d(evader_speed=params.v_r, pursuer_speed=v_p, evader_max_turn_rate=params.w_max, pursuer_max_turn_rate=0.0, control_mode="max", disturbance_mode="min")
    ell0 = jnp.linalg.norm(robot_grid.states[..., :2], axis=-1)
    settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    robot_values = jnp.asarray(hj.step(settings, air3d, robot_grid, 0.0, ell0, -horizon, progress_bar=False))

    ped_grid_spec = build_grid_spec(params, horizon, value_cap, pedestrian_radius_max, v_p, dxy, n_psi, margin)
    ped_grid = ped_grid_spec.build()
    ped_dynamics = RelativeUnicyclePedestrianSlice(v_p=v_p, v_r=params.v_r, w_max=params.w_max)
    ped_values = solve_slice(ped_dynamics, ped_grid, np.array([0.0, -horizon]))[-1]

    x, y, theta = (np.asarray(ped_grid.states[..., i]) for i in range(3))
    query = jnp.stack(
        [jnp.asarray(x) * jnp.cos(jnp.asarray(theta)) + jnp.asarray(y) * jnp.sin(jnp.asarray(theta)), -jnp.asarray(x) * jnp.sin(jnp.asarray(theta)) + jnp.asarray(y) * jnp.cos(jnp.asarray(theta)), -jnp.asarray(theta)],
        axis=-1,
    ).reshape(-1, 3)
    robot_values_at_ped_points = np.asarray(jax.vmap(lambda s: robot_grid.interpolate(robot_values, s))(query)).reshape(x.shape)

    ell = np.hypot(x, y)
    decision = (ell >= 0.2) & (ell <= oob_cap)
    diff = np.abs(ped_values - robot_values_at_ped_points)
    decision_diff = float(np.max(diff[decision])) if decision.any() else 0.0
    if decision_diff > tol:
        raise AssertionError(f"gate G6 failed for v_p={v_p}: max diff vs Air3d cross-check = {decision_diff} > {tol}")
    return {"decision_band_diff_vs_air3d": decision_diff, "robot_frame_grid_shape": list(shape)}


def gate_g8_convergence(params: RobotParams, v_p: float, horizon: float, value_cap: float, pedestrian_radius_max: float, v_p_max: float, dxy_coarse: float, dxy_fine: float, n_theta_coarse: int, n_theta_fine: int, margin: float, tol: float) -> dict:
    """Solves the same slice at two resolutions; a small coarse-vs-fine gap means the coarser grid has converged."""

    def solved_at(dxy: float, n_theta: int) -> tuple[hj.Grid, np.ndarray]:
        spec = build_grid_spec(params, horizon, value_cap, pedestrian_radius_max, v_p_max, dxy, n_theta, margin)
        grid = spec.build()
        dynamics = RelativeUnicyclePedestrianSlice(v_p=v_p, v_r=params.v_r, w_max=params.w_max)
        return grid, solve_slice(dynamics, grid, np.array([0.0, -horizon]))[-1]

    coarse_grid, coarse_values = solved_at(dxy_coarse, n_theta_coarse)
    fine_grid, fine_values = solved_at(dxy_fine, n_theta_fine)

    coarse_values_j = jnp.asarray(coarse_values)
    fine_states = fine_grid.states.reshape(-1, 3)
    coarse_at_fine_points = np.asarray(jax.vmap(lambda s: coarse_grid.interpolate(coarse_values_j, s))(fine_states)).reshape(fine_grid.shape)

    x, y, _ = (np.asarray(fine_grid.states[..., i]) for i in range(3))
    ell = np.hypot(x, y)
    oob_cap = value_cap + params.radius + pedestrian_radius_max
    decision = (ell >= 0.2) & (ell <= oob_cap)
    diff = np.abs(coarse_at_fine_points - fine_values)
    decision_diff = float(np.max(diff[decision])) if decision.any() else 0.0
    if decision_diff > tol:
        raise AssertionError(f"gate G8 failed for v_p={v_p}: coarse-vs-fine decision-band diff = {decision_diff} > {tol}")
    return {"decision_band_diff_coarse_vs_fine": decision_diff, "coarse_shape": list(coarse_grid.shape), "fine_shape": list(fine_grid.shape)}


def pack_table(values_by_vp: np.ndarray, v_p_values: np.ndarray, grid_spec: GridSpec, params: RobotParams, horizon: float, value_cap: float, pedestrian_radius_max: float, gate_reports: list[dict], out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Clips to [0, oob_cap], quantizes to uint16 and writes <robot_id>_brt_v1.npz plus its metadata JSON."""
    oob_cap = value_cap + params.radius + pedestrian_radius_max
    clipped = np.clip(values_by_vp, 0.0, oob_cap)
    quantized = np.round(clipped / oob_cap * np.iinfo(np.uint16).max).astype(np.uint16)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{params.robot_id}_brt_v1.npz"
    np.savez_compressed(
        npz_path,
        values=quantized,
        x=np.linspace(grid_spec.x_lo, grid_spec.x_hi, grid_spec.shape[0]),
        y=np.linspace(-grid_spec.y_half, grid_spec.y_half, grid_spec.shape[1]),
        theta=np.linspace(-np.pi, np.pi, grid_spec.shape[2], endpoint=False),
        pedestrian_speed=v_p_values,
    )
    sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    json_path = out_dir / f"{params.robot_id}_brt_v1.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "robot_id": params.robot_id,
                "horizon_s": horizon,
                "robot_rated_speed_mps": params.v_r,
                "robot_max_yaw_rate_radps": params.w_max,
                "robot_radius_m": params.radius,
                "pedestrian_radius_assumed_max_m": pedestrian_radius_max,
                "value_cap_m": value_cap,
                "oob_cap_m": oob_cap,
                "axes": ["pedestrian_speed", "x", "y", "theta"],
                "periodic_axes": ["theta"],
                "grid_shape": list(quantized.shape),
                "dtype": "uint16",
                "sign_convention": "value_is_the_undilated_min_clearance_over_the_horizon;_subtract_contact_radius_at_lookup_time",
                "solver_version": hj.__version__,
                "generator_commit": _git_commit(),
                "array_sha256": sha256,
                "gate_reports": gate_reports,
            },
            indent=2,
        )
    )
    return npz_path, json_path


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).parent, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="Needs the reachability dependency group: uv sync --group reachability")
    parser.add_argument("--robot-caps", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--value-cap", type=float, default=1.5)
    parser.add_argument("--pedestrian-radius-max", type=float, default=0.40)
    parser.add_argument("--v-p-max", type=float, default=2.5)
    parser.add_argument("--v-p-step", type=float, default=0.25)
    parser.add_argument("--dxy", type=float, default=0.1)
    parser.add_argument("--n-theta", type=int, default=72)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--dt-snapshot", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=0.05, help="max abs error tolerated away from the point-target singularity")
    parser.add_argument("--near-target-band", type=float, default=0.2, help="closed-form distance below which the point-target discretization artifact (see gate G1) applies")
    parser.add_argument("--near-target-tol", type=float, default=0.3, help="looser tolerance inside --near-target-band; does not affect any realistic contact-radius decision")
    parser.add_argument("--validate-only", action="store_true", help="run gates G6 (Air3d cross-check) and G8 (resolution convergence) for one slice instead of solving the full table")
    parser.add_argument("--validate-v-p", type=float, default=1.0)
    parser.add_argument("--gate6-dxy", type=float, default=0.15)
    parser.add_argument("--gate6-n-psi", type=int, default=48)
    parser.add_argument("--gate6-tol", type=float, default=0.2, help="two independently discretized grids (robot frame vs pedestrian frame) compound their own near-target dissipation")
    parser.add_argument("--gate8-dxy-fine", type=float, default=0.06)
    parser.add_argument("--gate8-n-theta-fine", type=int, default=96)
    parser.add_argument("--gate8-tol", type=float, default=0.08, help="max decision-band gap tolerated between the production grid (--dxy/--n-theta) and this finer reference; measures whether --dxy=0.1 is fine enough, not a bug check")
    args = parser.parse_args()

    params = load_robot_params(args.robot_caps)

    if args.validate_only:
        t0 = time.monotonic()
        g6 = gate_g6_air3d_crosscheck(params, args.validate_v_p, args.horizon, args.value_cap, args.pedestrian_radius_max, args.gate6_dxy, args.gate6_n_psi, args.margin, args.gate6_tol)
        print(f"G6 (Air3d cross-check) v_p={args.validate_v_p}: {g6}  [{time.monotonic() - t0:.1f}s]")
        t0 = time.monotonic()
        g8 = gate_g8_convergence(params, args.validate_v_p, args.horizon, args.value_cap, args.pedestrian_radius_max, args.v_p_max, args.dxy, args.gate8_dxy_fine, args.n_theta, args.gate8_n_theta_fine, args.margin, args.gate8_tol)
        print(f"G8 (convergence) v_p={args.validate_v_p}: {g8}  [{time.monotonic() - t0:.1f}s]")
        return

    grid_spec = build_grid_spec(params, args.horizon, args.value_cap, args.pedestrian_radius_max, args.v_p_max, args.dxy, args.n_theta, args.margin)
    grid = grid_spec.build()
    oob_cap = args.value_cap + params.radius + args.pedestrian_radius_max
    n_snapshots = round(args.horizon / args.dt_snapshot) + 1
    times = np.linspace(0.0, -args.horizon, n_snapshots)
    v_p_values = np.arange(0.0, args.v_p_max + args.v_p_step / 2, args.v_p_step)

    print(f"robot={params.robot_id} v_r={params.v_r} w_max={params.w_max} radius={params.radius}")
    print(f"grid shape={grid_spec.shape} domain=x[{grid_spec.x_lo:.2f},{grid_spec.x_hi:.2f}] y=+-{grid_spec.y_half:.2f}")
    print(f"pedestrian speeds: {v_p_values}")

    values_by_vp = np.empty((len(v_p_values), *grid_spec.shape), dtype=np.float32)
    gate_reports = []
    for i, v_p in enumerate(v_p_values):
        t0 = time.monotonic()
        dynamics = RelativeUnicyclePedestrianSlice(v_p=float(v_p), v_r=params.v_r, w_max=params.w_max)
        history = solve_slice(dynamics, grid, times)
        report = run_core_gates(history, grid, float(v_p), params.v_r, args.horizon, args.tol, args.near_target_band, args.near_target_tol, oob_cap)
        values_by_vp[i] = history[-1]
        gate_reports.append({"v_p": float(v_p), **report})
        print(f"v_p={v_p:.2f} solved in {time.monotonic() - t0:.1f}s gates={report}")

    if not np.isfinite(values_by_vp).all():
        raise AssertionError("solved table contains non-finite values")

    npz_path, json_path = pack_table(values_by_vp, v_p_values, grid_spec, params, args.horizon, args.value_cap, args.pedestrian_radius_max, gate_reports, args.out_dir)
    print(f"wrote {npz_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
