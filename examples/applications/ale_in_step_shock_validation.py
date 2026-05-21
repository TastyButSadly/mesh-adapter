from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ShockMetrics:
    cells: int
    adaptive: bool
    steps: int
    final_time: float
    shock_position: float
    shock_position_error: float
    shock_speed: float
    shock_speed_error: float
    front_width_10_90: float
    l1_error_exact: float
    l1_error_fine: float | None
    mass_error: float
    bounds_violation: float
    min_dx: float
    min_dx_near_shock: float
    mean_dx: float
    near_shock_dx_ratio: float
    max_relative_cfl: float
    max_mesh_cfl: float
    min_applied_relaxation: float


@dataclass
class ShockResult:
    x: np.ndarray
    u: np.ndarray
    metrics: ShockMetrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate direct ALE-in-step mesh motion on a 1D Burgers shock.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ale_in_step_shock_validation"))
    parser.add_argument("--coarse-cells", type=int, default=120)
    parser.add_argument("--fine-cells", type=int, default=960)
    parser.add_argument("--final-time", type=float, default=0.4)
    args = parser.parse_args()

    report = run_validation(args.output_dir, coarse_cells=args.coarse_cells, fine_cells=args.fine_cells, final_time=args.final_time)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


def run_validation(output_dir: Path, *, coarse_cells: int, fine_cells: int, final_time: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gcl = run_gcl_check()
    fixed = run_burgers_shock(cells=coarse_cells, final_time=final_time, adaptive=False)
    adaptive = run_burgers_shock(cells=coarse_cells, final_time=final_time, adaptive=True)
    fine = run_burgers_shock(cells=fine_cells, final_time=final_time, adaptive=False)

    fixed.metrics.l1_error_fine = _l1_between_piecewise_constant(fixed.x, fixed.u, fine.x, fine.u)
    adaptive.metrics.l1_error_fine = _l1_between_piecewise_constant(adaptive.x, adaptive.u, fine.x, fine.u)

    metrics = {
        "gcl": gcl,
        "fixed_coarse": asdict(fixed.metrics),
        "adaptive_every_step": asdict(adaptive.metrics),
        "fixed_fine_reference": asdict(fine.metrics),
    }
    metrics["summary"] = {
        "adaptation_frequency": "every step",
        "front_width_ratio_adaptive_to_fixed": adaptive.metrics.front_width_10_90 / fixed.metrics.front_width_10_90,
        "l1_exact_ratio_adaptive_to_fixed": adaptive.metrics.l1_error_exact / fixed.metrics.l1_error_exact,
        "l1_fine_ratio_adaptive_to_fixed": adaptive.metrics.l1_error_fine / fixed.metrics.l1_error_fine,
        "shock_speed_exact": 0.5,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    _write_plot(output_dir / "shock_profiles.png", fixed, adaptive, fine)
    return metrics


def run_gcl_check(*, cells: int = 80, steps: int = 5, value: float = 0.73, amplitude: float = 0.01) -> dict[str, float]:
    initial_x = np.linspace(0.0, 1.0, cells + 1)
    x = initial_x.copy()
    u_ale = np.full(cells, value)
    u_naive = np.full(cells, value)
    dt = 1.0 / steps

    for step in range(steps):
        phase = 2.0 * np.pi * (step + 1) / steps
        target = initial_x + amplitude * np.sin(np.pi * initial_x) * np.sin(phase)
        u_ale = _ale_burgers_step(x, u_ale, target, dt, u_left=value, u_right=value)
        u_naive = _naive_burgers_step_without_mesh_flux(x, u_naive, target, dt, u_left=value, u_right=value)
        x = target

    return {
        "constant_state": value,
        "ale_max_abs_error": float(np.max(np.abs(u_ale - value))),
        "naive_without_mesh_flux_max_abs_error": float(np.max(np.abs(u_naive - value))),
        "ale_mass_error": float(abs(np.sum(np.diff(x) * u_ale) - value)),
        "naive_without_mesh_flux_mass_error": float(abs(np.sum(np.diff(x) * u_naive) - value)),
    }


def run_burgers_shock(
    *,
    cells: int,
    final_time: float,
    adaptive: bool,
    shock_initial_position: float = 0.25,
    cfl: float = 0.35,
) -> ShockResult:
    x = np.linspace(0.0, 1.0, cells + 1)
    centers = _cell_centers(x)
    u = _exact_shock(centers, 0.0, shock_initial_position)
    t = 0.0
    steps = 0
    max_relative_cfl = 0.0
    max_mesh_cfl = 0.0
    min_applied_relaxation = 1.0

    while t < final_time - 1e-14:
        dt = min(cfl * np.min(np.diff(x)), final_time - t)
        target = x.copy()
        relative_cfl = 0.0
        mesh_cfl = 0.0
        applied_relaxation = 1.0
        if adaptive:
            raw_target = _monitor_equidistributed_target(x, u)
            target, applied_relaxation, relative_cfl, mesh_cfl = _limited_target(x, u, raw_target, dt)
        u = _ale_burgers_step(x, u, target, dt, u_left=1.0, u_right=0.0)
        x = target
        t += dt
        steps += 1
        max_relative_cfl = max(max_relative_cfl, relative_cfl)
        max_mesh_cfl = max(max_mesh_cfl, mesh_cfl)
        min_applied_relaxation = min(min_applied_relaxation, applied_relaxation)

    metrics = _shock_metrics(
        x,
        u,
        cells=cells,
        adaptive=adaptive,
        steps=steps,
        final_time=t,
        shock_initial_position=shock_initial_position,
        max_relative_cfl=max_relative_cfl,
        max_mesh_cfl=max_mesh_cfl,
        min_applied_relaxation=min_applied_relaxation,
    )
    return ShockResult(x=x, u=u, metrics=metrics)


def _ale_burgers_step(x: np.ndarray, u: np.ndarray, target: np.ndarray, dt: float, *, u_left: float, u_right: float) -> np.ndarray:
    w = (target - x) / dt
    u_l = np.concatenate(([u_left], u))
    u_r = np.concatenate((u, [u_right]))
    g_l = _burgers_flux(u_l) - w * u_l
    g_r = _burgers_flux(u_r) - w * u_r
    speed = np.maximum(np.abs(u_l - w), np.abs(u_r - w))
    numerical_flux = 0.5 * (g_l + g_r) - 0.5 * speed * (u_r - u_l)
    content = np.diff(x) * u - dt * (numerical_flux[1:] - numerical_flux[:-1])
    return content / np.diff(target)


def _naive_burgers_step_without_mesh_flux(
    x: np.ndarray,
    u: np.ndarray,
    target: np.ndarray,
    dt: float,
    *,
    u_left: float,
    u_right: float,
) -> np.ndarray:
    u_l = np.concatenate(([u_left], u))
    u_r = np.concatenate((u, [u_right]))
    numerical_flux = 0.5 * (_burgers_flux(u_l) + _burgers_flux(u_r)) - 0.5 * np.maximum(np.abs(u_l), np.abs(u_r)) * (u_r - u_l)
    content = np.diff(x) * u - dt * (numerical_flux[1:] - numerical_flux[:-1])
    return content / np.diff(target)


def _monitor_equidistributed_target(
    x: np.ndarray,
    u: np.ndarray,
    *,
    monitor_alpha: float = 25.0,
    target_relaxation: float = 0.5,
) -> np.ndarray:
    jumps = np.zeros_like(u)
    adjacent_jump = np.abs(np.diff(u))
    jumps[:-1] += adjacent_jump
    jumps[1:] += adjacent_jump
    if float(np.max(jumps)) > 0.0:
        jumps = jumps / np.max(jumps)
    density = 1.0 + monitor_alpha * _smooth(jumps)
    mass = density * np.diff(x)
    cumulative = np.concatenate(([0.0], np.cumsum(mass)))
    target_mass = np.linspace(0.0, cumulative[-1], len(x))

    raw = np.empty_like(x)
    raw[0] = x[0]
    raw[-1] = x[-1]
    for node, quota in enumerate(target_mass[1:-1], start=1):
        cell = np.searchsorted(cumulative, quota, side="right") - 1
        cell = int(np.clip(cell, 0, len(mass) - 1))
        fraction = 0.0 if mass[cell] == 0.0 else (quota - cumulative[cell]) / mass[cell]
        raw[node] = x[cell] + fraction * (x[cell + 1] - x[cell])
    return x + target_relaxation * (raw - x)


def _limited_target(
    x: np.ndarray,
    u: np.ndarray,
    raw_target: np.ndarray,
    dt: float,
    *,
    cfl_limit: float = 0.45,
    max_mesh_cfl: float = 0.5,
    min_cell_fraction: float = 0.12,
) -> tuple[np.ndarray, float, float, float]:
    delta = raw_target - x
    base_min_dx = float(np.min(np.diff(x)))
    relaxation = 1.0
    for _ in range(30):
        target = x + relaxation * delta
        new_dx = np.diff(target)
        if float(np.min(new_dx)) <= min_cell_fraction * base_min_dx:
            relaxation *= 0.5
            continue
        w = (target - x) / dt
        u_l = np.concatenate(([1.0], u))
        u_r = np.concatenate((u, [0.0]))
        relative_speed = np.maximum(np.abs(u_l - w), np.abs(u_r - w))
        relative_cfl = float(dt * np.max(relative_speed) / np.min(new_dx))
        mesh_cfl = float(np.max(np.abs(target - x)) / np.min(new_dx))
        if relative_cfl <= cfl_limit and mesh_cfl <= max_mesh_cfl:
            return target, relaxation, relative_cfl, mesh_cfl
        relaxation *= 0.5
    return x.copy(), 0.0, 0.0, 0.0


def _shock_metrics(
    x: np.ndarray,
    u: np.ndarray,
    *,
    cells: int,
    adaptive: bool,
    steps: int,
    final_time: float,
    shock_initial_position: float,
    max_relative_cfl: float,
    max_mesh_cfl: float,
    min_applied_relaxation: float,
) -> ShockMetrics:
    centers = _cell_centers(x)
    dx = np.diff(x)
    exact_front = shock_initial_position + 0.5 * final_time
    shock_position = _crossing_position(x, u, 0.5)
    x_09 = _crossing_position(x, u, 0.9)
    x_01 = _crossing_position(x, u, 0.1)
    front_width = x_01 - x_09
    exact = _exact_shock(centers, final_time, shock_initial_position)
    near_shock = np.abs(centers - exact_front) < 0.03
    min_dx_near = float(np.min(dx[near_shock])) if bool(np.any(near_shock)) else float("nan")
    return ShockMetrics(
        cells=cells,
        adaptive=adaptive,
        steps=steps,
        final_time=float(final_time),
        shock_position=float(shock_position),
        shock_position_error=float(shock_position - exact_front),
        shock_speed=float((shock_position - shock_initial_position) / final_time),
        shock_speed_error=float((shock_position - shock_initial_position) / final_time - 0.5),
        front_width_10_90=float(front_width),
        l1_error_exact=float(np.sum(dx * np.abs(u - exact))),
        l1_error_fine=None,
        mass_error=float(np.sum(dx * u) - exact_front),
        bounds_violation=float(max(np.max(u) - 1.0, -np.min(u), 0.0)),
        min_dx=float(np.min(dx)),
        min_dx_near_shock=min_dx_near,
        mean_dx=float(np.mean(dx)),
        near_shock_dx_ratio=float(min_dx_near / np.mean(dx)),
        max_relative_cfl=float(max_relative_cfl),
        max_mesh_cfl=float(max_mesh_cfl),
        min_applied_relaxation=float(min_applied_relaxation),
    )


def _l1_between_piecewise_constant(x: np.ndarray, u: np.ndarray, ref_x: np.ndarray, ref_u: np.ndarray) -> float:
    centers = _cell_centers(x)
    ref_centers = _cell_centers(ref_x)
    ref_at_centers = np.interp(centers, ref_centers, ref_u)
    return float(np.sum(np.diff(x) * np.abs(u - ref_at_centers)))


def _write_plot(path: Path, fixed: ShockResult, adaptive: ShockResult, fine: ShockResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for result, label, linewidth in (
        (fixed, "coarse fixed", 1.5),
        (adaptive, "coarse adaptive every step", 1.8),
        (fine, "fine fixed reference", 1.0),
    ):
        centers = _cell_centers(result.x)
        plt.plot(centers, result.u, label=label, linewidth=linewidth)
    plt.axvline(0.25 + 0.5 * fixed.metrics.final_time, color="black", linestyle="--", linewidth=1.0, label="exact shock")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.xlim(0.36, 0.54)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _smooth(values: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = values.copy()
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for _ in range(passes):
        smoothed = np.convolve(np.pad(smoothed, (2, 2), mode="edge"), kernel, mode="valid")
    return smoothed


def _exact_shock(centers: np.ndarray, time: float, shock_initial_position: float) -> np.ndarray:
    return np.where(centers < shock_initial_position + 0.5 * time, 1.0, 0.0)


def _crossing_position(x: np.ndarray, u: np.ndarray, level: float) -> float:
    centers = _cell_centers(x)
    for cell in range(len(u) - 1):
        left = u[cell] - level
        right = u[cell + 1] - level
        if left == 0.0:
            return float(centers[cell])
        if left * right <= 0.0 and u[cell] != u[cell + 1]:
            fraction = (level - u[cell]) / (u[cell + 1] - u[cell])
            return float(centers[cell] + fraction * (centers[cell + 1] - centers[cell]))
    return float("nan")


def _cell_centers(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x[:-1] + x[1:])


def _burgers_flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * u


if __name__ == "__main__":
    main()
