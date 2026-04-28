from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import firedrake as fd

from examples.firedrake.sphere_flow_mesh import (
    CHANNEL_HALF_WIDTH,
    CHANNEL_LENGTH,
    INLET_MARKER,
    OUTLET_MARKER,
    SPHERE_MARKER,
    SPHERE_RADIUS,
    WALL_MARKER,
)

NU_VALUE = 0.004
U_MEAN = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake-side 3D sphere-flow setup with differentiable adaptation.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--monitor-kind", choices=("velocity-gradient", "vorticity-magnitude"), default="velocity-gradient")
    parser.add_argument("--monitor-scale", type=float, default=3.0)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=1.0)
    parser.add_argument("--adapter-steps", type=int, default=8)
    parser.add_argument("--adapter-lr", type=float, default=5.0e-4)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only"), default="regularized")
    parser.add_argument("--adapter-exchange-dir", type=Path, required=True)
    parser.add_argument("--adapter-poll-interval", type=float, default=0.02)
    args = parser.parse_args()

    paths = run_case(
        args.mesh,
        args.output_dir,
        dt=args.dt,
        steps=args.steps,
        adapt_every=args.adapt_every,
        save_every=args.save_every,
        monitor_kind=args.monitor_kind,
        monitor_scale=args.monitor_scale,
        adaptation_relaxation=args.adaptation_relaxation,
        max_grid_speed_limit=args.max_grid_speed,
        adapter_steps=args.adapter_steps,
        adapter_lr=args.adapter_lr,
        adapter_profile=args.adapter_profile,
        adapter_exchange_dir=args.adapter_exchange_dir,
        adapter_poll_interval=args.adapter_poll_interval,
    )
    print(f"series: {paths['series']}")
    print(f"metrics: {paths['metrics']}")


def run_case(
    mesh_path: Path,
    output_dir: Path,
    *,
    dt: float,
    steps: int,
    adapt_every: int,
    save_every: int,
    monitor_kind: str,
    monitor_scale: float,
    adaptation_relaxation: float,
    max_grid_speed_limit: float,
    adapter_steps: int,
    adapter_lr: float,
    adapter_profile: str,
    adapter_exchange_dir: Path,
    adapter_poll_interval: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_exchange_dir.mkdir(parents=True, exist_ok=True)
    mesh = fd.Mesh(str(mesh_path))
    init_coord = mesh.coordinates.copy(deepcopy=True).dat.data_ro.copy()

    state = _create_solver_state(mesh, dt)
    tetrahedra = state.Q.cell_node_list
    initial_signed_volume = _signed_tetra_volume(init_coord, tetrahedra)
    characteristic_cell_size = float(np.abs(initial_signed_volume).mean() ** (1.0 / 3.0))
    vtk = fd.VTKFile(str(output_dir / "sphere_flow_diff_adapter.pvd"))
    metrics: dict[str, object] = {
        "case": "sphere_flow_diff_adapter",
        "mesh": str(mesh_path),
        "num_vertices": int(mesh.num_vertices()),
        "num_cells": int(mesh.num_cells()),
        "characteristic_cell_size": characteristic_cell_size,
        "initial_mesh_quality": _tetra_quality(init_coord, tetrahedra, initial_signed_volume),
        "dt": dt,
        "steps": steps,
        "adapt_every": adapt_every,
        "save_every": save_every,
        "monitor_kind": monitor_kind,
        "monitor_frame": "current",
        "monitor_scale": monitor_scale,
        "adaptation_relaxation": adaptation_relaxation,
        "max_grid_speed_limit": max_grid_speed_limit,
        "adapter_steps": adapter_steps,
        "adapter_lr": adapter_lr,
        "adapter_profile": adapter_profile,
        "adapter_exchange_dir": str(adapter_exchange_dir),
        "adapter_poll_interval": adapter_poll_interval,
        "reynolds_number": U_MEAN * (2.0 * SPHERE_RADIUS) / NU_VALUE,
        "adaptations": [],
        "timings_s": {"solve": 0.0, "monitor": 0.0, "adapter": 0.0},
    }

    monitor_val = fd.Function(state.Q, name="monitor")
    _write_state(vtk, state, init_coord, time_value=0.0, monitor=monitor_val)
    max_grid_speed = 0.0
    t = 0.0

    for step in range(1, steps + 1):
        solve_start = time.perf_counter()
        _solve_step(state)
        metrics["timings_s"]["solve"] += time.perf_counter() - solve_start
        t += dt

        if step % adapt_every == 0:
            current_coord = mesh.coordinates.dat.data_ro.copy()

            monitor_start = time.perf_counter()
            monitor_val = _sphere_monitor(mesh, state.u_next, monitor_kind=monitor_kind, monitor_scale=monitor_scale)
            metrics["timings_s"]["monitor"] += time.perf_counter() - monitor_start
            monitor_data = monitor_val.dat.data_ro.copy()

            adapter_start = time.perf_counter()
            raw_target_coord, adapter_info = _diff_adapter_coordinates(
                mesh,
                monitor_val,
                adapter_steps=adapter_steps,
                adapter_lr=adapter_lr,
                adapter_profile=adapter_profile,
                adapter_exchange_dir=adapter_exchange_dir,
                adapter_poll_interval=adapter_poll_interval,
                request_index=len(metrics["adaptations"]),
            )
            adapter_elapsed = time.perf_counter() - adapter_start
            metrics["timings_s"]["adapter"] += adapter_elapsed

            raw_quality = _tetra_quality(raw_target_coord, tetrahedra, initial_signed_volume)
            delta_coord = raw_target_coord - current_coord
            raw_grid_speed = float(np.abs(delta_coord).max() / dt)
            applied_relaxation = adaptation_relaxation
            if raw_grid_speed > 0.0:
                applied_relaxation = min(applied_relaxation, max_grid_speed_limit / raw_grid_speed)
            target_coord = current_coord + applied_relaxation * delta_coord
            relaxed_quality = _tetra_quality(target_coord, tetrahedra, initial_signed_volume)
            if relaxed_quality["orientation_flips"] > 0:
                applied_relaxation = 0.0
                target_coord = current_coord
                relaxed_quality = _tetra_quality(target_coord, tetrahedra, initial_signed_volume)

            mesh.coordinates.dat.data[:] = target_coord
            state.u_grid.dat.data[:] = (target_coord - current_coord) / dt
            applied_grid_speed = float(np.abs(state.u_grid.dat.data_ro).max())
            max_grid_speed = max(max_grid_speed, applied_grid_speed)
            state.u_now.assign(state.u_next)
            state.p_now.assign(state.p_next)

            metrics["adaptations"].append(
                {
                    "step": step,
                    "time": float(t),
                    "adapter_loss_initial": adapter_info["initial_loss"],
                    "adapter_loss_final": adapter_info["final_loss"],
                    "adapter_steps_completed": adapter_info["steps_completed"],
                    "adapter_early_stopped": adapter_info["early_stopped"],
                    "adapter_s": adapter_elapsed,
                    "adapter_service_s": adapter_info["service_s"],
                    "adapter_exchange_overhead_s": max(0.0, adapter_elapsed - adapter_info["service_s"]),
                    "adapter_response_wait_s": adapter_info["response_wait_s"],
                    "monitor_min": float(monitor_data.min()),
                    "monitor_max": float(monitor_data.max()),
                    "monitor_mean": float(monitor_data.mean()),
                    "requested_relaxation": adaptation_relaxation,
                    "applied_relaxation": float(applied_relaxation),
                    "raw_grid_speed": raw_grid_speed,
                    "applied_grid_speed": applied_grid_speed,
                    "max_raw_step_displacement": float(np.linalg.norm(raw_target_coord - current_coord, axis=1).max()),
                    "max_relaxed_step_displacement": float(np.linalg.norm(target_coord - current_coord, axis=1).max()),
                    "max_target_displacement_from_initial": float(np.linalg.norm(target_coord - init_coord, axis=1).max()),
                    "max_target_displacement_from_initial_over_h": float(
                        np.linalg.norm(target_coord - init_coord, axis=1).max() / characteristic_cell_size
                    ),
                    "raw_target_mesh_quality": raw_quality,
                    "relaxed_target_mesh_quality": relaxed_quality,
                }
            )
        else:
            state.u_now.assign(state.u_next)
            state.p_now.assign(state.p_next)
            state.u_grid.dat.data[:] = 0.0

        if step % save_every == 0 or step == steps:
            print(f"step {step}/{steps}, t={t:.3f}, adaptations={len(metrics['adaptations'])}", flush=True)
            _write_state(vtk, state, init_coord, time_value=t, monitor=monitor_val)

    displacement = np.linalg.norm(mesh.coordinates.dat.data_ro - init_coord, axis=1)
    _update_vorticity_outputs(state)
    metrics["final_time"] = float(t)
    metrics["max_grid_speed"] = max_grid_speed
    metrics["max_displacement"] = float(displacement.max())
    metrics["mean_displacement"] = float(displacement.mean())
    metrics["max_displacement_over_h"] = float(displacement.max() / characteristic_cell_size)
    metrics["final_mesh_quality"] = _tetra_quality(mesh.coordinates.dat.data_ro, tetrahedra, initial_signed_volume)
    metrics["min_vorticity_magnitude"] = float(state.vorticity_magnitude.dat.data_ro.min())
    metrics["max_vorticity_magnitude"] = float(state.vorticity_magnitude.dat.data_ro.max())

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"series": output_dir / "sphere_flow_diff_adapter.pvd", "metrics": metrics_path}


def _create_solver_state(mesh, dt: float):
    state = argparse.Namespace()
    state.mesh = mesh
    state.nu = fd.Constant(NU_VALUE)
    state.k = fd.Constant(dt)
    state.V = fd.VectorFunctionSpace(mesh, "CG", 2)
    state.Q = fd.FunctionSpace(mesh, "CG", 1)
    state.W = fd.VectorFunctionSpace(mesh, "CG", 1)
    state.u_grid = fd.Function(mesh.coordinates.function_space(), name="gridvelocity")
    state.u_now = fd.Function(state.V, name="velocity")
    state.u_next = fd.Function(state.V, name="velocity")
    state.u_star = fd.Function(state.V, name="velocity_star")
    state.p_now = fd.Function(state.Q, name="pressure")
    state.p_next = fd.Function(state.Q, name="pressure")
    state.vorticity = fd.Function(state.W, name="vorticity")
    state.vorticity_magnitude = fd.Function(state.Q, name="vorticity_magnitude")
    state.monitor_out = fd.Function(state.Q, name="monitor")
    state.displacement = fd.Function(state.Q, name="mesh_displacement")
    state.grid_speed = fd.Function(state.Q, name="grid_speed")

    u = fd.TrialFunction(state.V)
    v = fd.TestFunction(state.V)
    p = fd.TrialFunction(state.Q)
    q = fd.TestFunction(state.Q)
    n = fd.FacetNormal(mesh)
    _x, y, z = fd.SpatialCoordinate(mesh)
    f = fd.Constant((0.0, 0.0, 0.0))
    u_mid = 0.5 * (state.u_now + u)
    u_adv = state.u_now - state.u_grid
    inlet_speed = 2.25 * U_MEAN * (1.0 - (y / CHANNEL_HALF_WIDTH) ** 2) * (1.0 - (z / CHANNEL_HALF_WIDTH) ** 2)

    def sigma(velocity, pressure):
        return 2 * state.nu * fd.sym(fd.nabla_grad(velocity)) - pressure * fd.Identity(len(velocity))

    bcu = [
        fd.DirichletBC(state.V, fd.Constant((0.0, 0.0, 0.0)), (WALL_MARKER, SPHERE_MARKER)),
        fd.DirichletBC(state.V, (inlet_speed, 0.0, 0.0), INLET_MARKER),
    ]
    bcp = [fd.DirichletBC(state.Q, fd.Constant(0.0), OUTLET_MARKER)]

    F1 = (
        fd.inner((u - state.u_now) / state.k, v) * fd.dx
        + fd.inner(fd.dot(u_adv, fd.nabla_grad(u_mid)), v) * fd.dx
        + fd.inner(sigma(u_mid, state.p_now), fd.sym(fd.nabla_grad(v))) * fd.dx
        + fd.inner(state.p_now * n, v) * fd.ds
        - fd.inner(state.nu * fd.dot(fd.nabla_grad(u_mid), n), v) * fd.ds
        - fd.inner(f, v) * fd.dx
    )
    a1, L1 = fd.system(F1)
    a2 = fd.inner(fd.nabla_grad(p), fd.nabla_grad(q)) * fd.dx
    L2 = fd.inner(fd.nabla_grad(state.p_now), fd.nabla_grad(q)) * fd.dx - (1 / state.k) * fd.inner(
        fd.div(state.u_star), q
    ) * fd.dx
    a3 = fd.inner(u, v) * fd.dx
    L3 = fd.inner(state.u_star, v) * fd.dx - state.k * fd.inner(
        fd.nabla_grad(state.p_next - state.p_now), v
    ) * fd.dx

    state.solve1 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a1, L1, state.u_star, bcs=bcu),
        solver_parameters={"ksp_type": "gmres", "pc_type": "sor"},
    )
    state.solve2 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a2, L2, state.p_next, bcs=bcp),
        solver_parameters={"ksp_type": "cg", "pc_type": "gamg"},
    )
    state.solve3 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a3, L3, state.u_next),
        solver_parameters={"ksp_type": "cg", "pc_type": "sor"},
    )
    return state


def _solve_step(state) -> None:
    state.solve1.solve()
    state.solve2.solve()
    state.solve3.solve()


def _sphere_monitor(mesh, velocity, *, monitor_kind: str, monitor_scale: float) -> fd.Function:
    function_space = fd.FunctionSpace(mesh, "CG", 1)
    grad_norm = fd.Function(function_space, name="monitor")
    if monitor_kind == "velocity-gradient":
        tensor_space = fd.TensorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.Function(tensor_space)
        uh_grad.interpolate(fd.grad(velocity))
        grad_norm.interpolate(sum(uh_grad[i, j] ** 2 for i in range(3) for j in range(3)))
    elif monitor_kind == "vorticity-magnitude":
        vorticity = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 1))
        vorticity.project(fd.curl(velocity))
        grad_norm.interpolate(fd.sqrt(fd.dot(vorticity, vorticity)))
    else:
        raise ValueError(f"Unknown monitor kind: {monitor_kind}")

    np.minimum(grad_norm.dat.data, 1.0e3, out=grad_norm.dat.data)
    np.maximum(grad_norm.dat.data, 0.0, out=grad_norm.dat.data)
    max_value = max(float(grad_norm.dat.data_ro.max()), 1.0e-12)
    grad_norm.dat.data[:] = grad_norm.dat.data_ro / max_value

    u = fd.TrialFunction(function_space)
    v = fd.TestFunction(function_space)
    dx_mean = float(mesh.cell_sizes.dat.data_ro.mean())
    k_smooth = 15 * dx_mean**2 / 4
    rhs = (monitor_scale * grad_norm) * v * fd.dx(domain=mesh)
    lhs = (k_smooth * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
    smoothed = fd.Function(function_space)
    fd.solve(
        lhs == rhs,
        smoothed,
        solver_parameters={"ksp_type": "cg", "pc_type": "none"},
        bcs=fd.DirichletBC(function_space, monitor_scale * grad_norm, "on_boundary"),
    )

    monitor = fd.Function(function_space, name="monitor")
    monitor.project(1.0 + smoothed)
    return monitor


def _diff_adapter_coordinates(
    mesh,
    monitor: fd.Function,
    *,
    adapter_steps: int,
    adapter_lr: float,
    adapter_profile: str,
    adapter_exchange_dir: Path,
    adapter_poll_interval: float,
    request_index: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    points_np = mesh.coordinates.dat.data_ro.copy()
    cells_np = monitor.function_space().cell_node_list.copy()
    monitor_np = monitor.dat.data_ro.copy()
    request_path = adapter_exchange_dir / f"request_{request_index:06d}.npz"
    response_path = adapter_exchange_dir / f"response_{request_index:06d}.npz"
    tmp_path = adapter_exchange_dir / f"request_{request_index:06d}.tmp"
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            points=points_np,
            cells=cells_np,
            monitor=monitor_np,
            steps=np.array(adapter_steps, dtype=np.int64),
            lr=np.array(adapter_lr, dtype=np.float64),
            profile=np.array(adapter_profile),
        )
    tmp_path.rename(request_path)

    deadline = time.monotonic() + 1800.0
    wait_start = time.perf_counter()
    sleep_interval = max(adapter_poll_interval, 0.001)
    while not response_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for adapter response: {response_path}")
        time.sleep(sleep_interval)
    response_wait_s = time.perf_counter() - wait_start

    data = np.load(response_path)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data["points"], {
        "initial_loss": float(data["initial_loss"]),
        "final_loss": float(data["final_loss"]),
        "steps_completed": int(data["steps_completed"]),
        "early_stopped": bool(data["early_stopped"]),
        "service_s": float(data["service_s"]) if "service_s" in data else float("nan"),
        "response_wait_s": response_wait_s,
    }


def _write_state(vtk, state, init_coord: np.ndarray, *, time_value: float, monitor: fd.Function | None) -> None:
    state.displacement.dat.data[:] = np.linalg.norm(state.mesh.coordinates.dat.data_ro - init_coord, axis=1)
    if monitor is not None:
        state.monitor_out.assign(monitor)
    _update_vorticity_outputs(state)
    state.grid_speed.interpolate(fd.sqrt(fd.dot(state.u_grid, state.u_grid)))
    vtk.write(
        state.u_now,
        state.p_now,
        state.vorticity_magnitude,
        state.monitor_out,
        state.displacement,
        state.grid_speed,
        time=time_value,
    )


def _update_vorticity_outputs(state) -> None:
    state.vorticity.project(fd.curl(state.u_now))
    state.vorticity_magnitude.interpolate(fd.sqrt(fd.dot(state.vorticity, state.vorticity)))


def _signed_tetra_volume(coordinates: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    p0 = coordinates[tetrahedra[:, 0]]
    p1 = coordinates[tetrahedra[:, 1]]
    p2 = coordinates[tetrahedra[:, 2]]
    p3 = coordinates[tetrahedra[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def _tetra_quality(
    coordinates: np.ndarray,
    tetrahedra: np.ndarray,
    reference_signed_volume: np.ndarray,
) -> dict[str, float | int]:
    signed_volume = _signed_tetra_volume(coordinates, tetrahedra)
    volume = np.abs(signed_volume)
    positive_volume = np.maximum(volume, 1.0e-16)
    orientation_flips = np.sign(signed_volume) != np.sign(reference_signed_volume)
    return {
        "min_signed_volume": float(signed_volume.min()),
        "min_volume": float(volume.min()),
        "max_volume": float(volume.max()),
        "volume_ratio": float(positive_volume.max() / positive_volume.min()),
        "orientation_flips": int(np.count_nonzero(orientation_flips)),
    }


if __name__ == "__main__":
    main()
