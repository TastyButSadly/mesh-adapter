from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import firedrake as fd
import movement as mv

CHANNEL_HEIGHT = 0.41
CYLINDER_DIAMETER = 0.1
NU_VALUE = 0.001
INLET_MARKER = 10
OUTLET_MARKER = 11
WALL_MARKER = 12
CYLINDER_MARKER = 13


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake-side Navier-Stokes cylinder flow with MA adaptation.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--adapt-start", type=float, default=0.4)
    parser.add_argument("--adapt-interval", type=int, default=50)
    parser.add_argument("--motion-substeps", type=int, default=8)
    parser.add_argument("--write-stride", type=int, default=25)
    parser.add_argument("--mean-speed", type=float, default=1.0)
    parser.add_argument("--inlet-perturbation", type=float, default=0.02)
    parser.add_argument("--monitor-kind", choices=("gradient", "vorticity"), default="vorticity")
    parser.add_argument("--monitor-gain", type=float, default=6.0)
    parser.add_argument("--monitor-ceiling", type=float, default=1.0e3)
    parser.add_argument("--adapt-relaxation", type=float, default=0.5)
    parser.add_argument("--max-adapt-displacement", type=float, default=0.006)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()

    paths = run_case(
        args.mesh,
        args.output_dir,
        dt=args.dt,
        final_time=args.final_time,
        adapt_start=args.adapt_start,
        adapt_interval=args.adapt_interval,
        motion_substeps=args.motion_substeps,
        write_stride=args.write_stride,
        mean_speed=args.mean_speed,
        inlet_perturbation=args.inlet_perturbation,
        monitor_kind=args.monitor_kind,
        monitor_gain=args.monitor_gain,
        monitor_ceiling=args.monitor_ceiling,
        adapt_relaxation=args.adapt_relaxation,
        max_adapt_displacement=args.max_adapt_displacement,
        rtol=args.rtol,
        maxiter=args.maxiter,
    )
    print(f"series: {paths['series']}")
    print(f"metrics: {paths['metrics']}")


def run_case(
    mesh_path: Path,
    output_dir: Path,
    *,
    dt: float,
    final_time: float,
    adapt_start: float,
    adapt_interval: int,
    motion_substeps: int,
    write_stride: int,
    mean_speed: float,
    inlet_perturbation: float,
    monitor_kind: str,
    monitor_gain: float,
    monitor_ceiling: float,
    adapt_relaxation: float,
    max_adapt_displacement: float,
    rtol: float,
    maxiter: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = fd.Mesh(str(mesh_path))
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()

    state = _create_solver_state(mesh, dt, mean_speed=mean_speed, inlet_perturbation=inlet_perturbation)
    state.monitor_kind = monitor_kind
    state.monitor_gain = monitor_gain
    state.monitor_ceiling = monitor_ceiling
    series = output_dir / "navier_stokes_monge_ampere_cylinder.pvd"
    vtk = fd.VTKFile(str(series))

    timings: dict[str, float] = {}
    drag_lift: list[dict[str, float]] = []
    adaptations: list[dict[str, float | int]] = []
    time_value = 0.0
    max_grid_speed = 0.0
    total_steps = max(1, int(round(final_time / dt)))
    adapt_start_step = max(1, int(round(adapt_start / dt)))
    adapt_interval = max(1, adapt_interval)
    motion_substeps = max(1, motion_substeps)
    write_stride = max(1, write_stride)
    pending_target: np.ndarray | None = None
    motion_start: np.ndarray | None = None
    motion_remaining = 0

    _write_state(vtk, state, initial_coordinates, time_value)
    start = time.perf_counter()
    timings["monitor_s"] = 0.0
    timings["adapter_s"] = 0.0
    for step in range(1, total_steps + 1):
        should_adapt = step >= adapt_start_step and (step - adapt_start_step) % adapt_interval == 0
        if should_adapt and motion_remaining == 0:
            monitor_start = time.perf_counter()
            monitor = _solution_monitor(
                mesh,
                state.u_now,
                kind=monitor_kind,
                gain=monitor_gain,
                ceiling=monitor_ceiling,
            )
            timings["monitor_s"] += time.perf_counter() - monitor_start

            adapter_start = time.perf_counter()
            raw_target_coordinates, iteration_index = _monge_ampere_target(
                mesh,
                monitor,
                rtol=rtol,
                maxiter=maxiter,
            )
            adapter_elapsed = time.perf_counter() - adapter_start
            timings["adapter_s"] += adapter_elapsed

            motion_start = mesh.coordinates.dat.data_ro.copy()
            target_coordinates = _limit_target_displacement(
                motion_start,
                raw_target_coordinates,
                relaxation=adapt_relaxation,
                max_displacement=max_adapt_displacement,
            )
            pending_target = target_coordinates
            motion_remaining = motion_substeps
            target_displacement = np.linalg.norm(pending_target - motion_start, axis=1)
            adaptations.append(
                {
                    "step": step,
                    "time": float(time_value),
                    "iterations": int(iteration_index) + 1,
                    "adapter_s": adapter_elapsed,
                    "max_target_displacement": float(target_displacement.max()),
                    "mean_target_displacement": float(target_displacement.mean()),
                }
            )

        if motion_remaining and pending_target is not None and motion_start is not None:
            previous_coordinates = mesh.coordinates.dat.data_ro.copy()
            completed = motion_substeps - motion_remaining + 1
            fraction = completed / motion_substeps
            next_coordinates = motion_start + fraction * (pending_target - motion_start)
            mesh.coordinates.dat.data[:] = next_coordinates
            state.u_grid.dat.data[:] = (next_coordinates - previous_coordinates) / dt
            max_grid_speed = max(max_grid_speed, float(np.linalg.norm(state.u_grid.dat.data_ro, axis=1).max()))
            motion_remaining -= 1
        else:
            state.u_grid.dat.data[:] = 0.0

        _solve_step(state, time_value)
        time_value += dt
        if step % write_stride == 0 or step == total_steps:
            _write_state(vtk, state, initial_coordinates, time_value)
            drag_lift.append(_drag_lift(state, time_value))
    timings["solve_loop_s"] = time.perf_counter() - start
    state.u_grid.dat.data[:] = 0.0
    _write_final_plot(output_dir / "final_vorticity_mesh.png", state)

    displacement = np.linalg.norm(mesh.coordinates.dat.data_ro.copy() - initial_coordinates, axis=1)
    re_num = mean_speed * CYLINDER_DIAMETER / NU_VALUE
    metrics = output_dir / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "adapter": "movement_monge_ampere",
                "case": "firedrake_cylinder_navier_stokes_ale",
                "mesh": str(mesh_path),
                "num_vertices": int(mesh.num_vertices()),
                "num_cells": int(mesh.num_cells()),
                "dt": dt,
                "final_time": time_value,
                "target_final_time": final_time,
                "total_steps": total_steps,
                "adapt_start": adapt_start,
                "adapt_interval": adapt_interval,
                "motion_substeps": motion_substeps,
                "write_stride": write_stride,
                "mean_speed": mean_speed,
                "reynolds_number": re_num,
                "inlet_perturbation": inlet_perturbation,
                "monitor_kind": monitor_kind,
                "monitor_gain": monitor_gain,
                "monitor_ceiling": monitor_ceiling,
                "adapt_relaxation": adapt_relaxation,
                "max_adapt_displacement": max_adapt_displacement,
                "rtol": rtol,
                "maxiter": maxiter,
                "adaptation_count": len(adaptations),
                "adaptations": adaptations,
                "timings_s": timings,
                "max_displacement": float(displacement.max()),
                "mean_displacement": float(displacement.mean()),
                "max_grid_speed": max_grid_speed,
                "max_vorticity": float(state.vorticity_out.dat.data_ro.max()),
                "min_vorticity": float(state.vorticity_out.dat.data_ro.min()),
                "final_drag_lift": drag_lift[-1] if drag_lift else None,
                "drag_lift": drag_lift,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {"series": series, "metrics": metrics}


class _SolverState:
    pass


def _create_solver_state(mesh, dt: float, *, mean_speed: float, inlet_perturbation: float) -> _SolverState:
    state = _SolverState()
    state.mesh = mesh
    state.dt = dt
    state.nu = fd.Constant(NU_VALUE)
    state.k = fd.Constant(dt)
    state.time = fd.Constant(0.0)
    state.mean_speed = mean_speed
    state.V = fd.VectorFunctionSpace(mesh, "CG", 2)
    state.Q = fd.FunctionSpace(mesh, "CG", 1)
    state.u_grid = fd.Function(mesh.coordinates.function_space(), name="grid_velocity")
    state.u_now = fd.Function(state.V, name="velocity")
    state.u_next = fd.Function(state.V, name="velocity")
    state.u_star = fd.Function(state.V, name="velocity_star")
    state.p_now = fd.Function(state.Q, name="pressure")
    state.p_next = fd.Function(state.Q, name="pressure")
    state.monitor_out = fd.Function(state.Q, name="monitor")
    state.displacement_out = fd.Function(state.Q, name="mesh_displacement")
    state.grid_speed_out = fd.Function(state.Q, name="grid_speed")
    state.vorticity_out = fd.Function(state.Q, name="vorticity")

    u = fd.TrialFunction(state.V)
    v = fd.TestFunction(state.V)
    p = fd.TrialFunction(state.Q)
    q = fd.TestFunction(state.Q)
    n = fd.FacetNormal(mesh)
    x, y = fd.SpatialCoordinate(mesh)
    f = fd.Constant((0.0, 0.0))
    u_mid = 0.5 * (state.u_now + u)
    u_adv = state.u_now - state.u_grid

    profile_shape = 4.0 * y * (CHANNEL_HEIGHT - y) / CHANNEL_HEIGHT**2
    profile = mean_speed * profile_shape
    trigger = inlet_perturbation * mean_speed * fd.sin(2.0 * np.pi * state.time) * profile_shape
    inflow = (profile, trigger)
    state.bcu = [
        fd.DirichletBC(state.V, fd.Constant((0.0, 0.0)), (WALL_MARKER, CYLINDER_MARKER)),
        fd.DirichletBC(state.V, inflow, INLET_MARKER),
    ]
    state.bcp = [fd.DirichletBC(state.Q, fd.Constant(0.0), OUTLET_MARKER)]

    F1 = (
        fd.inner((u - state.u_now) / state.k, v) * fd.dx
        + fd.inner(fd.dot(u_adv, fd.nabla_grad(u_mid)), v) * fd.dx
        + fd.inner(_sigma(u_mid, state.p_now, state.nu), fd.sym(fd.nabla_grad(v))) * fd.dx
        + fd.inner(state.p_now * n, v) * fd.ds
        - fd.inner(state.nu * fd.dot(fd.nabla_grad(u_mid), n), v) * fd.ds
        - fd.inner(f, v) * fd.dx
    )
    a1, L1 = fd.system(F1)

    a2 = fd.inner(fd.nabla_grad(p), fd.nabla_grad(q)) * fd.dx
    L2 = fd.inner(fd.nabla_grad(state.p_now), fd.nabla_grad(q)) * fd.dx - (1.0 / state.k) * fd.inner(
        fd.div(state.u_star), q
    ) * fd.dx

    a3 = fd.inner(u, v) * fd.dx
    L3 = fd.inner(state.u_star, v) * fd.dx - state.k * fd.inner(
        fd.nabla_grad(state.p_next - state.p_now), v
    ) * fd.dx

    state.solve1 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a1, L1, state.u_star, bcs=state.bcu),
        solver_parameters={"ksp_type": "gmres", "pc_type": "sor"},
    )
    state.solve2 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a2, L2, state.p_next, bcs=state.bcp),
        solver_parameters={"ksp_type": "cg", "pc_type": "gamg"},
    )
    state.solve3 = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(a3, L3, state.u_next),
        solver_parameters={"ksp_type": "cg", "pc_type": "sor"},
    )
    return state


def _solve_step(state: _SolverState, time_value: float) -> None:
    state.time.assign(time_value)
    state.solve1.solve()
    state.solve2.solve()
    state.solve3.solve()
    state.u_now.assign(state.u_next)
    state.p_now.assign(state.p_next)


def _sigma(u, p, nu):
    return 2.0 * nu * fd.sym(fd.nabla_grad(u)) - p * fd.Identity(len(u))


def _solution_monitor(mesh, velocity, *, kind: str, gain: float, ceiling: float) -> fd.Function:
    Q = fd.FunctionSpace(mesh, "CG", 1)
    signal = fd.Function(Q)
    if kind == "gradient":
        tensor_space = fd.TensorFunctionSpace(mesh, "CG", 1)
        grad_u = fd.Function(tensor_space)
        grad_u.interpolate(fd.grad(velocity))
        signal.interpolate(
            fd.sqrt(grad_u[0, 0] ** 2 + grad_u[0, 1] ** 2 + grad_u[1, 0] ** 2 + grad_u[1, 1] ** 2 + 1.0e-12)
        )
    else:
        signal.interpolate(fd.sqrt(fd.curl(velocity) ** 2 + 1.0e-12))

    if ceiling > 0.0:
        np.minimum(signal.dat.data, ceiling, out=signal.dat.data)
    max_value = max(float(signal.dat.data_ro.max()), 1e-12)
    monitor = fd.Function(Q, name="monitor")
    monitor.dat.data[:] = 1.0 + gain * signal.dat.data_ro / max_value
    return monitor


def _write_state(vtk, state: _SolverState, initial_coordinates: np.ndarray, time_value: float) -> None:
    displacement = np.linalg.norm(state.mesh.coordinates.dat.data_ro - initial_coordinates, axis=1)
    state.displacement_out.dat.data[:] = displacement
    state.monitor_out.assign(
        _solution_monitor(
            state.mesh,
            state.u_now,
            kind=state.monitor_kind,
            gain=state.monitor_gain,
            ceiling=state.monitor_ceiling,
        )
    )
    state.grid_speed_out.interpolate(fd.sqrt(fd.dot(state.u_grid, state.u_grid)))
    state.vorticity_out.project(fd.curl(state.u_now))
    vtk.write(
        state.u_now,
        state.p_now,
        state.vorticity_out,
        state.monitor_out,
        state.displacement_out,
        state.grid_speed_out,
        time=time_value,
    )


def _monge_ampere_target(mesh, monitor: fd.Function, *, rtol: float, maxiter: int) -> tuple[np.ndarray, int]:
    mover_mesh = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
    mover_monitor = fd.Function(fd.FunctionSpace(mover_mesh, "CG", 1), name="monitor")
    mover_monitor.dat.data[:] = monitor.dat.data_ro
    mover = mv.MongeAmpereMover(
        mover_mesh,
        lambda _mesh: mover_monitor,
        rtol=rtol,
        maxiter=maxiter,
        fixed_boundary_segments=["on_boundary"],
    )
    iteration_index = mover.move()
    return mover.mesh.coordinates.dat.data_ro.copy(), iteration_index


def _limit_target_displacement(
    current_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    relaxation: float,
    max_displacement: float,
) -> np.ndarray:
    displacement = max(0.0, relaxation) * (target_coordinates - current_coordinates)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    largest = float(displacement_norm.max())
    if max_displacement > 0.0 and largest > max_displacement:
        displacement *= max_displacement / largest
    return current_coordinates + displacement


def _write_final_plot(path: Path, state: _SolverState) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        state.vorticity_out.project(fd.curl(state.u_now))
        fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), constrained_layout=True)
        colors = fd.tripcolor(state.vorticity_out, axes=axes[0], cmap="coolwarm")
        fig.colorbar(colors, ax=axes[0], shrink=0.85)
        fd.triplot(state.mesh, axes=axes[1], linewidth=0.25)
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.4)
            ax.set_ylim(0.0, CHANNEL_HEIGHT)
        axes[0].set_title("Vorticity")
        axes[1].set_title("Adapted mesh")
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


def _drag_lift(state: _SolverState, time_value: float) -> dict[str, float]:
    n = fd.FacetNormal(state.mesh)
    stress = _sigma(state.u_now, state.p_now, state.nu)
    force_x = fd.assemble(fd.dot(n, stress)[0] * fd.ds(CYLINDER_MARKER))
    force_y = fd.assemble(fd.dot(n, stress)[1] * fd.ds(CYLINDER_MARKER))
    scale = 2.0 / CYLINDER_DIAMETER
    return {
        "time": float(time_value),
        "drag_coefficient": float(scale * force_x),
        "lift_coefficient": float(scale * force_y),
    }


if __name__ == "__main__":
    main()
