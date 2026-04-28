from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import firedrake as fd

CHANNEL_HEIGHT = 0.41
CHANNEL_LENGTH = 2.2
CYLINDER_DIAMETER = 0.1
CYLINDER_CENTER = (0.2, 0.2)
NU_VALUE = 0.001
U_MEAN = 1.0
WALL_MARKER = 1
INLET_MARKER = 2
OUTLET_MARKER = 3
CYLINDER_MARKER = 4
UM2N_GRID_PROJECTION_THRESHOLD = 5.0
MAX_RELAXATION_BACKTRACKS = 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake-side UM2N cylinder setup with Monge-Ampere adaptation.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--monitor-kind", choices=("velocity-gradient", "wake-vorticity"), default="velocity-gradient")
    parser.add_argument("--monitor-frame", choices=("reference", "current"), default="reference")
    parser.add_argument("--monitor-scale", type=float, default=0.2)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=5.0)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()

    paths = run_case(
        args.mesh,
        args.output_dir,
        dt=args.dt,
        steps=args.steps,
        adapt_every=args.adapt_every,
        save_every=args.save_every,
        monitor_kind=args.monitor_kind,
        monitor_frame=args.monitor_frame,
        monitor_scale=args.monitor_scale,
        adaptation_relaxation=args.adaptation_relaxation,
        max_grid_speed_limit=args.max_grid_speed,
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
    steps: int,
    adapt_every: int,
    save_every: int,
    monitor_kind: str,
    monitor_frame: str,
    monitor_scale: float,
    adaptation_relaxation: float,
    max_grid_speed_limit: float,
    rtol: float,
    maxiter: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = fd.Mesh(str(mesh_path))
    adapted_mesh = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
    init_coord = mesh.coordinates.copy(deepcopy=True).dat.data_ro.copy()

    state = _create_solver_state(mesh, dt)
    adapted_state = _create_adapted_fields(adapted_mesh)
    triangles = state.Q.cell_node_list
    initial_signed_area = _signed_triangle_area(init_coord, triangles)
    characteristic_cell_size = float(np.sqrt(np.abs(initial_signed_area).mean()))
    vtk = fd.VTKFile(str(output_dir / "um2n_monge_ampere_cylinder.pvd"))
    metrics: dict[str, object] = {
        "case": "um2n_cylinder_monge_ampere_baseline",
        "mesh": str(mesh_path),
        "num_vertices": int(mesh.num_vertices()),
        "num_cells": int(mesh.num_cells()),
        "characteristic_cell_size": characteristic_cell_size,
        "initial_mesh_quality": _triangle_quality(init_coord, triangles, initial_signed_area),
        "dt": dt,
        "steps": steps,
        "adapt_every": adapt_every,
        "save_every": save_every,
        "monitor_kind": monitor_kind,
        "monitor_frame": monitor_frame,
        "monitor_scale": monitor_scale,
        "adaptation_relaxation": adaptation_relaxation,
        "max_grid_speed_limit": max_grid_speed_limit,
        "um2n_grid_projection_threshold": UM2N_GRID_PROJECTION_THRESHOLD,
        "max_relaxation_backtracks": MAX_RELAXATION_BACKTRACKS,
        "rtol": rtol,
        "maxiter": maxiter,
        "reynolds_number": U_MEAN * CYLINDER_DIAMETER / NU_VALUE,
        "adaptations": [],
        "timings_s": {"solve": 0.0, "monitor": 0.0, "movement": 0.0, "projection": 0.0},
    }

    _write_state(vtk, state, init_coord, time_value=0.0, monitor=None)
    monitor_val = fd.Function(fd.FunctionSpace(mesh, "CG", 1), name="monitor")
    max_grid_speed = 0.0
    t = 0.0

    for step in range(1, steps + 1):
        solve_start = time.perf_counter()
        _solve_step(state)
        metrics["timings_s"]["solve"] += time.perf_counter() - solve_start
        t += dt

        if step % adapt_every == 0:
            current_coord = mesh.coordinates.dat.data_ro.copy()
            projection_start = time.perf_counter()
            if monitor_frame == "reference":
                adapted_mesh.coordinates.dat.data[:] = current_coord
                mesh.coordinates.dat.data[:] = init_coord
                u_for_monitor = fd.Function(state.V)
                adapted_state["u"].dat.data[:] = state.u_next.dat.data_ro
                u_for_monitor.project(adapted_state["u"])
            elif monitor_frame == "current":
                u_for_monitor = state.u_next
            else:
                raise ValueError(f"Unknown monitor_frame: {monitor_frame}")
            metrics["timings_s"]["projection"] += time.perf_counter() - projection_start

            monitor_start = time.perf_counter()
            monitor_val = _um2n_monge_ampere_monitor(
                mesh,
                u_for_monitor,
                monitor_kind=monitor_kind,
                monitor_scale=monitor_scale,
            )
            metrics["timings_s"]["monitor"] += time.perf_counter() - monitor_start
            monitor_data = monitor_val.dat.data_ro

            movement_start = time.perf_counter()
            movement_error = None
            try:
                target_coord, iterations = _monge_ampere_coordinates(mesh, monitor_val, rtol=rtol, maxiter=maxiter)
            except Exception as exc:
                target_coord = current_coord
                iterations = -1
                movement_error = f"{type(exc).__name__}: {exc}"
            movement_elapsed = time.perf_counter() - movement_start
            metrics["timings_s"]["movement"] += movement_elapsed

            mesh.coordinates.dat.data[:] = current_coord
            raw_target_coord = target_coord
            raw_quality = _triangle_quality(raw_target_coord, triangles, initial_signed_area)
            delta_coord = raw_target_coord - current_coord
            raw_grid_speed = float(np.abs(delta_coord).max() / dt)
            applied_relaxation = adaptation_relaxation
            if raw_grid_speed > 0.0:
                applied_relaxation = min(applied_relaxation, max_grid_speed_limit / raw_grid_speed)
            target_coord = current_coord + applied_relaxation * delta_coord
            relaxed_quality = _triangle_quality(target_coord, triangles, initial_signed_area)
            relaxation_backtracks = 0
            while (
                relaxed_quality["orientation_flips"] > 0
                and relaxation_backtracks < MAX_RELAXATION_BACKTRACKS
            ):
                applied_relaxation *= 0.5
                target_coord = current_coord + applied_relaxation * delta_coord
                relaxed_quality = _triangle_quality(target_coord, triangles, initial_signed_area)
                relaxation_backtracks += 1
            accepted_adaptation = relaxed_quality["orientation_flips"] == 0
            if relaxed_quality["orientation_flips"] > 0:
                applied_relaxation = 0.0
                target_coord = current_coord
                relaxed_quality = _triangle_quality(target_coord, triangles, initial_signed_area)
            if movement_error is not None:
                accepted_adaptation = False
            mesh.coordinates.dat.data[:] = target_coord
            adapted_mesh.coordinates.dat.data[:] = target_coord
            state.u_grid.dat.data[:] = (target_coord - current_coord) / dt
            applied_grid_speed = float(np.abs(state.u_grid.dat.data_ro).max())
            max_grid_speed = max(max_grid_speed, applied_grid_speed)
            used_post_adaptation_projection = accepted_adaptation and applied_grid_speed >= UM2N_GRID_PROJECTION_THRESHOLD
            if used_post_adaptation_projection:
                adapted_state["u"].dat.data[:] = state.u_next.dat.data_ro
                adapted_state["p"].dat.data[:] = state.p_next.dat.data_ro
                state.u_now.project(adapted_state["u"])
                state.p_now.project(adapted_state["p"])
                state.u_grid.dat.data[:] = 0.0
            else:
                state.u_now.assign(state.u_next)
                state.p_now.assign(state.p_next)

            metrics["adaptations"].append(
                {
                    "step": step,
                    "time": float(t),
                    "iterations": int(iterations) + 1,
                    "accepted_adaptation": accepted_adaptation,
                    "movement_error": movement_error,
                    "movement_s": movement_elapsed,
                    "monitor_min": float(monitor_data.min()),
                    "monitor_max": float(monitor_data.max()),
                    "monitor_mean": float(monitor_data.mean()),
                    "requested_relaxation": adaptation_relaxation,
                    "applied_relaxation": float(applied_relaxation),
                    "relaxation_backtracks": relaxation_backtracks,
                    "raw_grid_speed": raw_grid_speed,
                    "applied_grid_speed": applied_grid_speed,
                    "used_post_adaptation_projection": used_post_adaptation_projection,
                    "max_raw_step_displacement": float(np.linalg.norm(raw_target_coord - current_coord, axis=1).max()),
                    "max_relaxed_step_displacement": float(np.linalg.norm(target_coord - current_coord, axis=1).max()),
                    "max_target_displacement_from_initial": float(np.linalg.norm(target_coord - init_coord, axis=1).max()),
                    "max_raw_step_displacement_over_h": float(
                        np.linalg.norm(raw_target_coord - current_coord, axis=1).max() / characteristic_cell_size
                    ),
                    "max_relaxed_step_displacement_over_h": float(
                        np.linalg.norm(target_coord - current_coord, axis=1).max() / characteristic_cell_size
                    ),
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
    state.vorticity.project(fd.curl(state.u_now))
    metrics["final_time"] = float(t)
    metrics["max_grid_speed"] = max_grid_speed
    metrics["max_displacement"] = float(displacement.max())
    metrics["mean_displacement"] = float(displacement.mean())
    metrics["final_mesh_quality"] = _triangle_quality(mesh.coordinates.dat.data_ro, triangles, initial_signed_area)
    metrics["min_vorticity"] = float(state.vorticity.dat.data_ro.min())
    metrics["max_vorticity"] = float(state.vorticity.dat.data_ro.max())

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_final_plot(output_dir / "final_vorticity_mesh.png", state, init_coord)
    return {"series": output_dir / "um2n_monge_ampere_cylinder.pvd", "metrics": metrics_path}


def _create_solver_state(mesh, dt: float):
    state = argparse.Namespace()
    state.mesh = mesh
    state.nu = fd.Constant(NU_VALUE)
    state.k = fd.Constant(dt)
    state.V = fd.VectorFunctionSpace(mesh, "CG", 2)
    state.Q = fd.FunctionSpace(mesh, "CG", 1)
    state.u_grid = fd.Function(mesh.coordinates.function_space(), name="gridvelocity")
    state.u_now = fd.Function(state.V, name="velocity")
    state.u_next = fd.Function(state.V, name="velocity")
    state.u_star = fd.Function(state.V, name="velocity_star")
    state.p_now = fd.Function(state.Q, name="pressure")
    state.p_next = fd.Function(state.Q, name="pressure")
    state.vorticity = fd.Function(state.Q, name="vorticity")
    state.monitor_out = fd.Function(state.Q, name="monitor")
    state.displacement = fd.Function(state.Q, name="mesh_displacement")
    state.grid_speed = fd.Function(state.Q, name="grid_speed")

    u = fd.TrialFunction(state.V)
    v = fd.TestFunction(state.V)
    p = fd.TrialFunction(state.Q)
    q = fd.TestFunction(state.Q)
    n = fd.FacetNormal(mesh)
    x, y = fd.SpatialCoordinate(mesh)
    f = fd.Constant((0.0, 0.0))
    u_mid = 0.5 * (state.u_now + u)
    u_adv = state.u_now - state.u_grid

    def sigma(velocity, pressure):
        return 2 * state.nu * fd.sym(fd.nabla_grad(velocity)) - pressure * fd.Identity(len(velocity))

    bcu = [
        fd.DirichletBC(state.V, fd.Constant((0.0, 0.0)), (WALL_MARKER, CYLINDER_MARKER)),
        fd.DirichletBC(state.V, ((4.0 * 1.5 * y * (CHANNEL_HEIGHT - y) / CHANNEL_HEIGHT**2), 0.0), INLET_MARKER),
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


def _create_adapted_fields(adapted_mesh):
    return {
        "u": fd.Function(fd.VectorFunctionSpace(adapted_mesh, "CG", 2)),
        "p": fd.Function(fd.FunctionSpace(adapted_mesh, "CG", 1)),
    }


def _solve_step(state) -> None:
    state.solve1.solve()
    state.solve2.solve()
    state.solve3.solve()


def _um2n_monge_ampere_monitor(
    mesh,
    velocity,
    *,
    monitor_kind: str,
    monitor_scale: float,
) -> fd.Function:
    function_space = fd.FunctionSpace(mesh, "CG", 1)
    grad_norm = fd.Function(function_space, name="monitor")
    if monitor_kind == "velocity-gradient":
        tensor_space = fd.TensorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.Function(tensor_space)
        uh_grad.interpolate(fd.grad(velocity))
        grad_norm.interpolate(
            uh_grad[0, 0] ** 2 + uh_grad[0, 1] ** 2 + uh_grad[1, 0] ** 2 + uh_grad[1, 1] ** 2
        )
    elif monitor_kind == "wake-vorticity":
        vorticity = fd.Function(function_space)
        vorticity.project(fd.curl(velocity))
        grad_norm.dat.data[:] = np.abs(vorticity.dat.data_ro)
        _apply_wake_weight(grad_norm)
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


def _apply_wake_weight(monitor: fd.Function) -> None:
    coordinates = monitor.function_space().mesh().coordinates.dat.data_ro
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    wall_distance = np.minimum(y, CHANNEL_HEIGHT - y)
    wall_weight = np.clip(wall_distance / 0.06, 0.0, 1.0)
    wake_weight = 1.0 / (1.0 + np.exp(-(x - 0.25) / 0.035))
    cylinder_radius = CYLINDER_DIAMETER / 2
    cylinder_distance = np.sqrt((x - CYLINDER_CENTER[0]) ** 2 + (y - CYLINDER_CENTER[1]) ** 2)
    cylinder_weight = 0.2 + 0.8 * np.clip((cylinder_distance - cylinder_radius) / 0.08, 0.0, 1.0)
    monitor.dat.data[:] *= wall_weight * wake_weight * cylinder_weight


def _signed_triangle_area(coordinates: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]
    return 0.5 * (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def _triangle_quality(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    reference_signed_area: np.ndarray,
) -> dict[str, float | int]:
    signed_area = _signed_triangle_area(coordinates, triangles)
    area = np.abs(signed_area)
    positive_area = np.maximum(area, 1.0e-16)
    orientation_flips = np.sign(signed_area) != np.sign(reference_signed_area)
    return {
        "min_signed_area": float(signed_area.min()),
        "min_area": float(area.min()),
        "max_area": float(area.max()),
        "area_ratio": float(positive_area.max() / positive_area.min()),
        "orientation_flips": int(np.count_nonzero(orientation_flips)),
    }


def _monge_ampere_coordinates(mesh, monitor: fd.Function, *, rtol: float, maxiter: int) -> tuple[np.ndarray, int]:
    import movement as mv

    mover_mesh = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
    mover_monitor = fd.Function(fd.FunctionSpace(mover_mesh, "CG", 1), name="monitor")
    mover_monitor.dat.data[:] = monitor.dat.data_ro
    mover = mv.MongeAmpereMover(
        mover_mesh,
        lambda _mesh: mover_monitor,
        method="relaxation",
        rtol=rtol,
        maxiter=maxiter,
        fixed_boundary_segments=["on_boundary"],
    )
    iteration_index = mover.move()
    return mover.mesh.coordinates.dat.data_ro.copy(), iteration_index


def _write_state(vtk, state, init_coord: np.ndarray, *, time_value: float, monitor: fd.Function | None) -> None:
    state.displacement.dat.data[:] = np.linalg.norm(state.mesh.coordinates.dat.data_ro - init_coord, axis=1)
    if monitor is not None:
        state.monitor_out.assign(monitor)
    state.vorticity.project(fd.curl(state.u_now))
    state.grid_speed.interpolate(fd.sqrt(fd.dot(state.u_grid, state.u_grid)))
    vtk.write(
        state.u_now,
        state.p_now,
        state.vorticity,
        state.monitor_out,
        state.displacement,
        state.grid_speed,
        time=time_value,
    )


def _write_final_plot(path: Path, state, init_coord: np.ndarray, *, adapter_label: str = "Monge-Ampere") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        state.vorticity.project(fd.curl(state.u_now))
        coordinates = state.mesh.coordinates.dat.data_ro
        triangles = state.Q.cell_node_list
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
        fig, axes = plt.subplots(5, 1, figsize=(12, 11.0), constrained_layout=True)
        vorticity = state.vorticity.dat.data_ro.copy()
        color_limit = float(np.percentile(np.abs(vorticity), 97.0))
        colors = axes[0].tripcolor(
            triangulation,
            vorticity,
            cmap="coolwarm",
            shading="gouraud",
            vmin=-color_limit,
            vmax=color_limit,
        )
        fig.colorbar(colors, ax=axes[0], shrink=0.85, extend="both")
        monitor_colors = axes[1].tripcolor(
            triangulation,
            state.monitor_out.dat.data_ro,
            cmap="magma",
            shading="gouraud",
        )
        axes[1].triplot(triangulation, color="white", linewidth=0.08, alpha=0.25)
        fig.colorbar(monitor_colors, ax=axes[1], shrink=0.85)
        displacement = np.linalg.norm(coordinates - init_coord, axis=1)
        displacement_colors = axes[2].tripcolor(
            triangulation,
            displacement,
            cmap="viridis",
            shading="gouraud",
        )
        fig.colorbar(displacement_colors, ax=axes[2], shrink=0.85)
        axes[3].triplot(triangulation, color="black", linewidth=0.2)
        axes[4].triplot(triangulation, color="black", linewidth=0.25)
        for ax in axes[:4]:
            ax.set_aspect("equal")
            ax.set_xlim(0.0, CHANNEL_LENGTH)
            ax.set_ylim(0.0, CHANNEL_HEIGHT)
        axes[4].set_aspect("equal")
        axes[4].set_xlim(0.05, 0.65)
        axes[4].set_ylim(0.02, 0.38)
        axes[0].set_title("Vorticity")
        axes[1].set_title("Monitor")
        axes[2].set_title("Mesh displacement")
        axes[3].set_title(f"{adapter_label} adapted mesh")
        axes[4].set_title("Adapted mesh near cylinder")
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
