from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import firedrake as fd


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake ALE free-stream/GCL validation on a moving mesh.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--amplitude", type=float, default=0.035)
    args = parser.parse_args()

    metrics_path = run_case(
        args.output_dir,
        nx=args.nx,
        ny=args.ny,
        steps=args.steps,
        dt=args.dt,
        amplitude=args.amplitude,
    )
    print(f"metrics: {metrics_path}")


def run_case(
    output_dir: Path,
    *,
    nx: int,
    ny: int,
    steps: int,
    dt: float,
    amplitude: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = fd.UnitSquareMesh(nx, ny)
    reference_coord = mesh.coordinates.dat.data_ro.copy()
    triangles = fd.FunctionSpace(mesh, "CG", 1).cell_node_list
    initial_signed_area = _signed_triangle_area(reference_coord, triangles)

    vector_constant = np.array([0.37, -0.22], dtype=float)
    scalar_constant = 0.73

    state = _create_state(mesh, dt, vector_constant, scalar_constant)
    _interpolate_smooth_scalar(state.q_smooth_now, mesh)
    state.q_smooth_next.assign(state.q_smooth_now)
    _interpolate_smooth_scalar(state.q_exact, mesh)
    initial_smooth_mass = fd.assemble(state.q_exact * fd.dx(domain=mesh))

    max_grid_speed = 0.0
    max_vector_linf_error = 0.0
    max_scalar_linf_error = 0.0
    max_scalar_mass_error = 0.0
    max_smooth_scalar_linf_error = 0.0
    max_smooth_scalar_l2_error = 0.0
    max_smooth_scalar_mass_error = 0.0
    max_domain_area_error = 0.0
    min_cell_area = float("inf")
    max_orientation_flips = 0
    t = 0.0

    for step in range(1, steps + 1):
        current_coord = mesh.coordinates.dat.data_ro.copy()
        target_coord = _target_coordinates(reference_coord, t + dt, amplitude)
        mesh.coordinates.dat.data[:] = target_coord
        state.u_grid.dat.data[:] = (target_coord - current_coord) / dt
        max_grid_speed = max(max_grid_speed, float(np.linalg.norm(state.u_grid.dat.data_ro, axis=1).max()))

        state.vector_solver.solve()
        state.scalar_solver.solve()
        state.smooth_scalar_solver.solve()
        state.u_now.assign(state.u_next)
        state.q_now.assign(state.q_next)
        state.q_smooth_now.assign(state.q_smooth_next)
        t += dt

        _interpolate_smooth_scalar(state.q_exact, mesh)
        vector_error = np.abs(state.u_now.dat.data_ro - vector_constant).max()
        scalar_error = np.abs(state.q_now.dat.data_ro - scalar_constant).max()
        scalar_mass = fd.assemble(state.q_now * fd.dx(domain=mesh))
        smooth_scalar_linf_error = np.abs(state.q_smooth_now.dat.data_ro - state.q_exact.dat.data_ro).max()
        smooth_scalar_l2_error = np.sqrt(fd.assemble((state.q_smooth_now - state.q_exact) ** 2 * fd.dx(domain=mesh)))
        smooth_scalar_mass = fd.assemble(state.q_smooth_now * fd.dx(domain=mesh))
        domain_area = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        quality = _triangle_quality(mesh.coordinates.dat.data_ro, triangles, initial_signed_area)

        max_vector_linf_error = max(max_vector_linf_error, float(vector_error))
        max_scalar_linf_error = max(max_scalar_linf_error, float(scalar_error))
        max_scalar_mass_error = max(max_scalar_mass_error, float(abs(scalar_mass - scalar_constant)))
        max_smooth_scalar_linf_error = max(max_smooth_scalar_linf_error, float(smooth_scalar_linf_error))
        max_smooth_scalar_l2_error = max(max_smooth_scalar_l2_error, float(smooth_scalar_l2_error))
        max_smooth_scalar_mass_error = max(max_smooth_scalar_mass_error, float(abs(smooth_scalar_mass - initial_smooth_mass)))
        max_domain_area_error = max(max_domain_area_error, float(abs(domain_area - 1.0)))
        min_cell_area = min(min_cell_area, quality["min_area"])
        max_orientation_flips = max(max_orientation_flips, quality["orientation_flips"])

        if step == 1 or step == steps:
            print(
                f"step {step}/{steps}, "
                f"vector_linf={vector_error:.3e}, "
                f"scalar_linf={scalar_error:.3e}, "
                f"smooth_scalar_l2={smooth_scalar_l2_error:.3e}, "
                f"mass_error={abs(scalar_mass - scalar_constant):.3e}",
                flush=True,
            )

    metrics = {
        "case": "firedrake_ale_gcl_validation",
        "nx": nx,
        "ny": ny,
        "steps": steps,
        "dt": dt,
        "final_time": t,
        "amplitude": amplitude,
        "mesh_motion": "fixed-boundary smooth interior displacement",
        "vector_form": "current Navier-Stokes-style advective ALE form",
        "scalar_form": "advective ALE form",
        "vector_constant": vector_constant.tolist(),
        "scalar_constant": scalar_constant,
        "max_grid_speed": max_grid_speed,
        "max_vector_linf_error": max_vector_linf_error,
        "max_scalar_linf_error": max_scalar_linf_error,
        "max_scalar_mass_error": max_scalar_mass_error,
        "smooth_scalar": "1 + 0.2 sin(2 pi x) sin(2 pi y)",
        "max_smooth_scalar_linf_error": max_smooth_scalar_linf_error,
        "max_smooth_scalar_l2_error": max_smooth_scalar_l2_error,
        "max_smooth_scalar_mass_error": max_smooth_scalar_mass_error,
        "max_domain_area_error": max_domain_area_error,
        "min_cell_area": min_cell_area,
        "max_orientation_flips": max_orientation_flips,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_path


def _create_state(mesh, dt: float, vector_constant: np.ndarray, scalar_constant: float):
    state = argparse.Namespace()
    state.u_grid = fd.Function(mesh.coordinates.function_space(), name="grid_velocity")

    state.V = fd.VectorFunctionSpace(mesh, "CG", 2)
    state.S = fd.FunctionSpace(mesh, "CG", 1)
    state.u_now = fd.Function(state.V, name="vector_state")
    state.u_next = fd.Function(state.V, name="vector_state")
    state.q_now = fd.Function(state.S, name="scalar_state")
    state.q_next = fd.Function(state.S, name="scalar_state")
    state.q_smooth_now = fd.Function(state.S, name="smooth_scalar_state")
    state.q_smooth_next = fd.Function(state.S, name="smooth_scalar_state")
    state.q_exact = fd.Function(state.S, name="smooth_scalar_exact")

    state.u_now.interpolate(fd.Constant(tuple(float(v) for v in vector_constant)))
    state.u_next.assign(state.u_now)
    state.q_now.interpolate(fd.Constant(float(scalar_constant)))
    state.q_next.assign(state.q_now)

    u = fd.TrialFunction(state.V)
    v = fd.TestFunction(state.V)
    u_mid = 0.5 * (state.u_now + u)
    u_adv = state.u_now - state.u_grid
    vector_form = (
        fd.inner((u - state.u_now) / dt, v) * fd.dx
        + fd.inner(fd.dot(u_adv, fd.nabla_grad(u_mid)), v) * fd.dx
    )
    vector_a, vector_l = fd.system(vector_form)
    state.vector_solver = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(vector_a, vector_l, state.u_next),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )

    q = fd.TrialFunction(state.S)
    r = fd.TestFunction(state.S)
    q_mid = 0.5 * (state.q_now + q)
    scalar_form = ((q - state.q_now) / dt) * r * fd.dx - fd.dot(state.u_grid, fd.grad(q_mid)) * r * fd.dx
    scalar_a, scalar_l = fd.system(scalar_form)
    state.scalar_solver = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(scalar_a, scalar_l, state.q_next),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )

    q_smooth = fd.TrialFunction(state.S)
    smooth_mid = 0.5 * (state.q_smooth_now + q_smooth)
    smooth_form = ((q_smooth - state.q_smooth_now) / dt) * r * fd.dx - fd.dot(state.u_grid, fd.grad(smooth_mid)) * r * fd.dx
    smooth_a, smooth_l = fd.system(smooth_form)
    state.smooth_scalar_solver = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(smooth_a, smooth_l, state.q_smooth_next),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return state


def _interpolate_smooth_scalar(function, mesh) -> None:
    x, y = fd.SpatialCoordinate(mesh)
    function.interpolate(1.0 + 0.2 * fd.sin(2.0 * np.pi * x) * fd.sin(2.0 * np.pi * y))


def _target_coordinates(reference_coord: np.ndarray, time_value: float, amplitude: float) -> np.ndarray:
    x = reference_coord[:, 0]
    y = reference_coord[:, 1]
    envelope = np.sin(np.pi * x) * np.sin(np.pi * y)
    displacement = np.column_stack(
        [
            amplitude * envelope * np.sin(2.0 * np.pi * time_value),
            0.7 * amplitude * envelope * np.sin(3.0 * np.pi * time_value),
        ]
    )
    return reference_coord + displacement


def _signed_triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vertices = points[triangles]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    return 0.5 * (edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])


def _triangle_quality(points: np.ndarray, triangles: np.ndarray, initial_signed_area: np.ndarray) -> dict[str, float | int]:
    signed_area = _signed_triangle_area(points, triangles)
    oriented_area = signed_area * np.sign(initial_signed_area)
    abs_area = np.abs(oriented_area)
    return {
        "min_area": float(abs_area.min()),
        "max_area": float(abs_area.max()),
        "area_ratio": float(abs_area.max() / max(abs_area.min(), 1e-300)),
        "orientation_flips": int(np.count_nonzero(oriented_area <= 0.0)),
    }


if __name__ == "__main__":
    main()
