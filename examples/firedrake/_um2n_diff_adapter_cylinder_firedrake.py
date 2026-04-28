from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import firedrake as fd

from examples.firedrake._um2n_monge_ampere_cylinder_firedrake import (  # noqa: E402
    CYLINDER_DIAMETER,
    NU_VALUE,
    U_MEAN,
    _create_adapted_fields,
    _create_solver_state,
    _signed_triangle_area,
    _solve_step,
    _triangle_quality,
    _um2n_monge_ampere_monitor,
    _write_final_plot,
    _write_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake-side UM2N cylinder setup with differentiable adaptation.")
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
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=8e-4)
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
        monitor_frame=args.monitor_frame,
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
    monitor_frame: str,
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
    adapted_mesh = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
    init_coord = mesh.coordinates.copy(deepcopy=True).dat.data_ro.copy()

    state = _create_solver_state(mesh, dt)
    adapted_state = _create_adapted_fields(adapted_mesh)
    triangles = state.Q.cell_node_list
    initial_signed_area = _signed_triangle_area(init_coord, triangles)
    vtk = fd.VTKFile(str(output_dir / "um2n_diff_adapter_cylinder.pvd"))
    metrics: dict[str, object] = {
        "case": "um2n_cylinder_diff_adapter",
        "mesh": str(mesh_path),
        "num_vertices": int(mesh.num_vertices()),
        "num_cells": int(mesh.num_cells()),
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
        "adapter_steps": adapter_steps,
        "adapter_lr": adapter_lr,
        "adapter_profile": adapter_profile,
        "adapter_exchange_dir": str(adapter_exchange_dir),
        "adapter_poll_interval": adapter_poll_interval,
        "reynolds_number": U_MEAN * CYLINDER_DIAMETER / NU_VALUE,
        "adaptations": [],
        "timings_s": {"solve": 0.0, "monitor": 0.0, "adapter": 0.0, "projection": 0.0},
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

            mesh.coordinates.dat.data[:] = current_coord
            raw_quality = _triangle_quality(raw_target_coord, triangles, initial_signed_area)
            delta_coord = raw_target_coord - current_coord
            raw_grid_speed = float(np.abs(delta_coord).max() / dt)
            applied_relaxation = adaptation_relaxation
            if raw_grid_speed > 0.0:
                applied_relaxation = min(applied_relaxation, max_grid_speed_limit / raw_grid_speed)
            target_coord = current_coord + applied_relaxation * delta_coord
            mesh.coordinates.dat.data[:] = target_coord
            adapted_mesh.coordinates.dat.data[:] = target_coord
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
                    "raw_target_mesh_quality": raw_quality,
                    "relaxed_target_mesh_quality": _triangle_quality(target_coord, triangles, initial_signed_area),
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
    _write_final_plot(output_dir / "final_vorticity_mesh.png", state, init_coord, adapter_label="Differentiable")
    return {"series": output_dir / "um2n_diff_adapter_cylinder.pvd", "metrics": metrics_path}


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
    triangles_np = monitor.function_space().cell_node_list.copy()
    monitor_np = monitor.dat.data_ro.copy()
    request_path = adapter_exchange_dir / f"request_{request_index:06d}.npz"
    response_path = adapter_exchange_dir / f"response_{request_index:06d}.npz"
    tmp_path = adapter_exchange_dir / f"request_{request_index:06d}.tmp"
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            points=points_np,
            triangles=triangles_np,
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


if __name__ == "__main__":
    main()
