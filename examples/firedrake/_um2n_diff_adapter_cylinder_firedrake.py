from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np

import firedrake as fd

from examples.firedrake._adapter_tcp_protocol import recv_message, send_message
from examples.firedrake._um2n_monge_ampere_cylinder_firedrake import (  # noqa: E402
    CYLINDER_DIAMETER,
    MAX_RELAXATION_BACKTRACKS,
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
    parser.add_argument(
        "--monitor-build",
        choices=("firedrake-smoothed", "raw-gradient", "graph-gradient", "raw-vorticity", "composite-grad-vort"),
        default="firedrake-smoothed",
    )
    parser.add_argument("--monitor-scale", type=float, default=0.2)
    parser.add_argument("--monitor-graph-smoothing-steps", type=int, default=4)
    parser.add_argument("--monitor-graph-smoothing-weight", type=float, default=0.5)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=5.0)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=8e-4)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only"), default="regularized")
    parser.add_argument("--adapter-preset", choices=("custom", "accurate", "fast", "faster"), default="custom")
    parser.add_argument("--adapter-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--adapter-transport", choices=("npz", "tcp"), default="npz")
    parser.add_argument("--adapter-exchange-dir", type=Path, required=True)
    parser.add_argument("--adapter-poll-interval", type=float, default=0.001)
    parser.add_argument("--adapter-tcp-host", default="host.docker.internal")
    parser.add_argument("--adapter-tcp-port", type=int, default=None)
    parser.add_argument("--adapter-torch-threads", type=int, default=1)
    args = parser.parse_args()
    adapter_steps = _adapter_steps_for_preset(args.adapter_preset, args.adapter_steps)

    paths = run_case(
        args.mesh,
        args.output_dir,
        dt=args.dt,
        steps=args.steps,
        adapt_every=args.adapt_every,
        save_every=args.save_every,
        monitor_kind=args.monitor_kind,
        monitor_frame=args.monitor_frame,
        monitor_build=args.monitor_build,
        monitor_scale=args.monitor_scale,
        monitor_graph_smoothing_steps=args.monitor_graph_smoothing_steps,
        monitor_graph_smoothing_weight=args.monitor_graph_smoothing_weight,
        adaptation_relaxation=args.adaptation_relaxation,
        max_grid_speed_limit=args.max_grid_speed,
        adapter_steps=adapter_steps,
        adapter_lr=args.adapter_lr,
        adapter_profile=args.adapter_profile,
        adapter_preset=args.adapter_preset,
        adapter_dtype=args.adapter_dtype,
        adapter_transport=args.adapter_transport,
        adapter_exchange_dir=args.adapter_exchange_dir,
        adapter_poll_interval=args.adapter_poll_interval,
        adapter_tcp_host=args.adapter_tcp_host,
        adapter_tcp_port=args.adapter_tcp_port,
        adapter_torch_threads=args.adapter_torch_threads,
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
    monitor_build: str,
    monitor_scale: float,
    monitor_graph_smoothing_steps: int,
    monitor_graph_smoothing_weight: float,
    adaptation_relaxation: float,
    max_grid_speed_limit: float,
    adapter_steps: int,
    adapter_lr: float,
    adapter_profile: str,
    adapter_preset: str,
    adapter_dtype: str,
    adapter_transport: str,
    adapter_exchange_dir: Path,
    adapter_poll_interval: float,
    adapter_tcp_host: str,
    adapter_tcp_port: int | None,
    adapter_torch_threads: int,
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
        "monitor_build": monitor_build,
        "monitor_scale": monitor_scale,
        "monitor_graph_smoothing_steps": monitor_graph_smoothing_steps,
        "monitor_graph_smoothing_weight": monitor_graph_smoothing_weight,
        "adaptation_relaxation": adaptation_relaxation,
        "max_grid_speed_limit": max_grid_speed_limit,
        "adapter_steps": adapter_steps,
        "adapter_lr": adapter_lr,
        "adapter_profile": adapter_profile,
        "adapter_preset": adapter_preset,
        "adapter_dtype": adapter_dtype,
        "adapter_transport": adapter_transport,
        "adapter_exchange_dir": str(adapter_exchange_dir),
        "adapter_poll_interval": adapter_poll_interval,
        "adapter_tcp_host": adapter_tcp_host if adapter_transport == "tcp" else None,
        "adapter_tcp_port": adapter_tcp_port if adapter_transport == "tcp" else None,
        "adapter_torch_threads": adapter_torch_threads,
        "coupling_scheme": "direct_ale_in_step",
        "max_relaxation_backtracks": MAX_RELAXATION_BACKTRACKS,
        "reynolds_number": U_MEAN * CYLINDER_DIAMETER / NU_VALUE,
        "adaptations": [],
        "timings_s": {"solve": 0.0, "monitor": 0.0, "adapter": 0.0, "projection": 0.0},
    }

    _write_state(vtk, state, init_coord, time_value=0.0, monitor=None)
    monitor_val = fd.Function(fd.FunctionSpace(mesh, "CG", 1), name="monitor")
    cheap_monitor_builder: _CheapMonitorBuilder | None = None
    adapter_tcp_client = _AdapterTcpClient(adapter_tcp_host, adapter_tcp_port) if adapter_transport == "tcp" else None
    max_grid_speed = 0.0
    t = 0.0

    try:
        for step in range(1, steps + 1):
            adaptation_record: dict[str, object] | None = None
            if step % adapt_every == 0:
                current_coord = mesh.coordinates.dat.data_ro.copy()
                projection_start = time.perf_counter()
                if monitor_frame == "reference":
                    adapted_mesh.coordinates.dat.data[:] = current_coord
                    mesh.coordinates.dat.data[:] = init_coord
                    u_for_monitor = fd.Function(state.V)
                    adapted_state["u"].dat.data[:] = state.u_now.dat.data_ro
                    u_for_monitor.project(adapted_state["u"])
                elif monitor_frame == "current":
                    u_for_monitor = state.u_now
                else:
                    raise ValueError(f"Unknown monitor_frame: {monitor_frame}")
                metrics["timings_s"]["projection"] += time.perf_counter() - projection_start

                monitor_start = time.perf_counter()
                monitor_val, cell_monitor_data, cheap_monitor_builder = _build_adapter_monitor(
                    mesh=mesh,
                    cells=triangles,
                    velocity=u_for_monitor,
                    monitor_kind=monitor_kind,
                    monitor_build=monitor_build,
                    monitor_scale=monitor_scale,
                    graph_smoothing_steps=monitor_graph_smoothing_steps,
                    graph_smoothing_weight=monitor_graph_smoothing_weight,
                    cheap_monitor_builder=cheap_monitor_builder,
                )
                monitor_elapsed = time.perf_counter() - monitor_start
                metrics["timings_s"]["monitor"] += monitor_elapsed
                monitor_data = monitor_val.dat.data_ro

                adapter_start = time.perf_counter()
                raw_target_coord, adapter_info = _diff_adapter_coordinates(
                    mesh,
                    monitor_val,
                    adapter_steps=adapter_steps,
                    adapter_lr=adapter_lr,
                    adapter_profile=adapter_profile,
                    adapter_dtype=adapter_dtype,
                    cell_monitor_np=cell_monitor_data,
                    adapter_transport=adapter_transport,
                    adapter_tcp_client=adapter_tcp_client,
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
                if not accepted_adaptation:
                    applied_relaxation = 0.0
                    target_coord = current_coord
                    relaxed_quality = _triangle_quality(target_coord, triangles, initial_signed_area)
                mesh.coordinates.dat.data[:] = target_coord
                adapted_mesh.coordinates.dat.data[:] = target_coord
                state.u_grid.dat.data[:] = (target_coord - current_coord) / dt
                applied_grid_speed = float(np.abs(state.u_grid.dat.data_ro).max())
                max_grid_speed = max(max_grid_speed, applied_grid_speed)

                adaptation_record = {
                    "step": step,
                    "time_start": float(t),
                    "time_end": float(t + dt),
                    "adapter_loss_initial": adapter_info["initial_loss"],
                    "adapter_loss_final": adapter_info["final_loss"],
                    "adapter_steps_completed": adapter_info["steps_completed"],
                    "adapter_early_stopped": adapter_info["early_stopped"],
                    "monitor_s": monitor_elapsed,
                    "adapter_s": adapter_elapsed,
                    "adapter_service_s": adapter_info["service_s"],
                    "adapter_exchange_overhead_s": max(0.0, adapter_elapsed - adapter_info["service_s"]),
                    "adapter_response_wait_s": adapter_info["response_wait_s"],
                    "accepted_adaptation": accepted_adaptation,
                    "monitor_min": float(monitor_data.min()),
                    "monitor_max": float(monitor_data.max()),
                    "monitor_mean": float(monitor_data.mean()),
                    "requested_relaxation": adaptation_relaxation,
                    "applied_relaxation": float(applied_relaxation),
                    "relaxation_backtracks": relaxation_backtracks,
                    "raw_grid_speed": raw_grid_speed,
                    "applied_grid_speed": applied_grid_speed,
                    "max_raw_step_displacement": float(np.linalg.norm(raw_target_coord - current_coord, axis=1).max()),
                    "max_relaxed_step_displacement": float(np.linalg.norm(target_coord - current_coord, axis=1).max()),
                    "max_target_displacement_from_initial": float(np.linalg.norm(target_coord - init_coord, axis=1).max()),
                    "raw_target_mesh_quality": raw_quality,
                    "relaxed_target_mesh_quality": relaxed_quality,
                }
            else:
                state.u_grid.dat.data[:] = 0.0

            solve_start = time.perf_counter()
            _solve_step(state)
            metrics["timings_s"]["solve"] += time.perf_counter() - solve_start
            t += dt
            state.u_now.assign(state.u_next)
            state.p_now.assign(state.p_next)
            if adaptation_record is not None:
                metrics["adaptations"].append(adaptation_record)

            if step % save_every == 0 or step == steps:
                print(f"step {step}/{steps}, t={t:.3f}, adaptations={len(metrics['adaptations'])}", flush=True)
                _write_state(vtk, state, init_coord, time_value=t, monitor=monitor_val)
    finally:
        if adapter_tcp_client is not None:
            adapter_tcp_client.close()

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

class _CheapMonitorBuilder:
    def __init__(
        self,
        mesh,
        cells: np.ndarray,
        *,
        monitor_scale: float,
        graph_smoothing_steps: int,
        graph_smoothing_weight: float,
    ) -> None:
        self.cells = cells.copy()
        self.monitor_scale = monitor_scale
        self.graph_smoothing_steps = max(int(graph_smoothing_steps), 0)
        self.graph_smoothing_weight = min(max(float(graph_smoothing_weight), 0.0), 1.0)
        (
            self.smoothing_targets,
            self.smoothing_sources,
            self.smoothing_degree,
            self.smoothing_has_neighbors,
        ) = _cell_neighbor_scatter(self.cells)
        self.cell_node_columns = tuple(self.cells[:, i] for i in range(self.cells.shape[1]))
        self.node_flat_cells = self.cells.reshape(-1)
        self.node_cell_indices = np.repeat(np.arange(self.cells.shape[0]), self.cells.shape[1])
        self.node_count = np.bincount(self.node_flat_cells, minlength=mesh.num_vertices()).astype(np.float64)
        self.node_has_cells = self.node_count > 0.0
        self.function_space = fd.FunctionSpace(mesh, "CG", 1)
        self.grad_norm = fd.Function(self.function_space)
        self.vorticity = fd.Function(self.function_space)
        self.monitor = fd.Function(self.function_space, name="monitor")

    def build(self, velocity, monitor_build: str) -> tuple[fd.Function, np.ndarray]:
        raw_parts = []
        if monitor_build in ("raw-gradient", "graph-gradient", "composite-grad-vort"):
            self.grad_norm.interpolate(fd.inner(fd.grad(velocity), fd.grad(velocity)))
            raw_parts.append(_normalized_monitor_feature(self.grad_norm.dat.data_ro))

        if monitor_build in ("raw-vorticity", "composite-grad-vort"):
            self.vorticity.project(fd.curl(velocity))
            raw_parts.append(_normalized_monitor_feature(np.abs(self.vorticity.dat.data_ro)))

        if not raw_parts:
            raise ValueError(f"Unknown cheap monitor build: {monitor_build}")

        raw = raw_parts[0] if len(raw_parts) == 1 else np.maximum.reduce(raw_parts)
        self.monitor.dat.data[:] = 1.0 + self.monitor_scale * raw
        cell_monitor = _cell_average_from_node_columns(self.cell_node_columns, self.monitor.dat.data_ro)
        if monitor_build == "graph-gradient":
            cell_monitor = _smooth_cell_monitor(
                cell_monitor,
                self.smoothing_targets,
                self.smoothing_sources,
                self.smoothing_degree,
                self.smoothing_has_neighbors,
                steps=self.graph_smoothing_steps,
                weight=self.graph_smoothing_weight,
            )
            _write_cell_values_to_nodes(
                self.node_flat_cells,
                self.node_cell_indices,
                self.node_count,
                self.node_has_cells,
                cell_monitor,
                self.monitor.dat.data,
            )
        return self.monitor, cell_monitor


def _build_adapter_monitor(
    *,
    mesh,
    cells: np.ndarray,
    velocity,
    monitor_kind: str,
    monitor_build: str,
    monitor_scale: float,
    graph_smoothing_steps: int,
    graph_smoothing_weight: float,
    cheap_monitor_builder: _CheapMonitorBuilder | None,
) -> tuple[fd.Function, np.ndarray | None, _CheapMonitorBuilder | None]:
    if monitor_build == "firedrake-smoothed":
        monitor = _um2n_monge_ampere_monitor(
            mesh,
            velocity,
            monitor_kind=monitor_kind,
            monitor_scale=monitor_scale,
        )
        return monitor, None, cheap_monitor_builder

    if cheap_monitor_builder is None:
        cheap_monitor_builder = _CheapMonitorBuilder(
            mesh,
            cells,
            monitor_scale=monitor_scale,
            graph_smoothing_steps=graph_smoothing_steps,
            graph_smoothing_weight=graph_smoothing_weight,
        )
    monitor, cell_monitor = cheap_monitor_builder.build(velocity, monitor_build)
    return monitor, cell_monitor, cheap_monitor_builder


def _normalized_monitor_feature(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0e3)
    return clipped / max(float(clipped.max()), 1.0e-12)


def _cell_neighbor_pairs(cells: np.ndarray) -> np.ndarray:
    edge_to_cell: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for cell_index, cell in enumerate(cells):
        edges = ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))
        for a, b in edges:
            key = tuple(sorted((int(a), int(b))))
            previous = edge_to_cell.get(key)
            if previous is None:
                edge_to_cell[key] = cell_index
            else:
                pairs.append((previous, cell_index))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _cell_neighbor_scatter(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = _cell_neighbor_pairs(cells)
    if pairs.size == 0:
        empty_int = np.empty(0, dtype=np.int64)
        degree = np.zeros(cells.shape[0], dtype=np.float64)
        return empty_int, empty_int, degree, degree > 0.0
    left = pairs[:, 0]
    right = pairs[:, 1]
    targets = np.concatenate((left, right))
    sources = np.concatenate((right, left))
    degree = np.bincount(targets, minlength=cells.shape[0]).astype(np.float64)
    return targets, sources, degree, degree > 0.0


def _cell_average_from_node_columns(cell_node_columns: tuple[np.ndarray, ...], nodal_values: np.ndarray) -> np.ndarray:
    averaged = nodal_values[cell_node_columns[0]].copy()
    for column in cell_node_columns[1:]:
        averaged += nodal_values[column]
    averaged /= len(cell_node_columns)
    return averaged


def _smooth_cell_monitor(
    values: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    degree: np.ndarray,
    has_neighbors: np.ndarray,
    *,
    steps: int,
    weight: float,
) -> np.ndarray:
    if steps == 0 or weight == 0.0 or targets.size == 0:
        return values
    smoothed = values.copy()
    for _ in range(steps):
        neighbor_sum = np.bincount(targets, weights=smoothed[sources], minlength=smoothed.shape[0])
        smoothed[has_neighbors] = (1.0 - weight) * smoothed[has_neighbors] + weight * (
            neighbor_sum[has_neighbors] / degree[has_neighbors]
        )
    return smoothed


def _write_cell_values_to_nodes(
    flat_cells: np.ndarray,
    cell_indices: np.ndarray,
    node_count: np.ndarray,
    node_has_cells: np.ndarray,
    cell_values: np.ndarray,
    output: np.ndarray,
) -> None:
    node_sum = np.bincount(flat_cells, weights=cell_values[cell_indices], minlength=output.shape[0])
    output[node_has_cells] = node_sum[node_has_cells] / node_count[node_has_cells]


class _AdapterTcpClient:
    def __init__(self, host: str, port: int | None) -> None:
        if port is None:
            raise ValueError("adapter_tcp_port is required when adapter_transport=tcp")
        self.sock = socket.create_connection((host, port), timeout=30.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self._topology_key: tuple[tuple[int, ...], str, bytes] | None = None

    def adapt(
        self,
        *,
        points_np: np.ndarray,
        triangles_np: np.ndarray,
        monitor_np: np.ndarray | None,
        cell_monitor_np: np.ndarray | None,
        adapter_steps: int,
        adapter_lr: float,
        adapter_profile: str,
        adapter_dtype: str,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        arrays = {
            "points": points_np,
        }
        if monitor_np is not None:
            arrays["monitor"] = monitor_np
        topology_key = _topology_key(triangles_np)
        if self._topology_key != topology_key:
            arrays["triangles"] = triangles_np
        if cell_monitor_np is not None:
            arrays["cell_monitor"] = cell_monitor_np
        send_start = time.perf_counter()
        send_message(
            self.sock,
            scalars={
                "steps": adapter_steps,
                "lr": adapter_lr,
                "profile": adapter_profile,
                "dtype": adapter_dtype,
            },
            arrays=arrays,
        )
        response = recv_message(self.sock)
        response_wait_s = time.perf_counter() - send_start
        if response is None:
            raise RuntimeError("Adapter TCP connection closed")
        scalars, response_arrays = response
        if not bool(scalars.get("ok", False)):
            raise RuntimeError(str(scalars.get("error", "unknown adapter TCP error")))
        self._topology_key = topology_key
        return response_arrays["points"], {
            "initial_loss": float(scalars["initial_loss"]),
            "final_loss": float(scalars["final_loss"]),
            "steps_completed": int(scalars["steps_completed"]),
            "early_stopped": bool(scalars["early_stopped"]),
            "service_s": float(scalars["service_s"]),
            "response_wait_s": response_wait_s,
        }

    def close(self) -> None:
        self.sock.close()


def _topology_key(cells: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    contiguous = np.ascontiguousarray(cells)
    digest = hashlib.blake2b(memoryview(contiguous).cast("B"), digest_size=8).digest()
    return tuple(contiguous.shape), contiguous.dtype.str, digest


def _diff_adapter_coordinates(
    mesh,
    monitor: fd.Function,
    *,
    adapter_steps: int,
    adapter_lr: float,
    adapter_profile: str,
    adapter_dtype: str,
    cell_monitor_np: np.ndarray | None,
    adapter_transport: str,
    adapter_tcp_client: _AdapterTcpClient | None,
    adapter_exchange_dir: Path,
    adapter_poll_interval: float,
    request_index: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    points_np = mesh.coordinates.dat.data_ro.copy()
    triangles_np = monitor.function_space().cell_node_list.copy()
    monitor_np = None if cell_monitor_np is not None else monitor.dat.data_ro.copy()
    if adapter_transport == "tcp":
        if adapter_tcp_client is None:
            raise ValueError("adapter_tcp_client is required when adapter_transport=tcp")
        return adapter_tcp_client.adapt(
            points_np=points_np,
            triangles_np=triangles_np,
            monitor_np=monitor_np,
            cell_monitor_np=cell_monitor_np,
            adapter_steps=adapter_steps,
            adapter_lr=adapter_lr,
            adapter_profile=adapter_profile,
            adapter_dtype=adapter_dtype,
        )
    if adapter_transport != "npz":
        raise ValueError(f"Unknown adapter transport: {adapter_transport}")

    request_path = adapter_exchange_dir / f"request_{request_index:06d}.npz"
    response_path = adapter_exchange_dir / f"response_{request_index:06d}.npz"
    tmp_path = adapter_exchange_dir / f"request_{request_index:06d}.tmp"
    arrays = {
        "points": points_np,
        "triangles": triangles_np,
        "steps": np.array(adapter_steps, dtype=np.int64),
        "lr": np.array(adapter_lr, dtype=np.float64),
        "profile": np.array(adapter_profile),
        "dtype": np.array(adapter_dtype),
    }
    if monitor_np is not None:
        arrays["monitor"] = monitor_np
    if cell_monitor_np is not None:
        arrays["cell_monitor"] = cell_monitor_np
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
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


def _adapter_steps_for_preset(preset: str, custom_steps: int) -> int:
    if preset == "custom":
        return custom_steps
    if preset == "accurate":
        return 12
    if preset == "fast":
        return 6
    if preset == "faster":
        return 4
    raise ValueError(f"Unknown adapter preset: {preset}")


if __name__ == "__main__":
    main()
