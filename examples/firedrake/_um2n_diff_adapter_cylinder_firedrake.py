from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

import firedrake as fd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
_HYBRID_SPEC = importlib.util.spec_from_file_location(
    "_diff_mesh_adapter_hybrid_monitor",
    SRC / "diff_mesh_adapter" / "hybrid_monitor.py",
)
if _HYBRID_SPEC is None or _HYBRID_SPEC.loader is None:
    raise ImportError("Unable to load diff_mesh_adapter hybrid monitor helpers")
_HYBRID_MODULE = importlib.util.module_from_spec(_HYBRID_SPEC)
sys.modules[_HYBRID_SPEC.name] = _HYBRID_MODULE
_HYBRID_SPEC.loader.exec_module(_HYBRID_MODULE)
corrected_hybrid_indicator = _HYBRID_MODULE.corrected_hybrid_indicator
hybrid_log_correction = _HYBRID_MODULE.hybrid_log_correction
hybrid_proxy_intensity = _HYBRID_MODULE.hybrid_proxy_intensity
proxy_drift = _HYBRID_MODULE.proxy_drift
should_compute_full_monitor = _HYBRID_MODULE.should_compute_full_monitor
_HESSIAN_SPEC = importlib.util.spec_from_file_location(
    "_diff_mesh_adapter_hessian_monitor",
    SRC / "diff_mesh_adapter" / "hessian_monitor.py",
)
if _HESSIAN_SPEC is None or _HESSIAN_SPEC.loader is None:
    raise ImportError("Unable to load diff_mesh_adapter Hessian monitor helpers")
_HESSIAN_MODULE = importlib.util.module_from_spec(_HESSIAN_SPEC)
_HESSIAN_SPEC.loader.exec_module(_HESSIAN_MODULE)
recover_scalar_hessian_frobenius = _HESSIAN_MODULE.recover_scalar_hessian_frobenius
recover_velocity_hessian_frobenius = _HESSIAN_MODULE.recover_velocity_hessian_frobenius
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
        choices=("firedrake-smoothed", "raw-gradient", "graph-gradient", "hessian-zz", "raw-vorticity", "composite-grad-vort", "learned-v4", "ns-residual-jump", "ns-rj-hybrid", "ns-rj-vorticity-hessian"),
        default="firedrake-smoothed",
    )
    parser.add_argument("--learned-monitor-weights", type=Path, default=None)
    parser.add_argument("--learned-monitor-field", choices=("vorticity_abs", "grad_u_norm"), default="vorticity_abs")
    parser.add_argument("--learned-monitor-feature-mode", choices=("full7", "fast5"), default="full7")
    parser.add_argument("--learned-monitor-baseline-build",
                        choices=("none", "raw-gradient", "graph-gradient", "raw-vorticity", "composite-grad-vort"),
                        default="none")
    parser.add_argument("--learned-monitor-blend", type=float, default=1.0)
    parser.add_argument("--learned-monitor-recalibrate", action="store_true")
    parser.add_argument("--monitor-scale", type=float, default=0.2)
    parser.add_argument("--monitor-graph-smoothing-steps", type=int, default=4)
    parser.add_argument("--monitor-graph-smoothing-weight", type=float, default=0.5)
    parser.add_argument("--monitor-vorticity-hessian-weight", type=float, default=2.0)
    parser.add_argument("--monitor-vorticity-abs-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-full-every", type=int, default=5)
    parser.add_argument("--hybrid-drift-threshold", type=float, default=0.5)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=5.0)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=8e-4)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only", "analytic-fast", "analytic-fast-v2", "replicator-laplace", "sobolev-transport"), default="regularized")
    parser.add_argument("--adapter-preset", choices=("custom", "accurate", "fast", "faster"), default="custom")
    parser.add_argument("--adapter-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--adapter-movement-weight", type=float, default=None)
    parser.add_argument("--adapter-smoothness-weight", type=float, default=None)
    parser.add_argument("--adapter-max-step-edge-stretch", type=float, default=None)
    parser.add_argument("--adapter-min-step-edge-compression", type=float, default=None)
    parser.add_argument("--adapter-transport", choices=("npz", "tcp"), default="npz")
    parser.add_argument("--adapter-exchange-dir", type=Path, required=True)
    parser.add_argument("--adapter-poll-interval", type=float, default=0.001)
    parser.add_argument("--adapter-tcp-host", default="host.docker.internal")
    parser.add_argument("--adapter-tcp-port", type=int, default=None)
    parser.add_argument("--adapter-torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.monitor_build == "learned-v4":
        if args.learned_monitor_weights is None:
            parser.error("--learned-monitor-weights is required when --monitor-build=learned-v4")
        if args.adapter_profile in {"replicator-laplace", "analytic-fast", "analytic-fast-v2", "sobolev-transport"}:
            parser.error(f"--monitor-build=learned-v4 requires the regularized adapter profile, not {args.adapter_profile}")
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
        monitor_vorticity_hessian_weight=args.monitor_vorticity_hessian_weight,
        monitor_vorticity_abs_weight=args.monitor_vorticity_abs_weight,
        hybrid_full_every=args.hybrid_full_every,
        hybrid_drift_threshold=args.hybrid_drift_threshold,
        learned_monitor_weights=args.learned_monitor_weights,
        learned_monitor_field=args.learned_monitor_field,
        learned_monitor_feature_mode=args.learned_monitor_feature_mode,
        learned_monitor_baseline_build=args.learned_monitor_baseline_build,
        learned_monitor_blend=args.learned_monitor_blend,
        learned_monitor_recalibrate=args.learned_monitor_recalibrate,
        adaptation_relaxation=args.adaptation_relaxation,
        max_grid_speed_limit=args.max_grid_speed,
        adapter_steps=adapter_steps,
        adapter_lr=args.adapter_lr,
        adapter_profile=args.adapter_profile,
        adapter_preset=args.adapter_preset,
        adapter_dtype=args.adapter_dtype,
        adapter_movement_weight=args.adapter_movement_weight,
        adapter_smoothness_weight=args.adapter_smoothness_weight,
        adapter_max_step_edge_stretch=args.adapter_max_step_edge_stretch,
        adapter_min_step_edge_compression=args.adapter_min_step_edge_compression,
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
    monitor_vorticity_hessian_weight: float,
    monitor_vorticity_abs_weight: float,
    hybrid_full_every: int,
    hybrid_drift_threshold: float,
    learned_monitor_weights: Path | None,
    learned_monitor_field: str,
    learned_monitor_feature_mode: str,
    learned_monitor_baseline_build: str,
    learned_monitor_blend: float,
    learned_monitor_recalibrate: bool,
    adaptation_relaxation: float,
    max_grid_speed_limit: float,
    adapter_steps: int,
    adapter_lr: float,
    adapter_profile: str,
    adapter_preset: str,
    adapter_dtype: str,
    adapter_movement_weight: float | None,
    adapter_smoothness_weight: float | None,
    adapter_max_step_edge_stretch: float | None,
    adapter_min_step_edge_compression: float | None,
    adapter_transport: str,
    adapter_exchange_dir: Path,
    adapter_poll_interval: float,
    adapter_tcp_host: str,
    adapter_tcp_port: int | None,
    adapter_torch_threads: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_exchange_dir.mkdir(parents=True, exist_ok=True)
    adapter_effective_movement_weight = _adapter_effective_movement_weight(adapter_profile, adapter_movement_weight)
    adapter_effective_smoothness_weight = _adapter_effective_smoothness_weight(adapter_profile, adapter_smoothness_weight)
    adapter_effective_max_step_edge_stretch = _adapter_effective_max_step_edge_stretch(
        adapter_profile,
        adapter_max_step_edge_stretch,
    )
    adapter_effective_min_step_edge_compression = _adapter_effective_min_step_edge_compression(
        adapter_profile,
        adapter_min_step_edge_compression,
    )
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
        "monitor_vorticity_hessian_weight": monitor_vorticity_hessian_weight,
        "monitor_vorticity_abs_weight": monitor_vorticity_abs_weight,
        "hybrid_full_every": hybrid_full_every,
        "hybrid_drift_threshold": hybrid_drift_threshold,
        "learned_monitor_weights": str(learned_monitor_weights) if learned_monitor_weights is not None else None,
        "learned_monitor_field": learned_monitor_field,
        "learned_monitor_feature_mode": learned_monitor_feature_mode,
        "learned_monitor_baseline_build": learned_monitor_baseline_build,
        "learned_monitor_blend": learned_monitor_blend,
        "learned_monitor_recalibrate": learned_monitor_recalibrate,
        "adaptation_relaxation": adaptation_relaxation,
        "max_grid_speed_limit": max_grid_speed_limit,
        "adapter_steps": adapter_steps,
        "adapter_lr": adapter_lr,
        "adapter_profile": adapter_profile,
        "adapter_preset": adapter_preset,
        "adapter_dtype": adapter_dtype,
        "adapter_movement_weight": adapter_effective_movement_weight,
        "adapter_smoothness_weight": adapter_effective_smoothness_weight,
        "adapter_max_step_edge_stretch": adapter_effective_max_step_edge_stretch,
        "adapter_min_step_edge_compression": adapter_effective_min_step_edge_compression,
        "adapter_movement_weight_override": adapter_movement_weight,
        "adapter_smoothness_weight_override": adapter_smoothness_weight,
        "adapter_max_step_edge_stretch_override": adapter_max_step_edge_stretch,
        "adapter_min_step_edge_compression_override": adapter_min_step_edge_compression,
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
        "timings_s": {
            "solve": 0.0,
            "monitor": 0.0,
            "monitor_residual": 0.0,
            "monitor_jump": 0.0,
            "monitor_divergence": 0.0,
            "monitor_exact_assemble": 0.0,
            "monitor_smoothing": 0.0,
            "monitor_proxy": 0.0,
            "monitor_full": 0.0,
            "monitor_cold_compile": 0.0,
            "monitor_vorticity_hessian": 0.0,
            "adapter": 0.0,
            "adapter_forward_loss": 0.0,
            "adapter_backward_or_grad": 0.0,
            "adapter_sobolev_solve": 0.0,
            "adapter_validation": 0.0,
            "adapter_step": 0.0,
            "adapter_warmup": 0.0,
            "ale_update": 0.0,
            "projection": 0.0,
        },
    }

    _write_state(vtk, state, init_coord, time_value=0.0, monitor=None)
    monitor_val = fd.Function(fd.FunctionSpace(mesh, "CG", 1), name="monitor")
    cheap_monitor_builder: _CheapMonitorBuilder | None = None
    adapter_tcp_client = _AdapterTcpClient(adapter_tcp_host, adapter_tcp_port) if adapter_transport == "tcp" else None
    if adapter_tcp_client is not None and learned_monitor_weights is None:
        warmup_start = time.perf_counter()
        _warmup_adapter_tcp_client(
            adapter_tcp_client,
            points_np=mesh.coordinates.dat.data_ro.copy(),
            triangles_np=triangles.copy(),
            adapter_profile=adapter_profile,
            adapter_lr=adapter_lr,
            adapter_dtype=adapter_dtype,
        )
        metrics["timings_s"]["adapter_warmup"] = time.perf_counter() - warmup_start
    max_grid_speed = 0.0
    t = 0.0

    try:
        for step in range(1, steps + 1):
            adaptation_record: dict[str, object] | None = None
            if step % adapt_every == 0:
                current_coord = mesh.coordinates.dat.data_ro.copy()
                projection_start = time.perf_counter()
                if monitor_frame == "reference":
                    if monitor_build in {"ns-residual-jump", "ns-rj-hybrid", "ns-rj-vorticity-hessian"}:
                        raise ValueError(f"{monitor_build} currently supports --monitor-frame=current only")
                    adapted_mesh.coordinates.dat.data[:] = current_coord
                    mesh.coordinates.dat.data[:] = init_coord
                    u_for_monitor = fd.Function(state.V)
                    adapted_state["u"].dat.data[:] = state.u_now.dat.data_ro
                    u_for_monitor.project(adapted_state["u"])
                    p_for_monitor = fd.Function(state.Q)
                    adapted_state["p"].dat.data[:] = state.p_now.dat.data_ro
                    p_for_monitor.project(adapted_state["p"])
                elif monitor_frame == "current":
                    u_for_monitor = state.u_now
                    p_for_monitor = state.p_now
                else:
                    raise ValueError(f"Unknown monitor_frame: {monitor_frame}")
                projection_elapsed = time.perf_counter() - projection_start
                metrics["timings_s"]["projection"] += projection_elapsed

                monitor_start = time.perf_counter()
                monitor_val, cell_monitor_data, learned_u_nodes_data, cheap_monitor_builder, monitor_timings = _build_adapter_monitor(
                    mesh=mesh,
                    cells=triangles,
                    velocity=u_for_monitor,
                    pressure=p_for_monitor,
                    nu=state.nu,
                    monitor_kind=monitor_kind,
                    monitor_build=monitor_build,
                    monitor_scale=monitor_scale,
                    graph_smoothing_steps=monitor_graph_smoothing_steps,
                    graph_smoothing_weight=monitor_graph_smoothing_weight,
                    hybrid_full_every=hybrid_full_every,
                    hybrid_drift_threshold=hybrid_drift_threshold,
                    monitor_vorticity_hessian_weight=monitor_vorticity_hessian_weight,
                    monitor_vorticity_abs_weight=monitor_vorticity_abs_weight,
                    adaptation_index=len(metrics["adaptations"]),
                    previous_adaptation=metrics["adaptations"][-1] if metrics["adaptations"] else None,
                    learned_monitor_field=learned_monitor_field,
                    learned_monitor_baseline_build=learned_monitor_baseline_build,
                    cheap_monitor_builder=cheap_monitor_builder,
                )
                monitor_elapsed = time.perf_counter() - monitor_start
                metrics["timings_s"]["monitor"] += monitor_elapsed
                for timing_key, timing_value in monitor_timings.items():
                    if timing_key in metrics["timings_s"]:
                        metrics["timings_s"][timing_key] += timing_value
                monitor_data = monitor_val.dat.data_ro

                adapter_start = time.perf_counter()
                raw_target_coord, adapter_info = _diff_adapter_coordinates(
                    mesh,
                    monitor_val,
                    adapter_steps=adapter_steps,
                    adapter_lr=adapter_lr,
                    adapter_profile=adapter_profile,
                    adapter_dtype=adapter_dtype,
                    adapter_movement_weight=adapter_movement_weight,
                    adapter_smoothness_weight=adapter_smoothness_weight,
                    adapter_max_step_edge_stretch=adapter_max_step_edge_stretch,
                    adapter_min_step_edge_compression=adapter_min_step_edge_compression,
                    cell_monitor_np=cell_monitor_data,
                    learned_u_nodes_np=learned_u_nodes_data,
                    learned_weights_path=str(learned_monitor_weights) if learned_monitor_weights is not None else None,
                    learned_feature_mode=learned_monitor_feature_mode,
                    learned_recalibrate=learned_monitor_recalibrate,
                    learned_blend=learned_monitor_blend,
                    learned_smoothing_steps=monitor_graph_smoothing_steps,
                    learned_smoothing_weight=monitor_graph_smoothing_weight,
                    adapter_transport=adapter_transport,
                    adapter_tcp_client=adapter_tcp_client,
                    adapter_exchange_dir=adapter_exchange_dir,
                    adapter_poll_interval=adapter_poll_interval,
                    request_index=len(metrics["adaptations"]),
                )
                adapter_elapsed = time.perf_counter() - adapter_start
                metrics["timings_s"]["adapter"] += adapter_elapsed
                for timing_key in ("adapter_forward_loss", "adapter_backward_or_grad", "adapter_sobolev_solve", "adapter_validation", "adapter_step"):
                    metrics["timings_s"][timing_key] += float(adapter_info.get(timing_key, 0.0))

                ale_update_start = time.perf_counter()
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
                ale_update_elapsed = time.perf_counter() - ale_update_start
                metrics["timings_s"]["ale_update"] += ale_update_elapsed

                adaptation_record = {
                    "step": step,
                    "time_start": float(t),
                    "time_end": float(t + dt),
                    "monitor_build": monitor_build,
                    "adapter_loss_initial": adapter_info["initial_loss"],
                    "adapter_loss_final": adapter_info["final_loss"],
                    "adapter_steps_completed": adapter_info["steps_completed"],
                    "adapter_early_stopped": adapter_info["early_stopped"],
                    "projection_s": projection_elapsed,
                    "monitor_s": monitor_elapsed,
                    "monitor_residual_s": monitor_timings.get("monitor_residual", 0.0),
                    "monitor_jump_s": monitor_timings.get("monitor_jump", 0.0),
                    "monitor_divergence_s": monitor_timings.get("monitor_divergence", 0.0),
                    "monitor_exact_assemble_s": monitor_timings.get("monitor_exact_assemble", 0.0),
                    "monitor_smoothing_s": monitor_timings.get("monitor_smoothing", 0.0),
                    "monitor_proxy_s": monitor_timings.get("monitor_proxy", 0.0),
                    "monitor_full_s": monitor_timings.get("monitor_full", 0.0),
                    "monitor_cold_compile_s": monitor_timings.get("monitor_cold_compile", 0.0),
                    "monitor_vorticity_hessian_s": monitor_timings.get("monitor_vorticity_hessian", 0.0),
                    "monitor_used_full": bool(monitor_timings.get("monitor_used_full", False)),
                    "monitor_correction_age": int(monitor_timings.get("monitor_correction_age", 0)),
                    "monitor_drift": float(monitor_timings.get("monitor_drift", 0.0)),
                    "monitor_full_reason": str(monitor_timings.get("monitor_full_reason", "")),
                    "adapter_s": adapter_elapsed,
                    "adapter_service_s": adapter_info["service_s"],
                    "adapter_exchange_overhead_s": max(0.0, adapter_elapsed - adapter_info["service_s"]),
                    "adapter_response_wait_s": adapter_info["response_wait_s"],
                    "adapter_forward_loss_s": adapter_info.get("adapter_forward_loss", 0.0),
                    "adapter_backward_or_grad_s": adapter_info.get("adapter_backward_or_grad", 0.0),
                    "adapter_sobolev_solve_s": adapter_info.get("adapter_sobolev_solve", 0.0),
                    "adapter_validation_s": adapter_info.get("adapter_validation", 0.0),
                    "adapter_step_s": adapter_info.get("adapter_step", 0.0),
                    "adapter_alpha": adapter_info.get("adapter_alpha", 0.0),
                    "adapter_cg_iters": adapter_info.get("adapter_cg_iters", 0.0),
                    "adapter_cg_residual": adapter_info.get("adapter_cg_residual", 0.0),
                    "ale_update_s": ale_update_elapsed,
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
            solve_elapsed = time.perf_counter() - solve_start
            metrics["timings_s"]["solve"] += solve_elapsed
            t += dt
            state.u_now.assign(state.u_next)
            state.p_now.assign(state.p_next)
            if adaptation_record is not None:
                adaptation_record["solve_s"] = solve_elapsed
                adaptation_record["step_total_s"] = (
                    float(adaptation_record.get("projection_s", 0.0))
                    + float(adaptation_record.get("monitor_s", 0.0))
                    + float(adaptation_record.get("adapter_s", 0.0))
                    + float(adaptation_record.get("ale_update_s", 0.0))
                    + solve_elapsed
                )
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
        self.vector_function_space = fd.VectorFunctionSpace(mesh, "CG", 1)
        self.cell_space = fd.FunctionSpace(mesh, "DG", 0)
        self.velocity_p1 = fd.Function(self.vector_function_space)
        self.grad_norm = fd.Function(self.function_space)
        self.vorticity = fd.Function(self.function_space)
        self.learned_scalar = fd.Function(self.function_space)
        self.proxy_strain = fd.Function(self.function_space)
        self.proxy_divergence = fd.Function(self.function_space)
        self.monitor = fd.Function(self.function_space, name="monitor")
        self._cell_area = _assemble_cell_integrals(fd.Constant(1.0), self.cell_space)
        self._ns_indicator = None
        self._ns_forms_key: tuple[int, int, int] | None = None
        self._ns_forms = None
        self._ns_forms_cold = False
        self._ns_form_build_s = 0.0
        self._hybrid_log_correction: np.ndarray | None = None
        self._hybrid_proxy_at_last_full: np.ndarray | None = None
        self._hybrid_last_full_index: int | None = None
        self._hybrid_last_full_area_ratio: float | None = None

    def build(self, velocity, monitor_build: str) -> tuple[fd.Function, np.ndarray]:
        if monitor_build == "hessian-zz":
            self.velocity_p1.project(velocity)
            hessian_norm = recover_velocity_hessian_frobenius(
                self.monitor.function_space().mesh().coordinates.dat.data_ro,
                self.cells,
                self.velocity_p1.dat.data_ro,
            )
            cell_monitor = 1.0 + self.monitor_scale * _normalized_monitor_feature(hessian_norm)
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

    def build_ns_residual_jump(self, velocity, pressure, nu) -> tuple[fd.Function, np.ndarray, dict[str, float]]:
        cell_indicator, timings = self._build_full_ns_rj_indicator(velocity, pressure, nu)
        cell_monitor = self._cell_monitor_from_indicator(cell_indicator, timings)
        timings["monitor_used_full"] = 1.0
        timings["monitor_proxy"] = 0.0
        timings["monitor_correction_age"] = 0.0
        timings["monitor_drift"] = 0.0
        timings["monitor_full_reason"] = "exact"
        _write_cell_values_to_nodes(
            self.node_flat_cells,
            self.node_cell_indices,
            self.node_count,
            self.node_has_cells,
            cell_monitor,
            self.monitor.dat.data,
        )
        return self.monitor, cell_monitor, timings

    def build_ns_rj_vorticity_hessian(
            self,
            velocity,
            pressure,
            nu,
            *,
            hessian_weight: float,
            abs_weight: float,
    ) -> tuple[fd.Function, np.ndarray, dict[str, float]]:
        if hessian_weight < 0.0 or abs_weight < 0.0:
            raise ValueError("vorticity-Hessian monitor weights must be non-negative")

        cell_indicator, timings = self._build_full_ns_rj_indicator(velocity, pressure, nu)
        vorticity_start = time.perf_counter()
        self.vorticity.project(fd.curl(velocity))
        omega_nodes = self.vorticity.dat.data_ro.copy()
        omega_cell = _cell_average_from_node_columns(self.cell_node_columns, np.abs(omega_nodes))
        omega_hessian = recover_scalar_hessian_frobenius(
            self.monitor.function_space().mesh().coordinates.dat.data_ro,
            self.cells,
            omega_nodes,
        )
        hessian_indicator = np.maximum(self._cell_area, 1.0e-14) * omega_hessian
        z_rj = _normalized_quantile_feature(cell_indicator)
        z_hw = _normalized_quantile_feature(hessian_indicator)
        z_w = _normalized_quantile_feature(omega_cell)
        combined_indicator = np.sqrt(z_rj * z_rj + hessian_weight * z_hw * z_hw + abs_weight * z_w * z_w)
        timings["monitor_vorticity_hessian"] = time.perf_counter() - vorticity_start
        timings["monitor_used_full"] = 1.0
        timings["monitor_proxy"] = 0.0
        timings["monitor_correction_age"] = 0.0
        timings["monitor_drift"] = 0.0
        timings["monitor_full_reason"] = "exact+vorticity-hessian"
        cell_monitor = self._cell_monitor_from_indicator(combined_indicator, timings)
        _write_cell_values_to_nodes(
            self.node_flat_cells,
            self.node_cell_indices,
            self.node_count,
            self.node_has_cells,
            cell_monitor,
            self.monitor.dat.data,
        )
        return self.monitor, cell_monitor, timings

    def build_ns_rj_hybrid(
            self,
            velocity,
            pressure,
            nu,
            *,
            adaptation_index: int,
            previous_adaptation: dict[str, object] | None,
            full_every: int,
            drift_threshold: float,
    ) -> tuple[fd.Function, np.ndarray, dict[str, float]]:
        proxy_start = time.perf_counter()
        proxy_indicator = self._hybrid_proxy_indicator(velocity)
        proxy_elapsed = time.perf_counter() - proxy_start
        self._update_hybrid_area_baseline(previous_adaptation)
        previous_rejected = bool(previous_adaptation is not None and not previous_adaptation.get("accepted_adaptation", True))
        previous_flips = False
        if previous_adaptation is not None:
            raw_quality = previous_adaptation.get("raw_target_mesh_quality", {}) or {}
            relaxed_quality = previous_adaptation.get("relaxed_target_mesh_quality", {}) or {}
            previous_flips = bool(
                raw_quality.get("orientation_flips", 0) > 0
                or relaxed_quality.get("orientation_flips", 0) > 0
            )
        area_ratio_worsened = self._hybrid_area_ratio_worsened(previous_adaptation)
        drift = proxy_drift(proxy_indicator, self._hybrid_proxy_at_last_full)
        decision = should_compute_full_monitor(
            adaptation_index=adaptation_index,
            full_every=full_every,
            drift=drift,
            drift_threshold=drift_threshold,
            has_correction=self._hybrid_log_correction is not None,
            previous_rejected=previous_rejected,
            previous_flips=previous_flips,
            area_ratio_worsened=area_ratio_worsened,
        )
        timings = {
            "monitor_proxy": proxy_elapsed,
            "monitor_used_full": 1.0 if decision.use_full else 0.0,
            "monitor_correction_age": float(decision.age),
            "monitor_drift": 0.0 if not np.isfinite(decision.drift) else float(decision.drift),
            "monitor_full_reason": decision.reason,
            "monitor_residual": 0.0,
            "monitor_jump": 0.0,
            "monitor_divergence": 0.0,
            "monitor_exact_assemble": 0.0,
            "monitor_smoothing": 0.0,
            "monitor_full": 0.0,
            "monitor_cold_compile": 0.0,
        }
        if decision.use_full:
            full_indicator, full_timings = self._build_full_ns_rj_indicator(velocity, pressure, nu)
            timings.update(full_timings)
            log_correction = hybrid_log_correction(full_indicator, proxy_indicator)
            if self.graph_smoothing_steps > 0 and self.graph_smoothing_weight > 0.0:
                log_correction = _smooth_cell_monitor(
                    log_correction,
                    self.smoothing_targets,
                    self.smoothing_sources,
                    self.smoothing_degree,
                    self.smoothing_has_neighbors,
                    steps=self.graph_smoothing_steps,
                    weight=self.graph_smoothing_weight,
                )
            self._hybrid_log_correction = log_correction
            self._hybrid_proxy_at_last_full = proxy_indicator.copy()
            self._hybrid_last_full_index = adaptation_index
            cell_indicator = full_indicator
        else:
            age = (
                adaptation_index - self._hybrid_last_full_index
                if self._hybrid_last_full_index is not None
                else decision.age
            )
            timings["monitor_correction_age"] = float(age)
            cell_indicator = corrected_hybrid_indicator(
                proxy_indicator,
                self._hybrid_log_correction,
                age=age,
            )
        cell_monitor = self._cell_monitor_from_indicator(cell_indicator, timings)
        _write_cell_values_to_nodes(
            self.node_flat_cells,
            self.node_cell_indices,
            self.node_count,
            self.node_has_cells,
            cell_monitor,
            self.monitor.dat.data,
        )
        return self.monitor, cell_monitor, timings

    def _cell_monitor_from_indicator(self, cell_indicator: np.ndarray, timings: dict[str, float]) -> np.ndarray:
        cell_monitor = _bounded_quantile_monitor(cell_indicator, monitor_scale=self.monitor_scale)
        smoothing_start = time.perf_counter()
        if self.graph_smoothing_steps > 0 and self.graph_smoothing_weight > 0.0:
            log_monitor = np.log(np.maximum(cell_monitor, 1.0e-12))
            log_monitor = _smooth_cell_monitor(
                log_monitor,
                self.smoothing_targets,
                self.smoothing_sources,
                self.smoothing_degree,
                self.smoothing_has_neighbors,
                steps=self.graph_smoothing_steps,
                weight=self.graph_smoothing_weight,
            )
            cell_monitor = np.exp(log_monitor)
        timings["monitor_smoothing"] = time.perf_counter() - smoothing_start
        return cell_monitor

    def _build_full_ns_rj_indicator(self, velocity, pressure, nu) -> tuple[np.ndarray, dict[str, float]]:
        form = self._ns_residual_jump_forms(velocity, pressure, nu)
        cell_indicator, timings, self._ns_indicator = _ns_residual_jump_indicator(
            form,
            self._ns_indicator,
            self._cell_area,
        )
        full_s = timings["monitor_exact_assemble"]
        timings["monitor_full"] = full_s
        timings["monitor_cold_compile"] = full_s + self._ns_form_build_s if self._ns_forms_cold else 0.0
        return cell_indicator, timings

    def _ns_residual_jump_forms(self, velocity, pressure, nu):
        key = (id(velocity), id(pressure), id(nu))
        if self._ns_forms_key == key and self._ns_forms is not None:
            self._ns_forms_cold = False
            self._ns_form_build_s = 0.0
            return self._ns_forms
        form_start = time.perf_counter()
        self._ns_forms_key = key
        self._ns_forms = _ns_residual_jump_forms(velocity, pressure, nu, self.cell_space)
        self._ns_form_build_s = time.perf_counter() - form_start
        self._ns_forms_cold = True
        return self._ns_forms

    def _hybrid_proxy_indicator(self, velocity) -> np.ndarray:
        grad_u = fd.grad(velocity)
        strain = fd.sym(grad_u)
        self.proxy_strain.interpolate(
            fd.sqrt(sum(strain[i, j] ** 2 for i in range(2) for j in range(2)) + 1.0e-30)
        )
        self.proxy_divergence.interpolate(fd.sqrt(fd.div(velocity) ** 2 + 1.0e-30))
        strain_cell = _cell_average_from_node_columns(self.cell_node_columns, self.proxy_strain.dat.data_ro)
        divergence_cell = _cell_average_from_node_columns(self.cell_node_columns, self.proxy_divergence.dat.data_ro)
        return hybrid_proxy_intensity(strain_cell, divergence_cell)

    def _update_hybrid_area_baseline(self, previous_adaptation: dict[str, object] | None) -> None:
        if previous_adaptation is None or not previous_adaptation.get("monitor_used_full", False):
            return
        quality = previous_adaptation.get("relaxed_target_mesh_quality", {}) or {}
        area_ratio = quality.get("area_ratio")
        if area_ratio is not None:
            self._hybrid_last_full_area_ratio = float(area_ratio)

    def _hybrid_area_ratio_worsened(self, previous_adaptation: dict[str, object] | None) -> bool:
        if previous_adaptation is None or self._hybrid_last_full_area_ratio is None:
            return False
        quality = previous_adaptation.get("relaxed_target_mesh_quality", {}) or {}
        area_ratio = quality.get("area_ratio")
        if area_ratio is None:
            return False
        return float(area_ratio) > 1.1 * self._hybrid_last_full_area_ratio

    def build_learned_field(self, velocity, learned_monitor_field: str) -> tuple[fd.Function, np.ndarray]:
        if learned_monitor_field == "vorticity_abs":
            self.learned_scalar.interpolate(fd.sqrt(fd.curl(velocity) ** 2 + 1.0e-30))
        elif learned_monitor_field == "grad_u_norm":
            grad_u = fd.grad(velocity)
            self.learned_scalar.interpolate(fd.sqrt(sum(grad_u[i, j] ** 2 for i in range(2) for j in range(2)) + 1.0e-30))
        else:
            raise ValueError(f"Unknown learned monitor field: {learned_monitor_field}")
        raw = np.array(self.learned_scalar.dat.data_ro, copy=True)
        self.monitor.dat.data[:] = 1.0 + self.monitor_scale * _normalized_monitor_feature(raw)
        return self.monitor, raw


def _build_adapter_monitor(
    *,
    mesh,
    cells: np.ndarray,
    velocity,
    pressure,
    nu,
    monitor_kind: str,
    monitor_build: str,
    monitor_scale: float,
    graph_smoothing_steps: int,
    graph_smoothing_weight: float,
    hybrid_full_every: int,
    hybrid_drift_threshold: float,
    monitor_vorticity_hessian_weight: float,
    monitor_vorticity_abs_weight: float,
    adaptation_index: int,
    previous_adaptation: dict[str, object] | None,
    learned_monitor_field: str,
    learned_monitor_baseline_build: str,
    cheap_monitor_builder: _CheapMonitorBuilder | None,
) -> tuple[fd.Function, np.ndarray | None, np.ndarray | None, _CheapMonitorBuilder | None, dict[str, float]]:
    if monitor_build == "firedrake-smoothed":
        monitor = _um2n_monge_ampere_monitor(
            mesh,
            velocity,
            monitor_kind=monitor_kind,
            monitor_scale=monitor_scale,
        )
        return monitor, None, None, cheap_monitor_builder, {}

    if cheap_monitor_builder is None:
        cheap_monitor_builder = _CheapMonitorBuilder(
            mesh,
            cells,
            monitor_scale=monitor_scale,
            graph_smoothing_steps=graph_smoothing_steps,
            graph_smoothing_weight=graph_smoothing_weight,
        )
    if monitor_build == "learned-v4":
        monitor, learned_u_nodes = cheap_monitor_builder.build_learned_field(velocity, learned_monitor_field)
        cell_monitor = None
        if learned_monitor_baseline_build != "none":
            _baseline_monitor, cell_monitor = cheap_monitor_builder.build(velocity, learned_monitor_baseline_build)
            monitor.dat.data[:] = _baseline_monitor.dat.data_ro
        return monitor, cell_monitor, learned_u_nodes, cheap_monitor_builder, {}
    if monitor_build == "ns-residual-jump":
        monitor, cell_monitor, timings = cheap_monitor_builder.build_ns_residual_jump(velocity, pressure, nu)
        return monitor, cell_monitor, None, cheap_monitor_builder, timings
    if monitor_build == "ns-rj-vorticity-hessian":
        monitor, cell_monitor, timings = cheap_monitor_builder.build_ns_rj_vorticity_hessian(
            velocity,
            pressure,
            nu,
            hessian_weight=monitor_vorticity_hessian_weight,
            abs_weight=monitor_vorticity_abs_weight,
        )
        return monitor, cell_monitor, None, cheap_monitor_builder, timings
    if monitor_build == "ns-rj-hybrid":
        monitor, cell_monitor, timings = cheap_monitor_builder.build_ns_rj_hybrid(
            velocity,
            pressure,
            nu,
            adaptation_index=adaptation_index,
            previous_adaptation=previous_adaptation,
            full_every=hybrid_full_every,
            drift_threshold=hybrid_drift_threshold,
        )
        return monitor, cell_monitor, None, cheap_monitor_builder, timings
    monitor, cell_monitor = cheap_monitor_builder.build(velocity, monitor_build)
    return monitor, cell_monitor, None, cheap_monitor_builder, {}


def _normalized_monitor_feature(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0e3)
    return clipped / max(float(clipped.max()), 1.0e-12)


def _normalized_quantile_feature(values: np.ndarray) -> np.ndarray:
    safe = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return safe / max(float(np.quantile(safe, 0.90)), 1.0e-12)


def _ns_residual_jump_forms(velocity, pressure, nu, cell_space):
    mesh = cell_space.mesh()
    q = fd.TestFunction(cell_space)
    n = fd.FacetNormal(mesh)
    h = fd.CellDiameter(mesh)
    h_facet = fd.FacetArea(mesh)
    dim = mesh.geometric_dimension
    stress = 2.0 * nu * fd.sym(fd.nabla_grad(velocity)) - pressure * fd.Identity(dim)
    residual = fd.dot(velocity, fd.nabla_grad(velocity)) - fd.div(stress)
    cell_form = (h**2 * fd.inner(residual, residual) + fd.div(velocity) ** 2) * q * fd.dx(domain=mesh)
    jump_form = (
        0.5
        * h_facet
        * fd.inner(fd.jump(stress, n), fd.jump(stress, n))
        * (q("+") + q("-"))
        * fd.dS(domain=mesh)
    )
    return cell_form + jump_form


def _ns_residual_jump_indicator(form, output_function, cell_area: np.ndarray):
    timings = {
        "monitor_residual": 0.0,
        "monitor_jump": 0.0,
        "monitor_divergence": 0.0,
        "monitor_exact_assemble": 0.0,
    }
    assemble_start = time.perf_counter()
    if output_function is None:
        output_function = fd.assemble(form)
    else:
        fd.assemble(form, tensor=output_function)
    timings["monitor_exact_assemble"] = time.perf_counter() - assemble_start
    eta_sq = np.maximum(output_function.dat.data_ro.copy(), 0.0)
    return np.sqrt(eta_sq / np.maximum(cell_area, 1.0e-14)), timings, output_function


def _assemble_cell_integrals(expr, cell_space) -> np.ndarray:
    q = fd.TestFunction(cell_space)
    return fd.assemble(expr * q * fd.dx(domain=cell_space.mesh())).dat.data_ro.copy()


def _bounded_quantile_monitor(indicator: np.ndarray, *, monitor_scale: float) -> np.ndarray:
    safe_indicator = np.maximum(np.asarray(indicator, dtype=np.float64), 0.0)
    quantile = max(float(np.quantile(safe_indicator, 0.90)), 1.0e-12)
    normalized_sq = (safe_indicator / quantile) ** 2
    return 1.0 + monitor_scale * normalized_sq / (1.0 + normalized_sq)


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
        learned_u_nodes_np: np.ndarray | None,
        learned_weights_path: str | None,
        learned_feature_mode: str,
        learned_recalibrate: bool,
        learned_blend: float,
        learned_smoothing_steps: int,
        learned_smoothing_weight: float,
        adapter_steps: int,
        adapter_lr: float,
        adapter_profile: str,
        adapter_dtype: str,
        adapter_movement_weight: float | None,
        adapter_smoothness_weight: float | None,
        adapter_max_step_edge_stretch: float | None,
        adapter_min_step_edge_compression: float | None,
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
        if learned_u_nodes_np is not None:
            arrays["learned_u_nodes"] = learned_u_nodes_np
        send_start = time.perf_counter()
        scalars = {
            "steps": adapter_steps,
            "lr": adapter_lr,
            "profile": adapter_profile,
            "dtype": adapter_dtype,
            "learned_weights_path": learned_weights_path,
            "learned_feature_mode": learned_feature_mode,
            "learned_recalibrate": learned_recalibrate,
            "learned_blend": learned_blend,
            "learned_smoothing_steps": learned_smoothing_steps,
            "learned_smoothing_weight": learned_smoothing_weight,
        }
        if adapter_movement_weight is not None:
            scalars["adapter_movement_weight"] = adapter_movement_weight
        if adapter_smoothness_weight is not None:
            scalars["adapter_smoothness_weight"] = adapter_smoothness_weight
        if adapter_max_step_edge_stretch is not None:
            scalars["adapter_max_step_edge_stretch"] = adapter_max_step_edge_stretch
        if adapter_min_step_edge_compression is not None:
            scalars["adapter_min_step_edge_compression"] = adapter_min_step_edge_compression
        send_message(
            self.sock,
            scalars=scalars,
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
            "adapter_forward_loss": float(scalars.get("adapter_forward_loss", 0.0)),
            "adapter_backward_or_grad": float(scalars.get("adapter_backward_or_grad", 0.0)),
            "adapter_sobolev_solve": float(scalars.get("adapter_sobolev_solve", 0.0)),
            "adapter_validation": float(scalars.get("adapter_validation", 0.0)),
            "adapter_step": float(scalars.get("adapter_step", 0.0)),
            "adapter_alpha": float(scalars.get("adapter_alpha", 0.0)),
            "adapter_cg_iters": float(scalars.get("adapter_cg_iters", 0.0)),
            "adapter_cg_residual": float(scalars.get("adapter_cg_residual", 0.0)),
        }

    def close(self) -> None:
        self.sock.close()


def _warmup_adapter_tcp_client(
    adapter_tcp_client: _AdapterTcpClient,
    *,
    points_np: np.ndarray,
    triangles_np: np.ndarray,
    adapter_profile: str,
    adapter_lr: float,
    adapter_dtype: str,
) -> None:
    adapter_tcp_client.adapt(
        points_np=points_np,
        triangles_np=triangles_np,
        monitor_np=None,
        cell_monitor_np=np.ones(triangles_np.shape[0], dtype=points_np.dtype),
        learned_u_nodes_np=None,
        learned_weights_path=None,
        learned_feature_mode="full7",
        learned_recalibrate=False,
        learned_blend=1.0,
        learned_smoothing_steps=0,
        learned_smoothing_weight=0.0,
        adapter_steps=0,
        adapter_lr=adapter_lr,
        adapter_profile=adapter_profile,
        adapter_dtype=adapter_dtype,
        adapter_movement_weight=None,
        adapter_smoothness_weight=None,
        adapter_max_step_edge_stretch=None,
        adapter_min_step_edge_compression=None,
    )


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
    adapter_movement_weight: float | None,
    adapter_smoothness_weight: float | None,
    adapter_max_step_edge_stretch: float | None,
    adapter_min_step_edge_compression: float | None,
    cell_monitor_np: np.ndarray | None,
    learned_u_nodes_np: np.ndarray | None,
    learned_weights_path: str | None,
    learned_feature_mode: str,
    learned_recalibrate: bool,
    learned_blend: float,
    learned_smoothing_steps: int,
    learned_smoothing_weight: float,
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
            learned_u_nodes_np=learned_u_nodes_np,
            learned_weights_path=learned_weights_path,
            learned_feature_mode=learned_feature_mode,
            learned_recalibrate=learned_recalibrate,
            learned_blend=learned_blend,
            learned_smoothing_steps=learned_smoothing_steps,
            learned_smoothing_weight=learned_smoothing_weight,
            adapter_steps=adapter_steps,
            adapter_lr=adapter_lr,
            adapter_profile=adapter_profile,
            adapter_dtype=adapter_dtype,
            adapter_movement_weight=adapter_movement_weight,
            adapter_smoothness_weight=adapter_smoothness_weight,
            adapter_max_step_edge_stretch=adapter_max_step_edge_stretch,
            adapter_min_step_edge_compression=adapter_min_step_edge_compression,
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
    if adapter_movement_weight is not None:
        arrays["adapter_movement_weight"] = np.array(adapter_movement_weight, dtype=np.float64)
    if adapter_smoothness_weight is not None:
        arrays["adapter_smoothness_weight"] = np.array(adapter_smoothness_weight, dtype=np.float64)
    if adapter_max_step_edge_stretch is not None:
        arrays["adapter_max_step_edge_stretch"] = np.array(adapter_max_step_edge_stretch, dtype=np.float64)
    if adapter_min_step_edge_compression is not None:
        arrays["adapter_min_step_edge_compression"] = np.array(adapter_min_step_edge_compression, dtype=np.float64)
    if learned_u_nodes_np is not None:
        arrays["learned_u_nodes"] = learned_u_nodes_np
        arrays["learned_weights_path"] = np.array(learned_weights_path or "")
        arrays["learned_feature_mode"] = np.array(learned_feature_mode)
        arrays["learned_recalibrate"] = np.array(learned_recalibrate, dtype=bool)
        arrays["learned_blend"] = np.array(learned_blend, dtype=np.float64)
        arrays["learned_smoothing_steps"] = np.array(learned_smoothing_steps, dtype=np.int64)
        arrays["learned_smoothing_weight"] = np.array(learned_smoothing_weight, dtype=np.float64)
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
        "adapter_forward_loss": float(data["adapter_forward_loss"]) if "adapter_forward_loss" in data else 0.0,
        "adapter_backward_or_grad": float(data["adapter_backward_or_grad"]) if "adapter_backward_or_grad" in data else 0.0,
        "adapter_sobolev_solve": float(data["adapter_sobolev_solve"]) if "adapter_sobolev_solve" in data else 0.0,
        "adapter_validation": float(data["adapter_validation"]) if "adapter_validation" in data else 0.0,
        "adapter_step": float(data["adapter_step"]) if "adapter_step" in data else 0.0,
        "adapter_alpha": float(data["adapter_alpha"]) if "adapter_alpha" in data else 0.0,
        "adapter_cg_iters": float(data["adapter_cg_iters"]) if "adapter_cg_iters" in data else 0.0,
        "adapter_cg_residual": float(data["adapter_cg_residual"]) if "adapter_cg_residual" in data else 0.0,
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


def _adapter_effective_movement_weight(profile: str, override: float | None) -> float | None:
    if override is not None:
        return override
    if profile in {"regularized", "analytic-fast", "analytic-fast-v2"}:
        return 2.0e-3
    if profile == "monitor-only":
        return 0.0
    return None


def _adapter_effective_smoothness_weight(profile: str, override: float | None) -> float | None:
    if override is not None:
        return override
    if profile in {"regularized", "analytic-fast", "analytic-fast-v2"}:
        return 5.0e-2
    if profile == "monitor-only":
        return 0.0
    return None


def _adapter_effective_max_step_edge_stretch(profile: str, override: float | None) -> float | None:
    if override is not None:
        return override
    if profile == "analytic-fast-v2":
        return 1.8
    if profile in {"regularized", "analytic-fast", "sobolev-transport"}:
        return 2.2
    return None


def _adapter_effective_min_step_edge_compression(profile: str, override: float | None) -> float | None:
    if override is not None:
        return override
    if profile == "analytic-fast-v2":
        return 0.45
    if profile in {"regularized", "analytic-fast", "sobolev-transport"}:
        return 0.3
    return None


if __name__ == "__main__":
    main()
