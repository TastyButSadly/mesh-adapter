from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.firedrake.multi_cylinder_meshes import CASES
from examples.firedrake.postprocess_multi_cylinder_benchmark import _load_series, _summary
from examples.firedrake.v_formation_compare import _fine_mesh_error_row


METHODS = ("fixed", "monge-ampere", "mmpde-winslow", "gd-nsrj")
DEFAULT_H_VALUES = (0.090, 0.075, 0.063, 0.053, 0.045, 0.038, 0.032)
PVD_BY_METHOD = {
    "fixed": "um2n_diff_adapter_cylinder.pvd",
    "gd-nsrj": "um2n_diff_adapter_cylinder.pvd",
    "monge-ampere": "um2n_monge_ampere_cylinder.pvd",
    "mmpde-winslow": "um2n_monge_ampere_cylinder.pvd",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diploma Pareto sweeps on uniform multi-cylinder meshes.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/diploma_pareto_uniform"))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=("v_formation", "multiple_cylinders"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--h-values", nargs="+", type=float, default=DEFAULT_H_VALUES)
    parser.add_argument("--reference-h", type=float, default=0.020)
    parser.add_argument("--mesh-mode", choices=("uniform", "cylinder-refined"), default="uniform")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--firedrake-run", default="firedrake-run")
    parser.add_argument("--gd-adapter-profile", choices=("regularized", "monitor-only", "analytic-fast", "analytic-fast-v2", "sobolev-transport"), default="analytic-fast")
    parser.add_argument("--gd-adapter-profiles", nargs="+", choices=("regularized", "monitor-only", "analytic-fast", "analytic-fast-v2", "sobolev-transport"), default=None)
    parser.add_argument("--gd-monitor-build", choices=("ns-residual-jump", "ns-rj-hybrid", "hessian-zz", "ns-rj-vorticity-hessian"), default="ns-residual-jump")
    parser.add_argument("--hybrid-full-every", type=int, default=5)
    parser.add_argument("--hybrid-drift-threshold", type=float, default=0.5)
    parser.add_argument("--ma-monitor-continuation-levels", type=int, default=1)
    parser.add_argument(
        "--ma-monitor-kind",
        choices=("velocity-gradient", "wake-vorticity", "um2n-flow-gradient", "um2n-ma-gradient"),
        default="velocity-gradient",
    )
    parser.add_argument(
        "--ma-solution-remap",
        choices=("assign", "um2n-project", "um2n-project-zero-grid"),
        default="assign",
    )
    parser.add_argument("--ma-quality-guard", choices=("none", "no-worse-current"), default="no-worse-current")
    parser.add_argument("--ma-quality-guard-tolerance", type=float, default=0.05)
    parser.add_argument("--ma-monitor-scale", type=float, default=0.3)
    parser.add_argument("--ma-adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--ma-max-grid-speed", type=float, default=0.5)
    parser.add_argument("--ma-max-area-ratio", type=float, default=18.0)
    parser.add_argument("--ma-rtol", type=float, default=0.03)
    parser.add_argument("--ma-maxiter", type=int, default=12)
    parser.add_argument("--monitor-vorticity-hessian-weight", type=float, default=2.0)
    parser.add_argument("--monitor-vorticity-abs-weight", type=float, default=0.5)
    parser.add_argument("--adapter-movement-weight", type=float, default=None)
    parser.add_argument("--adapter-smoothness-weight", type=float, default=None)
    parser.add_argument("--adapter-max-step-edge-stretch", type=float, default=None)
    parser.add_argument("--adapter-min-step-edge-compression", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions per run for averaging")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel runs (for multi-core servers)")
    parser.add_argument("--prepare-meshes", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    manifest = _manifest(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "sweep_manifest.json", manifest)

    if args.prepare_meshes:
        _prepare_meshes(args, manifest)
    if args.run:
        _run_missing(args, manifest)
    if args.postprocess:
        summary = _postprocess(args, manifest)
        paths = _write_summary(args.output_root / "pareto_report", summary)
        print(f"points: {paths['points_csv']}")
        print(f"aggregate: {paths['aggregate_csv']}")
        print(f"overhead: {paths['overhead_csv']}")
        print(f"vorticity front: {paths['aggregate_vorticity_png']}")
        print(f"velocity front: {paths['aggregate_velocity_png']}")
    print(f"manifest: {args.output_root / 'sweep_manifest.json'}")


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    h_values = [float(value) for value in args.h_values]
    gd_adapter_profiles = tuple(args.gd_adapter_profiles or (args.gd_adapter_profile,))
    entries = []
    for case_name in args.cases:
        case_root = args.output_root / case_name
        reference_mesh = case_root / "meshes" / f"{case_name}_reference_h{_h_label(args.reference_h)}.msh"
        entries.append(
            {
                "case": case_name,
                "h": float(args.reference_h),
                "role": "reference",
                "method": "fixed",
                "mesh": str(reference_mesh),
                "run_dir": str(case_root / "reference" / "fixed"),
            }
        )
        for h_value in h_values:
            mesh = case_root / "meshes" / f"{case_name}_h{_h_label(h_value)}.msh"
            for method in args.methods:
                profiles = gd_adapter_profiles if method == "gd-nsrj" else (None,)
                for adapter_profile in profiles:
                    run_leaf = method
                    if method == "gd-nsrj" and adapter_profile != "regularized":
                        run_leaf = f"{method}_{adapter_profile}"
                    if method == "gd-nsrj" and args.gd_monitor_build != "ns-residual-jump":
                        run_leaf = f"{run_leaf}_{args.gd_monitor_build}"
                    for rep in range(args.repeats):
                        rep_dir = f"rep_{rep}" if args.repeats > 1 else ""
                        entries.append(
                            {
                                "case": case_name,
                                "h": h_value,
                                "role": "sweep",
                                "method": method,
                                "repetition": rep,
                                "adapter_profile": adapter_profile if method == "gd-nsrj" else None,
                                "monitor_build": args.gd_monitor_build if method == "gd-nsrj" else None,
                                "monitor_vorticity_hessian_weight": args.monitor_vorticity_hessian_weight if method == "gd-nsrj" else None,
                                "monitor_vorticity_abs_weight": args.monitor_vorticity_abs_weight if method == "gd-nsrj" else None,
                                "adapter_movement_weight": args.adapter_movement_weight if method == "gd-nsrj" else None,
                                "adapter_smoothness_weight": args.adapter_smoothness_weight if method == "gd-nsrj" else None,
                                "adapter_max_step_edge_stretch": args.adapter_max_step_edge_stretch if method == "gd-nsrj" else None,
                                "adapter_min_step_edge_compression": args.adapter_min_step_edge_compression if method == "gd-nsrj" else None,
                                "mesh": str(mesh),
                                "run_dir": str(case_root / f"h{_h_label(h_value)}" / run_leaf / rep_dir) if rep_dir else str(case_root / f"h{_h_label(h_value)}" / run_leaf),
                            }
                        )
    return {
        "mesh_mode": args.mesh_mode,
        "cases": list(args.cases),
        "methods": list(args.methods),
        "h_values": h_values,
        "reference_h": float(args.reference_h),
        "dt": args.dt,
        "steps": args.steps,
        "adapt_every": args.adapt_every,
        "save_every": args.save_every,
        "gd_adapter_profile": args.gd_adapter_profile,
        "gd_adapter_profiles": list(gd_adapter_profiles),
        "gd_monitor_build": args.gd_monitor_build,
        "hybrid_full_every": args.hybrid_full_every,
        "hybrid_drift_threshold": args.hybrid_drift_threshold,
        "ma_monitor_continuation_levels": args.ma_monitor_continuation_levels,
        "ma_monitor_kind": args.ma_monitor_kind,
        "ma_solution_remap": args.ma_solution_remap,
        "ma_quality_guard": args.ma_quality_guard,
        "ma_quality_guard_tolerance": args.ma_quality_guard_tolerance,
        "ma_monitor_scale": args.ma_monitor_scale,
        "ma_adaptation_relaxation": args.ma_adaptation_relaxation,
        "ma_max_grid_speed": args.ma_max_grid_speed,
        "ma_max_area_ratio": args.ma_max_area_ratio,
        "ma_rtol": args.ma_rtol,
        "ma_maxiter": args.ma_maxiter,
        "monitor_vorticity_hessian_weight": args.monitor_vorticity_hessian_weight,
        "monitor_vorticity_abs_weight": args.monitor_vorticity_abs_weight,
        "adapter_movement_weight": args.adapter_movement_weight,
        "adapter_smoothness_weight": args.adapter_smoothness_weight,
        "adapter_max_step_edge_stretch": args.adapter_max_step_edge_stretch,
        "adapter_min_step_edge_compression": args.adapter_min_step_edge_compression,
        "entries": entries,
    }


def _prepare_meshes(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    from examples.firedrake.v_formation_compare import write_uniform_multi_cylinder_mesh
    written: set[Path] = set()
    for entry in manifest["entries"]:
        mesh_path = Path(entry["mesh"])
        if mesh_path in written:
            continue
        if mesh_path.exists() and not args.force:
            written.add(mesh_path)
            continue
        preview_path = mesh_path.with_suffix(".png") if args.preview else None
        writer = write_uniform_multi_cylinder_mesh
        stats = writer(mesh_path, CASES[entry["case"]], float(entry["h"]), preview_path=preview_path)
        print(f"mesh {mesh_path}: points={stats['points']} triangles={stats['triangles']} h={entry['h']}")
        written.add(mesh_path)


def _run_missing(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    pending = []
    for entry in manifest["entries"]:
        run_dir = Path(entry["run_dir"])
        pvd_path = run_dir / PVD_BY_METHOD[entry["method"]]
        metrics_path = run_dir / "metrics.json"
        if pvd_path.exists() and metrics_path.exists() and not args.force:
            print(f"reuse {entry['case']} h={entry['h']} {entry['method']}: {run_dir}")
            continue
        pending.append(entry)

    if not pending:
        print("All runs already completed.")
        return

    if args.parallel <= 1:
        for entry in pending:
            _run_entry(args, entry)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Running {len(pending)} entries with {args.parallel} parallel workers")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_run_entry, args, entry): entry for entry in pending}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"FAILED {entry['case']} h={entry['h']} {entry['method']}: {e}")


def _run_entry(args: argparse.Namespace, entry: dict[str, Any]) -> None:
    run_dir = Path(entry["run_dir"])
    command = _command_for_entry(args, entry)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - start
    _write_json(run_dir / "wall_time.json", {"wall_time_s": elapsed})


def _command_for_entry(args: argparse.Namespace, entry: dict[str, Any]) -> list[str]:
    method = entry["method"]
    adapt_every = args.steps + 1 if method == "fixed" else args.adapt_every
    direct = not args.firedrake_run or args.firedrake_run in ("", "direct")

    if method in ("fixed", "gd-nsrj"):
        base = [
            sys.executable, "-m",
            "examples.firedrake._um2n_diff_adapter_cylinder_firedrake" if direct
            else "examples.firedrake.um2n_diff_adapter_cylinder",
            "--output-dir", entry["run_dir"],
            "--mesh", entry["mesh"],
            "--dt", str(args.dt),
            "--steps", str(args.steps),
            "--adapt-every", str(adapt_every),
            "--save-every", str(args.save_every),
            "--monitor-kind", "wake-vorticity",
            "--monitor-frame", "current",
            "--monitor-build", args.gd_monitor_build if method == "gd-nsrj" else "raw-vorticity",
            "--monitor-scale", "5.0" if method == "gd-nsrj" else "0.1",
            "--monitor-graph-smoothing-steps", "4",
            "--monitor-graph-smoothing-weight", "0.5",
            "--monitor-vorticity-hessian-weight", str(args.monitor_vorticity_hessian_weight),
            "--monitor-vorticity-abs-weight", str(args.monitor_vorticity_abs_weight),
            "--hybrid-full-every", str(args.hybrid_full_every),
            "--hybrid-drift-threshold", str(args.hybrid_drift_threshold),
            "--adapter-profile", entry.get("adapter_profile") if method == "gd-nsrj" else "regularized",
            "--adapter-steps", "1" if entry.get("adapter_profile") == "sobolev-transport" else "12",
            "--adapter-lr", "0.0008",
            "--adapter-transport", "npz",
            "--adapter-torch-threads", "1",
            "--max-grid-speed", "2.0" if method == "gd-nsrj" else "0.0",
        ]
        if not direct:
            base.extend(["--firedrake-run", args.firedrake_run])
        else:
            exchange_dir = Path(entry["run_dir"]) / "adapter_exchange"
            base.extend(["--adapter-exchange-dir", str(exchange_dir)])
        if method == "gd-nsrj":
            if args.adapter_movement_weight is not None:
                base.extend(["--adapter-movement-weight", str(args.adapter_movement_weight)])
            if args.adapter_smoothness_weight is not None:
                base.extend(["--adapter-smoothness-weight", str(args.adapter_smoothness_weight)])
            if args.adapter_max_step_edge_stretch is not None:
                base.extend(["--adapter-max-step-edge-stretch", str(args.adapter_max_step_edge_stretch)])
            if args.adapter_min_step_edge_compression is not None:
                base.extend(["--adapter-min-step-edge-compression", str(args.adapter_min_step_edge_compression)])
        return base

    movement_method = "monge-ampere" if method == "monge-ampere" else "mmpde-winslow"
    base = [
        sys.executable, "-m",
        "examples.firedrake._um2n_monge_ampere_cylinder_firedrake" if direct
        else "examples.firedrake.um2n_monge_ampere_cylinder",
        "--output-dir", entry["run_dir"],
        "--mesh", entry["mesh"],
        "--dt", str(args.dt),
        "--steps", str(args.steps),
        "--adapt-every", str(adapt_every),
        "--save-every", str(args.save_every),
        "--monitor-kind", args.ma_monitor_kind,
        "--monitor-frame", "current",
        "--movement-method", movement_method,
        "--coupling-scheme", "um2n-post-solve-ale",
        "--solution-remap", args.ma_solution_remap,
        "--monitor-scale", str(args.ma_monitor_scale),
        "--monitor-continuation-levels", str(args.ma_monitor_continuation_levels),
        "--adaptation-relaxation", str(args.ma_adaptation_relaxation),
        "--max-grid-speed", str(args.ma_max_grid_speed),
        "--quality-guard", args.ma_quality_guard,
        "--quality-guard-tolerance", str(args.ma_quality_guard_tolerance),
        "--rtol", str(args.ma_rtol),
        "--maxiter", str(args.ma_maxiter),
    ]
    if not direct:
        idx = base.index("--mesh")
        base.insert(idx, "--firedrake-run")
        base.insert(idx + 1, args.firedrake_run)
    else:
        exchange_dir = Path(entry["run_dir"]) / "adapter_exchange"
        base.extend(["--adapter-exchange-dir", str(exchange_dir)])
    return base


def _postprocess(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    reference_by_case = {
        entry["case"]: entry
        for entry in manifest["entries"]
        if entry["role"] == "reference"
    }
    points = []
    overhead = []
    for entry in manifest["entries"]:
        if entry["role"] != "sweep":
            continue
        reference = reference_by_case[entry["case"]]
        pvd_path = Path(entry["run_dir"]) / PVD_BY_METHOD[entry["method"]]
        reference_pvd = Path(reference["run_dir"]) / PVD_BY_METHOD[reference["method"]]
        if not pvd_path.exists() or not reference_pvd.exists():
            continue
        error_summary = _error_summary(pvd_path, reference_pvd)
        metrics = json.loads((Path(entry["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        wall_time = _wall_time(Path(entry["run_dir"]))
        point = _point_from_entry(entry, error_summary, metrics, wall_time)
        points.append(point)
        overhead_row = _overhead_summary_from_entry(entry, metrics)
        if overhead_row is not None:
            overhead.append(overhead_row)
    _add_same_h_ratios(points)
    aggregate = _aggregate_points(points)
    _mark_pareto_by_case(points, "mean_vorticity_relative_l2", "pareto_vorticity")
    _mark_pareto_by_case(points, "mean_velocity_relative_l2", "pareto_velocity")
    _mark_pareto(aggregate, "aggregate_vorticity_ratio", "pareto_aggregate_vorticity")
    _mark_pareto(aggregate, "aggregate_velocity_ratio", "pareto_aggregate_velocity")
    return {"manifest": manifest, "points": points, "aggregate": aggregate, "overhead": overhead}


def _error_summary(pvd_path: Path, reference_pvd: Path) -> dict[str, float]:
    series = _load_series(pvd_path)
    reference = _load_series(reference_pvd)
    times = sorted(set(series) & set(reference))
    rows = [_fine_mesh_error_row(time_value, series[time_value], reference[time_value]) for time_value in times]
    return _summary([row for row in rows if row["time"] > 0.0])


def _point_from_entry(
    entry: dict[str, Any],
    error_summary: dict[str, float],
    metrics: dict[str, Any],
    wall_time: float | None,
) -> dict[str, Any]:
    timings = metrics.get("timings_s", {})
    quality = metrics.get("final_mesh_quality") or {}
    adaptations = metrics.get("adaptations", [])
    adapter_warmup = float(timings.get("adapter_warmup", 0.0) or 0.0)
    movement_time = timings.get("movement", timings.get("adapter", 0.0)) or 0.0
    if entry["method"] == "gd-nsrj":
        movement_time = float(movement_time) + adapter_warmup
    total_component = sum(
        float(timings.get(key, 0.0) or 0.0)
        for key in ("solve", "monitor", "movement", "adapter", "adapter_warmup", "projection")
    )
    return {
        "case": entry["case"],
        "h": float(entry["h"]),
        "method": entry["method"],
        "repetition": entry.get("repetition", 0),
        "method_variant": _method_variant(entry),
        "adapter_profile": entry.get("adapter_profile"),
        "monitor_build": entry.get("monitor_build"),
        "monitor_vorticity_hessian_weight": entry.get("monitor_vorticity_hessian_weight"),
        "monitor_vorticity_abs_weight": entry.get("monitor_vorticity_abs_weight"),
        "adapter_movement_weight": entry.get("adapter_movement_weight"),
        "adapter_smoothness_weight": entry.get("adapter_smoothness_weight"),
        "adapter_max_step_edge_stretch": entry.get("adapter_max_step_edge_stretch"),
        "adapter_min_step_edge_compression": entry.get("adapter_min_step_edge_compression"),
        "run_dir": entry["run_dir"],
        "vertices": metrics.get("num_vertices"),
        "cells": metrics.get("num_cells"),
        "wall_time_s": wall_time,
        "solve_s": timings.get("solve", 0.0),
        "monitor_s": timings.get("monitor", 0.0),
        "adapter_warmup_s": adapter_warmup,
        "movement_or_adapter_s": movement_time,
        "total_component_s": total_component,
        "mean_velocity_relative_l2": error_summary.get("mean_velocity_relative_l2"),
        "mean_vorticity_relative_l2": error_summary.get("mean_vorticity_relative_l2"),
        "max_velocity_relative_l2": error_summary.get("max_velocity_relative_l2"),
        "max_vorticity_relative_l2": error_summary.get("max_vorticity_relative_l2"),
        "adaptations": len(adaptations),
        "accepted": sum(1 for adaptation in adaptations if adaptation.get("accepted_adaptation")),
        "orientation_flips": quality.get("orientation_flips"),
        "final_area_ratio": quality.get("area_ratio"),
        "max_grid_speed": metrics.get("max_grid_speed"),
        "max_displacement": metrics.get("max_displacement"),
        "feasible": (quality.get("orientation_flips") or 0) == 0,
    }


def _overhead_summary_from_entry(entry: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any] | None:
    adaptations = metrics.get("adaptations", [])
    if not adaptations:
        return None
    row: dict[str, Any] = {
        "case": entry["case"],
        "h": float(entry["h"]),
        "method": entry["method"],
        "adapter_profile": entry.get("adapter_profile"),
        "monitor_build": entry.get("monitor_build"),
        "monitor_vorticity_hessian_weight": entry.get("monitor_vorticity_hessian_weight"),
        "monitor_vorticity_abs_weight": entry.get("monitor_vorticity_abs_weight"),
        "run_dir": entry["run_dir"],
        "adaptations": len(adaptations),
    }
    series = {
        "monitor": [float(adaptation.get("monitor_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter": [float(adaptation.get("adapter_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_service": [float(adaptation.get("adapter_service_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_validation": [float(adaptation.get("adapter_validation_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_forward_loss": [float(adaptation.get("adapter_forward_loss_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_backward_or_grad": [float(adaptation.get("adapter_backward_or_grad_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_sobolev_solve": [float(adaptation.get("adapter_sobolev_solve_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_step": [float(adaptation.get("adapter_step_s", 0.0) or 0.0) for adaptation in adaptations],
        "adapter_warmup": [float((metrics.get("timings_s", {}) or {}).get("adapter_warmup", 0.0) or 0.0)],
        "total_overhead": [
            float(adaptation.get("monitor_s", 0.0) or 0.0) + float(adaptation.get("adapter_s", 0.0) or 0.0)
            for adaptation in adaptations
        ],
    }
    for key, values in series.items():
        row[f"{key}_mean_s"] = float(np.mean(values))
        row[f"{key}_p50_s"] = _percentile(values, 50.0)
        row[f"{key}_p95_s"] = _percentile(values, 95.0)
    alpha_values = [float(adaptation.get("adapter_alpha", 0.0) or 0.0) for adaptation in adaptations]
    cg_iter_values = [float(adaptation.get("adapter_cg_iters", 0.0) or 0.0) for adaptation in adaptations]
    cg_residual_values = [float(adaptation.get("adapter_cg_residual", 0.0) or 0.0) for adaptation in adaptations]
    row["adapter_alpha_mean"] = float(np.mean(alpha_values)) if alpha_values else 0.0
    row["adapter_alpha_p50"] = _percentile(alpha_values, 50.0)
    row["adapter_cg_iters_mean"] = float(np.mean(cg_iter_values)) if cg_iter_values else 0.0
    row["adapter_cg_residual_mean"] = float(np.mean(cg_residual_values)) if cg_residual_values else 0.0
    return row


def _add_same_h_ratios(points: list[dict[str, Any]]) -> None:
    fixed_by_case_h = {
        (point["case"], point["h"]): point
        for point in points
        if point["method"] == "fixed"
    }
    for point in points:
        fixed = fixed_by_case_h.get((point["case"], point["h"]))
        if fixed is None:
            continue
        point["velocity_ratio_to_fixed_same_h"] = _safe_ratio(
            point.get("mean_velocity_relative_l2"),
            fixed.get("mean_velocity_relative_l2"),
        )
        point["vorticity_ratio_to_fixed_same_h"] = _safe_ratio(
            point.get("mean_vorticity_relative_l2"),
            fixed.get("mean_vorticity_relative_l2"),
        )


def _aggregate_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_h: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for point in points:
        by_method_h.setdefault((point.get("method_variant", point["method"]), point["h"]), []).append(point)
    aggregate = []
    for (method_variant, h_value), group in sorted(by_method_h.items(), key=lambda item: (item[0][0], item[0][1])):
        velocity_ratios = [point.get("velocity_ratio_to_fixed_same_h") for point in group]
        vorticity_ratios = [point.get("vorticity_ratio_to_fixed_same_h") for point in group]
        aggregate.append(
            {
                "method": group[0]["method"],
                "method_variant": method_variant,
                "h": h_value,
                "cases": ",".join(sorted(point["case"] for point in group)),
                "num_cases": len(group),
                "aggregate_velocity_ratio": _geomean(velocity_ratios),
                "aggregate_vorticity_ratio": _geomean(vorticity_ratios),
                "sum_total_component_s": sum(float(point["total_component_s"] or 0.0) for point in group),
                "sum_wall_time_s": _sum_optional(point.get("wall_time_s") for point in group),
                "max_final_area_ratio": max(float(point.get("final_area_ratio") or 0.0) for point in group),
                "feasible": all(point.get("feasible") for point in group),
            }
        )
    return aggregate


def _mark_pareto(points: list[dict[str, Any]], error_key: str, output_key: str) -> None:
    time_key = "sum_total_component_s" if points and "sum_total_component_s" in points[0] else "total_component_s"
    for point in points:
        point[output_key] = False
    candidates = [point for point in points if point.get("feasible") and point.get(error_key) is not None and point.get(time_key) is not None]
    for point in candidates:
        point[output_key] = not any(_dominates(other, point, error_key, time_key) for other in candidates if other is not point)


def _mark_pareto_by_case(points: list[dict[str, Any]], error_key: str, output_key: str) -> None:
    cases = sorted({point["case"] for point in points})
    for point in points:
        point[output_key] = False
    for case in cases:
        _mark_pareto([point for point in points if point["case"] == case], error_key, output_key)


def _dominates(left: dict[str, Any], right: dict[str, Any], error_key: str, time_key: str) -> bool:
    return (
        left[time_key] <= right[time_key]
        and left[error_key] <= right[error_key]
        and (left[time_key] < right[time_key] or left[error_key] < right[error_key])
    )


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_csv = output_dir / "points.csv"
    aggregate_csv = output_dir / "aggregate_points.csv"
    overhead_csv = output_dir / "overhead_summary.csv"
    summary_json = output_dir / "summary.json"
    points_md = output_dir / "points.md"
    aggregate_md = output_dir / "aggregate_points.md"
    _write_csv(points_csv, summary["points"])
    _write_csv(aggregate_csv, summary["aggregate"])
    _write_csv(overhead_csv, summary["overhead"])
    _write_markdown(points_md, summary["points"])
    _write_markdown(aggregate_md, summary["aggregate"])
    _write_json(summary_json, summary)
    case_paths = _write_case_outputs(output_dir, summary["points"])
    aggregate_vorticity_png = output_dir / "aggregate_vorticity_front.png"
    aggregate_velocity_png = output_dir / "aggregate_velocity_front.png"
    _write_front_plot(
        aggregate_vorticity_png,
        summary["aggregate"],
        error_key="aggregate_vorticity_ratio",
        time_key="sum_total_component_s",
        pareto_key="pareto_aggregate_vorticity",
        ylabel="Geometric mean vorticity error ratio to fixed(h)",
    )
    _write_front_plot(
        aggregate_velocity_png,
        summary["aggregate"],
        error_key="aggregate_velocity_ratio",
        time_key="sum_total_component_s",
        pareto_key="pareto_aggregate_velocity",
        ylabel="Geometric mean velocity error ratio to fixed(h)",
    )
    return {
        "points_csv": points_csv,
        "points_markdown": points_md,
        "aggregate_csv": aggregate_csv,
        "overhead_csv": overhead_csv,
        "aggregate_markdown": aggregate_md,
        "summary_json": summary_json,
        "aggregate_vorticity_png": aggregate_vorticity_png,
        "aggregate_velocity_png": aggregate_velocity_png,
        **case_paths,
    }


def _write_case_outputs(output_dir: Path, points: list[dict[str, Any]]) -> dict[str, Path]:
    paths = {}
    for case in sorted({point["case"] for point in points}):
        case_points = [point for point in points if point["case"] == case]
        csv_path = output_dir / f"{case}_points.csv"
        md_path = output_dir / f"{case}_points.md"
        vorticity_png = output_dir / f"{case}_vorticity_front.png"
        velocity_png = output_dir / f"{case}_velocity_front.png"
        _write_csv(csv_path, case_points)
        _write_markdown(md_path, case_points)
        _write_front_plot(
            vorticity_png,
            case_points,
            error_key="mean_vorticity_relative_l2",
            time_key="total_component_s",
            pareto_key="pareto_vorticity",
            ylabel="Mean vorticity relative L2 error",
            xlabel="Total component time (s)",
            title=f"{case}: vorticity Pareto front",
        )
        _write_front_plot(
            velocity_png,
            case_points,
            error_key="mean_velocity_relative_l2",
            time_key="total_component_s",
            pareto_key="pareto_velocity",
            ylabel="Mean velocity relative L2 error",
            xlabel="Total component time (s)",
            title=f"{case}: velocity Pareto front",
        )
        paths[f"{case}_csv"] = csv_path
        paths[f"{case}_markdown"] = md_path
        paths[f"{case}_vorticity_png"] = vorticity_png
        paths[f"{case}_velocity_png"] = velocity_png
    return paths


def _write_front_plot(
    path: Path,
    points: list[dict[str, Any]],
    *,
    error_key: str,
    time_key: str,
    pareto_key: str,
    ylabel: str,
    xlabel: str = "Total component time over all tasks (s)",
    title: str = "Aggregate Pareto front: lower-left is better",
) -> None:
    if not points:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    methods = sorted({point.get("method_variant", point["method"]) for point in points})
    for method in methods:
        method_points = sorted(
            [point for point in points if point.get("method_variant", point["method"]) == method and point.get(error_key) is not None],
            key=lambda p: p[time_key],
        )
        if not method_points:
            continue
        ax.scatter([p[time_key] for p in method_points], [p[error_key] for p in method_points], label=method)
        front = [point for point in method_points if point.get(pareto_key)]
        if front:
            ax.plot([p[time_key] for p in front], [p[error_key] for p in front], linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wall_time(run_dir: Path) -> float | None:
    path = run_dir / "wall_time.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["wall_time_s"])


def _safe_ratio(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    baseline = float(baseline)
    if baseline <= 0.0:
        return None
    return float(value) / baseline


def _method_variant(entry: dict[str, Any]) -> str:
    if entry["method"] != "gd-nsrj":
        return entry["method"]
    parts = [entry["method"]]
    if entry.get("monitor_build"):
        parts.append(str(entry["monitor_build"]))
    if entry.get("adapter_profile"):
        parts.append(str(entry["adapter_profile"]))
    return "+".join(parts)


def _geomean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and float(value) > 0.0]
    if not clean:
        return None
    return math.exp(sum(math.log(value) for value in clean) / len(clean))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _sum_optional(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean)


def _h_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p").rstrip("0").rstrip("p")


if __name__ == "__main__":
    main()
