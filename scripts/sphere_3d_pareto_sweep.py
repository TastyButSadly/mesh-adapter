"""3D sphere flow Pareto sweep: fixed vs sobolev-transport across mesh resolutions.

Mirrors scripts/diploma_pareto_sweep.py but for 3D sphere geometry.
"""
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

from examples.firedrake.postprocess_multi_cylinder_benchmark import _load_series, _summary

METHODS = ("fixed", "sobolev-transport")
PVD_FILE = "sphere_flow_diff_adapter.pvd"
MESHES_DIR = Path("outputs/sphere_3d_pareto/meshes")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D sphere Pareto sweep: fixed vs sobolev-transport.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sphere_3d_pareto"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--h-values", nargs="+", type=float, default=[0.200, 0.160, 0.130, 0.100, 0.080])
    parser.add_argument("--reference-h", type=float, default=0.055)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--adapter-steps", type=int, default=10)
    parser.add_argument("--adapter-lr", type=float, default=4e-4)
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
        print(f"pareto plot: {paths['pareto_png']}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    h_values = list(args.h_values) + [args.reference_h]
    entries = []
    for h in h_values:
        h_str = f"{h:.3f}".replace(".", "p")
        mesh_path = MESHES_DIR / f"sphere_h{h_str}.msh"
        is_reference = abs(h - args.reference_h) < 1e-9
        for method in args.methods:
            if is_reference and method != "fixed":
                continue
            run_dir = str(args.output_root / f"h{h:.3f}" / method)
            entry = {
                "h": h,
                "role": "reference" if is_reference else "sweep",
                "method": method,
                "mesh": str(mesh_path),
                "run_dir": run_dir,
                "adapter_profile": method if method != "fixed" else None,
            }
            entries.append(entry)
    return {
        "dt": args.dt,
        "steps": args.steps,
        "adapt_every": args.adapt_every,
        "save_every": args.save_every,
        "reference_h": args.reference_h,
        "h_values": args.h_values,
        "methods": list(args.methods),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Mesh preparation
# ---------------------------------------------------------------------------
def _prepare_meshes(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    from examples.firedrake.sphere_flow_mesh import write_sphere_channel_mesh, SphereMeshResolution

    all_h = sorted(set(e["h"] for e in manifest["entries"]))
    configs = {
        0.200: ("h0p200", 0.200, 0.070, 0.14, 0.3),
        0.160: ("h0p160", 0.160, 0.055, 0.10, 0.3),
        0.130: ("h0p130", 0.130, 0.045, 0.08, 0.3),
        0.100: ("h0p100", 0.100, 0.035, 0.07, 0.3),
        0.080: ("h0p080", 0.080, 0.028, 0.055, 0.3),
        0.055: ("h0p055", 0.055, 0.020, 0.040, 0.3),
    }
    for h in all_h:
        if h not in configs:
            print(f"WARNING: no mesh config for h={h}, skipping")
            continue
        name, ms, ss, ws, wd = configs[h]
        mesh_path = MESHES_DIR / (f"sphere_{name}" + ".msh")
        if mesh_path.exists() and not args.force:
            print(f"reuse mesh h={h}: {mesh_path}")
            continue
        res = SphereMeshResolution(name, mesh_size=ms, sphere_size=ss, wake_size=ws, wake_distance=wd)
        stats = write_sphere_channel_mesh(mesh_path, res)
        print(f"mesh h={h}: {stats['points']} pts, {stats['tetrahedra']} cells")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _run_missing(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    for entry in manifest["entries"]:
        run_dir = Path(entry["run_dir"])
        pvd_path = run_dir / PVD_FILE
        metrics_path = run_dir / "metrics.json"
        if pvd_path.exists() and metrics_path.exists() and not args.force:
            print(f"reuse {entry['h']} {entry['method']}: {run_dir}")
            continue
        command = _command_for_entry(args, entry)
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - start
        _write_json(run_dir / "wall_time.json", {"wall_time_s": elapsed})


def _command_for_entry(args: argparse.Namespace, entry: dict[str, Any]) -> list[str]:
    method = entry["method"]
    adapt_every = args.steps + 1 if method == "fixed" else args.adapt_every
    command = [
        sys.executable,
        "-m",
        "examples.firedrake.sphere_flow_diff_adapter",
        "--output-dir",
        entry["run_dir"],
        "--mesh",
        entry["mesh"],
        "--dt",
        str(args.dt),
        "--steps",
        str(args.steps),
        "--adapt-every",
        str(adapt_every),
        "--save-every",
        str(args.save_every),
        "--adapter-profile",
        entry.get("adapter_profile") or "regularized",
        "--adapter-steps",
        "1" if method == "sobolev-transport" else "10",
        "--adapter-lr",
        str(args.adapter_lr),
    ]
    return command


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------
def _postprocess(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    reference_entry = None
    for entry in manifest["entries"]:
        if entry["role"] == "reference":
            reference_entry = entry
            break
    if reference_entry is None:
        raise RuntimeError("No reference entry in manifest")

    reference_pvd = Path(reference_entry["run_dir"]) / PVD_FILE
    if not reference_pvd.exists():
        raise RuntimeError(f"Reference PVD not found: {reference_pvd}")

    points = []
    for entry in manifest["entries"]:
        if entry["role"] != "sweep":
            continue
        pvd_path = Path(entry["run_dir"]) / PVD_FILE
        if not pvd_path.exists():
            print(f"SKIP {entry['h']} {entry['method']}: no PVD")
            continue
        coarse_mesh_path = Path(entry["mesh"])
        fine_mesh_path = Path(reference_entry["mesh"])
        error_summary = _error_summary(pvd_path, reference_pvd, coarse_mesh_path, fine_mesh_path)
        metrics = json.loads((Path(entry["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        wall_time = _wall_time(Path(entry["run_dir"]))
        point = _point_from_entry(entry, error_summary, metrics, wall_time)
        points.append(point)

    _mark_pareto(points, "mean_vorticity_relative_l2", "pareto_vorticity")
    _mark_pareto(points, "mean_velocity_relative_l2", "pareto_velocity")
    return {"manifest": manifest, "points": points}


def _error_summary(pvd_path: Path, reference_pvd: Path, coarse_mesh: Path, fine_mesh: Path) -> dict[str, float]:
    """Compute L2 error using scipy LinearNDInterpolator (3D Delaunay).

    This is the 3D equivalent of matplotlib.tri.LinearTriInterpolator used in 2D.
    Exact P1 interpolation on the Delaunay tetrahedralization of the source mesh.
    """
    from examples.firedrake.postprocess_multi_cylinder_benchmark import _load_series

    # Check for cached result
    cache_path = pvd_path.parent / "l2_error.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    series = _load_series(pvd_path)
    reference = _load_series(reference_pvd)
    times = sorted(set(series) & set(reference))
    rows = [_error_row_3d(t, series[t], reference[t]) for t in times]
    result = _summary([row for row in rows if row["time"] > 0.0])

    # Cache
    cache_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _error_row_3d(time_value: float, current_vtu: Path, reference_vtu: Path) -> dict[str, float]:
    ref_points, ref_cells, ref_data = _read_vtu_3d(reference_vtu)
    cur_points, _cur_cells, cur_data = _read_vtu_3d(current_vtu)
    row: dict[str, float] = {"time": float(time_value)}

    # Subsample reference points for speed (max 4000 points)
    max_ref = 4000
    if len(ref_points) > max_ref:
        step = max(1, len(ref_points) // max_ref)
        idx = np.arange(0, len(ref_points), step)[:max_ref]
        ref_points_sub = ref_points[idx]
        ref_data_sub = {k: v[idx] if v.ndim == 1 else v[idx] for k, v in ref_data.items()}
    else:
        ref_points_sub = ref_points
        ref_data_sub = ref_data
        idx = np.arange(len(ref_points))

    # Compute vertex weights from full reference mesh
    vertex_weights = np.zeros(len(ref_points_sub))
    volumes = _tetra_volumes(ref_points_sub, _remap_cells(ref_cells, idx))

    for i in range(volumes.shape[1]) if volumes.ndim == 2 else range(1):
        pass
    # Simple uniform weights for subsampled points
    vertex_weights = np.ones(len(ref_points_sub))

    for name, out_prefix in [("velocity", "velocity"), ("vorticity_magnitude", "vorticity")]:
        if name not in ref_data_sub or name not in cur_data:
            continue
        ref_vals = ref_data_sub[name]
        cur_vals = cur_data[name]

        interpolated = _nn_interpolate(cur_points, cur_vals, ref_points_sub)

        valid = np.isfinite(interpolated).all(axis=-1) if interpolated.ndim == 2 else np.isfinite(interpolated)
        if not valid.any():
            continue

        diff = interpolated[valid] - ref_vals[valid]
        w = vertex_weights[valid]

        if diff.ndim == 2:
            diff_sq = np.sum(diff ** 2, axis=1)
            ref_sq = np.sum(ref_vals[valid] ** 2, axis=1)
        else:
            diff_sq = diff ** 2
            ref_sq = ref_vals[valid] ** 2

        l2_sq = float(np.average(diff_sq, weights=w))
        ref_l2_sq = float(np.average(ref_sq, weights=w))

        row[f"{out_prefix}_l2"] = float(np.sqrt(max(l2_sq, 0.0)))
        row[f"{out_prefix}_relative_l2"] = float(np.sqrt(max(l2_sq, 0.0)) / max(np.sqrt(max(ref_l2_sq, 0.0)), 1e-14))
        row[f"{out_prefix}_linf"] = float(np.nanmax(np.abs(diff)))
    return row


def _remap_cells(cells: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Remap cell connectivity to use only nodes in idx subset."""
    idx_set = set(idx.tolist())
    mask = np.array([all(n in idx_set for n in row) for row in cells.tolist()])
    if not mask.any():
        return np.empty((0, cells.shape[1]), dtype=cells.dtype)
    old_to_new = {old: new for new, old in enumerate(idx)}
    remapped = cells[mask].copy()
    for i in range(remapped.shape[1]):
        col = remapped[:, i]
        for j in range(len(col)):
            remapped[j, i] = old_to_new.get(col[j], -1)
    return remapped


def _nn_interpolate(src_points: np.ndarray, src_values: np.ndarray, tgt_points: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation on 3D Delaunay triangulation.

    The 3D equivalent of matplotlib.tri.LinearTriInterpolator used in 2D.
    Uses scipy.interpolate.LinearNDInterpolator on the Delaunay tetrahedralization.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    tri = Delaunay(src_points)
    if src_values.ndim == 1:
        interp = LinearNDInterpolator(tri, src_values, fill_value=np.nan)
        return np.asarray(interp(tgt_points))
    # Vector field: interpolate each component separately
    cols = []
    for c in range(src_values.shape[1]):
        interp = LinearNDInterpolator(tri, src_values[:, c], fill_value=np.nan)
        cols.append(np.asarray(interp(tgt_points)))
    return np.column_stack(cols)


def _read_vtu_3d(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read VTU file, handling Firedrake P2 tetrahedra (10-node, type 71)."""
    from examples.firedrake.postprocess_multi_cylinder_benchmark import _read_vtu
    points, cells, point_data = _read_vtu(path)
    # Firedrake P2 tets have 10 connectivity entries per cell (4 corner + 6 edge midpoint)
    # Extract only corner nodes (indices 0,1,2,3 of each block of 10)
    if cells.ndim == 2 and cells.shape[1] == 10:
        cells = cells[:, :4].copy()
    return points, cells, point_data


def _tetra_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if cells.shape[1] == 4:
        p0 = points[cells[:, 0]]
        p1 = points[cells[:, 1]]
        p2 = points[cells[:, 2]]
        p3 = points[cells[:, 3]]
        return np.abs(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0
    elif cells.shape[1] == 3:
        a = points[cells[:, 0], :2]
        b = points[cells[:, 1], :2]
        c = points[cells[:, 2], :2]
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return np.ones(len(cells)) / len(cells)


def _point_from_entry(
    entry: dict[str, Any],
    error_summary: dict[str, float],
    metrics: dict[str, Any],
    wall_time: float | None,
) -> dict[str, Any]:
    timings = metrics.get("timings_s", {})
    quality = metrics.get("final_mesh_quality") or {}
    adaptations = metrics.get("adaptations", [])
    total_component = sum(
        float(timings.get(k, 0.0) or 0.0)
        for k in ("solve", "monitor", "movement", "adapter", "adapter_warmup", "projection")
    )
    movement_time = timings.get("movement", timings.get("adapter", 0.0)) or 0.0
    return {
        "h": float(entry["h"]),
        "method": entry["method"],
        "adapter_profile": entry.get("adapter_profile"),
        "run_dir": entry["run_dir"],
        "vertices": metrics.get("num_vertices"),
        "cells": metrics.get("num_cells"),
        "wall_time_s": wall_time,
        "solve_s": timings.get("solve", 0.0),
        "monitor_s": timings.get("monitor", 0.0),
        "movement_or_adapter_s": float(movement_time),
        "total_component_s": total_component,
        "mean_velocity_relative_l2": error_summary.get("mean_velocity_relative_l2"),
        "mean_vorticity_relative_l2": error_summary.get("mean_vorticity_magnitude_relative_l2", error_summary.get("mean_vorticity_relative_l2")),
        "max_velocity_relative_l2": error_summary.get("max_velocity_relative_l2"),
        "max_vorticity_relative_l2": error_summary.get("max_vorticity_magnitude_relative_l2", error_summary.get("max_vorticity_relative_l2")),
        "adaptations": len(adaptations),
        "orientation_flips": quality.get("orientation_flips"),
        "volume_ratio": quality.get("volume_ratio"),
        "max_displacement": metrics.get("max_displacement"),
        "feasible": (quality.get("orientation_flips") or 0) == 0,
    }


# ---------------------------------------------------------------------------
# Pareto marking
# ---------------------------------------------------------------------------
def _mark_pareto(points: list[dict[str, Any]], error_key: str, output_key: str) -> None:
    feasible = [p for p in points if p.get("feasible", True) and p.get(error_key) is not None]
    for p in feasible:
        p[output_key] = not any(
            o["total_component_s"] <= p["total_component_s"]
            and o[error_key] <= p[error_key]
            and (o["total_component_s"] < p["total_component_s"] or o[error_key] < p[error_key])
            for o in feasible if o is not p
        )


# ---------------------------------------------------------------------------
# Write summary
# ---------------------------------------------------------------------------
def _write_summary(report_dir: Path, summary: dict[str, Any]) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    points = summary["points"]

    # points.csv
    columns = [
        "h", "method", "adapter_profile", "run_dir", "vertices", "cells",
        "wall_time_s", "solve_s", "monitor_s", "movement_or_adapter_s", "total_component_s",
        "mean_velocity_relative_l2", "mean_vorticity_relative_l2",
        "max_velocity_relative_l2", "max_vorticity_relative_l2",
        "adaptations", "orientation_flips", "volume_ratio", "max_displacement",
        "feasible", "pareto_vorticity", "pareto_velocity",
    ]
    points_csv = report_dir / "points.csv"
    with points_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    print(f"points CSV: {points_csv}")

    # Print table
    print(f"\n{'h':>8} {'method':<18} {'verts':>6} {'cells':>7} {'wall_s':>8} {'adapter_s':>10} "
          f"{'vel_L2':>10} {'vor_L2':>10} {'flips':>5} {'feasible':>8}")
    print("-" * 105)
    for p in sorted(points, key=lambda x: (x["h"], x["method"])):
        print(f"{p['h']:>8.3f} {p['method']:<18} {p['vertices'] or 0:>6} {p['cells'] or 0:>7} "
              f"{p['wall_time_s'] or 0:>8.1f} {p['movement_or_adapter_s']:>10.3f} "
              f"{p['mean_velocity_relative_l2'] or 0:>10.6f} {p['mean_vorticity_relative_l2'] or 0:>10.6f} "
              f"{p['orientation_flips'] or 0:>5} {str(p['feasible']):>8}")

    # Pareto plot
    pareto_png = _write_pareto_plot(report_dir, points)

    return {"points_csv": points_csv, "pareto_png": pareto_png}


def _write_pareto_plot(report_dir: Path, points: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    method_styles = {
        "fixed": {"marker": "o", "color": "#2196F3", "label": "Fixed"},
        "sobolev-transport": {"marker": "^", "color": "#E91E63", "label": "Sobolev transport"},
    }

    for ax, error_key, title in [
        (ax1, "mean_velocity_relative_l2", "Velocity L2 error"),
        (ax2, "mean_vorticity_relative_l2", "Vorticity L2 error"),
    ]:
        for method, style in method_styles.items():
            subset = [p for p in points if p["method"] == method and p.get(error_key) is not None]
            if not subset:
                continue
            xs = [p["total_component_s"] for p in subset]
            ys = [p[error_key] for p in subset]
            ax.scatter(xs, ys, s=60, zorder=3, **style)

        # Pareto front
        pareto = sorted(
            [p for p in points if p.get(f"pareto_{error_key.split('_')[1]}", False) and p.get(error_key) is not None],
            key=lambda p: p["total_component_s"],
        )
        if pareto:
            px = [p["total_component_s"] for p in pareto]
            py = [p[error_key] for p in pareto]
            ax.plot(px, py, "k--", linewidth=1.5, alpha=0.7, label="Pareto front")

        ax.set_xlabel("Total component time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("3D Sphere Flow Pareto Front: Fixed vs Sobolev Transport", fontsize=14)
    fig.tight_layout()
    png_path = report_dir / "sphere_3d_pareto.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"pareto plot: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wall_time(run_dir: Path) -> float | None:
    path = run_dir / "wall_time.json"
    if path.exists():
        return json.loads(path.read_text()).get("wall_time_s")
    return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
