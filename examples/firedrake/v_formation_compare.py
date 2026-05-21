from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from examples.firedrake.multi_cylinder_meshes import (
    CASES,
    CHANNEL_HEIGHT,
    CHANNEL_LENGTH,
    CYLINDER_MARKER,
    CYLINDER_RADIUS,
    INLET_MARKER,
    OUTLET_MARKER,
    RESOLUTIONS,
    WALL_MARKER,
    _classify_boundary_curves,
    _mesh_stats,
    _write_preview,
)
from examples.firedrake.postprocess_multi_cylinder_benchmark import (
    _load_series,
    _read_vtu,
    _summary,
)

PVD_NAME = "um2n_diff_adapter_cylinder.pvd"
UM2N_VORTICITY_PLOT_LIMIT = 100.0
UM2N_ERROR_PLOT_LIMIT = 100.0
RESOLUTION_BY_NAME = {resolution.name: resolution for resolution in RESOLUTIONS}
PAPER_COLUMNS = (
    "mesh",
    "mode",
    "wall_s",
    "wall_overhead_pct",
    "mean_rel_l2_vorticity",
    "mean_rel_l2_vorticity_improvement_pct",
    "mean_rel_l2_velocity",
    "mean_rel_l2_velocity_improvement_pct",
    "max_rel_l2_vorticity",
    "max_rel_l2_vorticity_improvement_pct",
    "max_rel_l2_velocity",
    "max_rel_l2_velocity_improvement_pct",
)


def main() -> None:
    args = _parse_args()
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    meshes = _mesh_paths_for_postprocess(args, run_dir) if args.postprocess_only or args.dry_run else _prepare_meshes(args, run_dir)
    manifest = _run_cases(args, run_dir, meshes)
    if args.dry_run:
        print(f"dry-run manifest: {run_dir / 'manifest.json'}")
        return

    comparison = _postprocess(args, run_dir, manifest)
    print(f"run dir: {run_dir}")
    print(f"summary table: {comparison['csv']}")
    print(f"paper table: {comparison['paper_png']}")
    print(f"summary plot: {comparison['plot']}")
    for label, path in comparison["diagnostics"].items():
        print(f"diagnostic {label}: {path}")
    print(f"ParaView series:")
    for label, pvd_path in comparison["pvd"].items():
        print(f"  {label}: {pvd_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or postprocess fine/coarse/medium adaptive multi-cylinder comparisons. "
            "Generated meshes are uniform-size channel meshes without deliberate "
            "cylinder/wake refinement."
        )
    )
    parser.add_argument("--case", choices=tuple(CASES), default="v_formation")
    parser.add_argument("--resolutions", nargs="+", choices=("coarse", "medium"), default=("coarse", "medium"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multi_cylinder_compare_runs"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact run directory. Overrides --output-root/--run-id.")
    parser.add_argument("--run-id", default=None, help="Run id below --output-root. Defaults to timestamp.")

    parser.add_argument("--mesh-dir", type=Path, default=Path("outputs/multi_cylinder_mesh_family"), help="Existing mesh family root.")
    parser.add_argument("--coarse-mesh", type=Path, default=None, help="Existing coarse .msh to copy into this run.")
    parser.add_argument("--medium-mesh", type=Path, default=None, help="Existing medium .msh to copy into this run.")
    parser.add_argument("--fine-mesh", type=Path, default=None, help="Existing fine .msh to copy into this run.")
    parser.add_argument("--coarse-mesh-size", type=float, default=0.05, help="Uniform coarse mesh size when --coarse-mesh is omitted.")
    parser.add_argument("--medium-mesh-size", type=float, default=0.03, help="Uniform medium mesh size when --medium-mesh is omitted.")
    parser.add_argument("--fine-mesh-size", type=float, default=0.016, help="Uniform fine mesh size when --fine-mesh is omitted.")
    parser.add_argument("--no-mesh-preview", action="store_true")

    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--firedrake-run", default="firedrake-run")

    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--monitor-kind", choices=("velocity-gradient", "wake-vorticity"), default="velocity-gradient")
    parser.add_argument("--monitor-frame", choices=("reference", "current"), default="current")
    parser.add_argument(
        "--monitor-build",
        choices=("firedrake-smoothed", "raw-gradient", "graph-gradient", "raw-vorticity", "composite-grad-vort"),
        default="firedrake-smoothed",
    )
    parser.add_argument("--monitor-scale", type=float, default=3.0)
    parser.add_argument("--monitor-graph-smoothing-steps", type=int, default=4)
    parser.add_argument("--monitor-graph-smoothing-weight", type=float, default=0.5)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only"), default="regularized")
    parser.add_argument("--adapter-steps", type=int, default=6)
    parser.add_argument("--adapter-lr", type=float, default=8.0e-4)
    parser.add_argument("--adapter-preset", choices=("custom", "accurate", "fast", "faster"), default="custom")
    parser.add_argument("--adapter-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--adapter-transport", choices=("npz", "tcp"), default="tcp")
    parser.add_argument("--adapter-torch-threads", type=int, default=1)
    parser.add_argument("--max-grid-speed", type=float, default=1.0)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)

    parser.add_argument("--sample-nx", type=int, default=420)
    parser.add_argument("--sample-ny", type=int, default=96)
    parser.add_argument("--cross-sections", nargs="+", type=float, default=(0.5, 1.0, 1.5, 2.0))

    parser.add_argument("--only-adaptive", action="store_true", help="Skip fine/fixed and coarse/fixed runs.")
    parser.add_argument("--postprocess-only", action="store_true", help="Do not run solvers; only rebuild tables/plot from existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest but do not run solvers or postprocess.")
    parser.add_argument("--force", action="store_true", help="Rerun every requested solver output directory.")
    parser.add_argument("--force-fixed", action="store_true", help="Rerun fixed fine/coarse outputs.")
    parser.add_argument("--force-adaptive", action="store_true", help="Rerun coarse adaptive output.")

    parser.add_argument(
        "--fine-fixed-dir",
        type=Path,
        default=None,
        help="Existing fine/fixed result directory for postprocessing and reference reuse.",
    )
    parser.add_argument("--coarse-fixed-dir", type=Path, default=None, help="Existing coarse/fixed result directory for postprocessing.")
    parser.add_argument("--medium-fixed-dir", type=Path, default=None, help="Existing medium/fixed result directory for postprocessing.")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / run_id


def _mesh_paths_for_postprocess(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    meshes = {"fine": args.fine_mesh or run_dir / "meshes" / f"{args.case}_fine.msh"}
    for resolution in args.resolutions:
        meshes[resolution] = _explicit_mesh_arg(args, resolution) or run_dir / "meshes" / f"{args.case}_{resolution}.msh"
    return meshes


def _prepare_meshes(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    mesh_dir = run_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    preview = not args.no_mesh_preview

    meshes = {"fine": _prepare_mesh(args, mesh_dir, "fine", preview)}
    for resolution in args.resolutions:
        meshes[resolution] = _prepare_mesh(args, mesh_dir, resolution, preview)
    return meshes


def _prepare_mesh(args: argparse.Namespace, mesh_dir: Path, resolution: str, preview: bool) -> Path:
    target = mesh_dir / f"{args.case}_{resolution}.msh"
    explicit = _explicit_mesh_arg(args, resolution)
    family_mesh = args.mesh_dir / args.case / f"{args.case}_{resolution}.msh"
    if explicit is not None:
        shutil.copy2(explicit, target)
    elif family_mesh.exists() and (args.force or not target.exists()):
        shutil.copy2(family_mesh, target)
    elif not target.exists() or args.force:
        write_uniform_multi_cylinder_mesh(
            target,
            CASES[args.case],
            _mesh_size_for_args(args, resolution),
            preview_path=target.with_suffix(".png") if preview else None,
        )
    return target


def _explicit_mesh_arg(args: argparse.Namespace, resolution: str) -> Path | None:
    if resolution == "coarse":
        return args.coarse_mesh
    if resolution == "medium":
        return args.medium_mesh
    if resolution == "fine":
        return args.fine_mesh
    raise ValueError(f"Unknown mesh resolution: {resolution}")


def _mesh_size_for_args(args: argparse.Namespace, resolution: str) -> float:
    if resolution == "coarse":
        return args.coarse_mesh_size
    if resolution == "medium":
        return args.medium_mesh_size
    if resolution == "fine":
        return args.fine_mesh_size
    raise ValueError(f"Unknown mesh resolution: {resolution}")


def write_uniform_multi_cylinder_mesh(
    path: Path,
    centers: tuple[tuple[float, float], ...],
    mesh_size: float,
    *,
    preview_path: Path | None = None,
) -> dict[str, int]:
    import gmsh

    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.add(path.stem)

        occ = gmsh.model.occ
        rectangle = occ.addRectangle(0.0, 0.0, 0.0, CHANNEL_LENGTH, CHANNEL_HEIGHT)
        cylinders = [occ.addDisk(x, y, 0.0, CYLINDER_RADIUS, CYLINDER_RADIUS) for x, y in centers]
        occ.cut([(2, rectangle)], [(2, tag) for tag in cylinders], removeObject=True, removeTool=True)
        occ.synchronize()

        surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        boundary_groups = _classify_boundary_curves(gmsh)
        for marker, name in (
            (WALL_MARKER, "walls"),
            (INLET_MARKER, "inlet"),
            (OUTLET_MARKER, "outlet"),
            (CYLINDER_MARKER, "cylinders"),
        ):
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name)

        field = gmsh.model.mesh.field
        field.add("Constant", 1)
        field.setNumber(1, "VIn", mesh_size)
        field.setAsBackgroundMesh(1)

        gmsh.model.mesh.generate(2)
        stats = _mesh_stats(gmsh)
        gmsh.write(str(path))
        if preview_path is not None:
            _write_preview(gmsh, centers, preview_path)
        return stats
    finally:
        gmsh.finalize()


def _run_cases(args: argparse.Namespace, run_dir: Path, meshes: dict[str, Path]) -> list[dict[str, object]]:
    previous_manifest = _previous_manifest_by_label(run_dir / "manifest.json")
    manifest = [
        _run_record(
            args,
            label="fine/fixed",
            mesh=meshes["fine"],
            output_dir=run_dir / "fine" / "fixed",
            adapt_every=args.steps + 1,
            force=args.force or args.force_fixed,
            skip=args.only_adaptive or args.postprocess_only or args.fine_fixed_dir is not None,
            previous=previous_manifest.get("fine/fixed"),
        ),
    ]
    for resolution in args.resolutions:
        manifest.append(
            _run_record(
                args,
                label=f"{resolution}/fixed",
                mesh=meshes[resolution],
                output_dir=run_dir / resolution / "fixed",
                adapt_every=args.steps + 1,
                force=args.force or args.force_fixed,
                skip=args.only_adaptive or args.postprocess_only,
                previous=previous_manifest.get(f"{resolution}/fixed"),
            )
        )
        manifest.append(
            _run_record(
                args,
                label=f"{resolution}/adapted",
                mesh=meshes[resolution],
                output_dir=run_dir / resolution / "adapted",
                adapt_every=args.adapt_every,
                force=args.force or args.force_adaptive,
                skip=args.postprocess_only,
                previous=previous_manifest.get(f"{resolution}/adapted"),
            )
        )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _run_record(
    args: argparse.Namespace,
    *,
    label: str,
    mesh: Path,
    output_dir: Path,
    adapt_every: int,
    force: bool,
    skip: bool,
    previous: dict[str, object] | None,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "examples.firedrake.um2n_diff_adapter_cylinder",
        "--mesh",
        mesh.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--firedrake-run",
        args.firedrake_run,
        "--dt",
        str(args.dt),
        "--steps",
        str(args.steps),
        "--save-every",
        str(args.save_every),
        "--adapt-every",
        str(adapt_every),
        "--monitor-kind",
        args.monitor_kind,
        "--monitor-frame",
        args.monitor_frame,
        "--monitor-build",
        args.monitor_build,
        "--monitor-scale",
        str(args.monitor_scale),
        "--monitor-graph-smoothing-steps",
        str(args.monitor_graph_smoothing_steps),
        "--monitor-graph-smoothing-weight",
        str(args.monitor_graph_smoothing_weight),
        "--adapter-profile",
        args.adapter_profile,
        "--adapter-steps",
        str(args.adapter_steps),
        "--adapter-lr",
        str(args.adapter_lr),
        "--adapter-preset",
        args.adapter_preset,
        "--adapter-dtype",
        args.adapter_dtype,
        "--adapter-transport",
        args.adapter_transport,
        "--adapter-torch-threads",
        str(args.adapter_torch_threads),
        "--max-grid-speed",
        str(args.max_grid_speed),
        "--adaptation-relaxation",
        str(args.adaptation_relaxation),
    ]
    record: dict[str, object] = {
        "label": label,
        "mesh": str(mesh),
        "output_dir": str(output_dir),
        "pvd": str(output_dir / PVD_NAME),
        "command": command,
    }
    ready = (output_dir / PVD_NAME).exists() and (output_dir / "metrics.json").exists()
    if skip:
        record["status"] = "skipped"
        if previous is not None and "wall_time_s" in previous:
            record["wall_time_s"] = previous["wall_time_s"]
        return record
    if ready and not force:
        record["status"] = "reused"
        if previous is not None and "wall_time_s" in previous:
            record["wall_time_s"] = previous["wall_time_s"]
        return record
    if args.dry_run:
        record["status"] = "planned"
        return record

    start = time.perf_counter()
    print(f"RUN {label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    record["wall_time_s"] = time.perf_counter() - start
    record["returncode"] = completed.returncode
    record["status"] = "done" if completed.returncode == 0 else "failed"
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return record


def _previous_manifest_by_label(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {record["label"]: record for record in records if "label" in record}


def _postprocess(args: argparse.Namespace, run_dir: Path, manifest: list[dict[str, object]]) -> dict[str, object]:
    fine_fixed_dir = args.fine_fixed_dir or run_dir / "fine" / "fixed"
    comparison_dir = run_dir / "comparison"
    figures_dir = run_dir / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_series(fine_fixed_dir / PVD_NAME)

    comparisons = []
    pvd_paths = {"fine/fixed": fine_fixed_dir / PVD_NAME}
    diagnostic_paths = {}
    for resolution in args.resolutions:
        for mode in ("fixed", "adapted"):
            label = f"{resolution}/{mode}"
            result_dir = _result_dir(args, run_dir, resolution, mode)
            pvd_path = result_dir / PVD_NAME
            pvd_paths[label] = pvd_path
            series = _load_series(pvd_path)
            common_times = sorted(set(reference) & set(series))
            rows = [_fine_mesh_error_row(time_value, series[time_value], reference[time_value]) for time_value in common_times]
            comparisons.append(
                {
                    "label": label,
                    "pvd": str(pvd_path),
                    "num_times": len(rows),
                    "summary": _summary(rows),
                    "summary_excluding_t0": _summary([row for row in rows if row["time"] > 0.0]),
                    "times": rows,
                }
            )
            if mode == "adapted":
                diagnostic_path = figures_dir / f"{args.case}_{resolution}_adapted_diagnostics.png"
                _write_adapted_diagnostics(diagnostic_path, pvd_path, case_name=args.case)
                diagnostic_paths[label] = diagnostic_path

    summary = {
        "case": args.case,
        "run_dir": str(run_dir),
        "reference": str(fine_fixed_dir / PVD_NAME),
        "error_method": "coarse/adapted fields linearly interpolated to fine-reference VTU nodes; triangle-weighted P1 L2 on fine-reference mesh",
        "manifest": manifest,
        "comparisons": comparisons,
    }
    summary_path = comparison_dir / "error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table_rows = _table_rows(args, run_dir, manifest, fine_fixed_dir, comparisons)
    csv_path = comparison_dir / "comparison_table.csv"
    md_path = comparison_dir / "comparison_table.md"
    _write_csv(csv_path, table_rows)
    _write_markdown_table(md_path, table_rows)
    paper_rows = _paper_table_rows(table_rows)
    paper_csv_path = comparison_dir / "paper_table.csv"
    paper_md_path = comparison_dir / "paper_table.md"
    paper_png_path = comparison_dir / "paper_table.png"
    _write_csv(paper_csv_path, paper_rows, fieldnames=list(PAPER_COLUMNS))
    _write_markdown_table(paper_md_path, paper_rows, headers=list(PAPER_COLUMNS))
    _write_paper_table_png(paper_png_path, paper_rows, table_rows)

    plot_path = comparison_dir / "vorticity_comparison.png"
    first_resolution = args.resolutions[0]
    _write_comparison_plot(
        plot_path,
        fine_fixed_dir / PVD_NAME,
        _result_dir(args, run_dir, first_resolution, "fixed") / PVD_NAME,
        _result_dir(args, run_dir, first_resolution, "adapted") / PVD_NAME,
        cross_sections=tuple(args.cross_sections),
        case_name=args.case,
    )
    _write_paraview_readme(comparison_dir / "PARAVIEW_SERIES.md", pvd_paths)
    return {
        "summary": summary_path,
        "csv": csv_path,
        "markdown": md_path,
        "paper_csv": paper_csv_path,
        "paper_markdown": paper_md_path,
        "paper_png": paper_png_path,
        "plot": plot_path,
        "diagnostics": diagnostic_paths,
        "pvd": pvd_paths,
    }


def _result_dir(args: argparse.Namespace, run_dir: Path, resolution: str, mode: str) -> Path:
    if mode == "fixed":
        override = args.coarse_fixed_dir if resolution == "coarse" else args.medium_fixed_dir if resolution == "medium" else None
        if override is not None:
            return override
    return run_dir / resolution / mode


def _table_rows(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: list[dict[str, object]],
    fine_fixed_dir: Path,
    comparisons: list[dict[str, object]],
) -> list[dict[str, object]]:
    manifest_by_output = {record["output_dir"]: record for record in manifest}
    comparison_by_label = {comparison["label"]: comparison for comparison in comparisons}
    rows = []
    result_dirs = [("fine/fixed", fine_fixed_dir)]
    for resolution in args.resolutions:
        result_dirs.append((f"{resolution}/fixed", _result_dir(args, run_dir, resolution, "fixed")))
        result_dirs.append((f"{resolution}/adapted", _result_dir(args, run_dir, resolution, "adapted")))
    for label, result_dir in result_dirs:
        metrics = _read_metrics(result_dir)
        record = _manifest_record_for_result_dir(manifest_by_output, result_dir)
        comparison = comparison_by_label.get(label)
        summary = comparison["summary_excluding_t0"] if comparison is not None else {}
        adaptations = metrics.get("adaptations", [])
        timings = metrics.get("timings_s", {})
        rows.append(
            {
                "run": label,
                "vertices": metrics.get("num_vertices"),
                "cells": metrics.get("num_cells"),
                "steps": metrics.get("steps"),
                "adapt_every": metrics.get("adapt_every"),
                "adaptations": len(adaptations),
                "accepted_adaptations": sum(1 for adaptation in adaptations if adaptation.get("accepted_adaptation")),
                "wall_time_s": record.get("wall_time_s"),
                "solve_s": timings.get("solve"),
                "monitor_s": timings.get("monitor"),
                "adapter_s": timings.get("adapter"),
                "max_grid_speed": metrics.get("max_grid_speed"),
                "max_displacement": metrics.get("max_displacement"),
                "mean_displacement": metrics.get("mean_displacement"),
                "orientation_flips": (metrics.get("final_mesh_quality") or {}).get("orientation_flips"),
                "mean_velocity_l2": summary.get("mean_velocity_l2"),
                "max_velocity_l2": summary.get("max_velocity_l2"),
                "mean_velocity_relative_l2": summary.get("mean_velocity_relative_l2"),
                "max_velocity_relative_l2": summary.get("max_velocity_relative_l2"),
                "mean_velocity_linf": summary.get("mean_velocity_linf"),
                "mean_vorticity_l2": summary.get("mean_vorticity_l2"),
                "max_vorticity_l2": summary.get("max_vorticity_l2"),
                "mean_vorticity_relative_l2": summary.get("mean_vorticity_relative_l2"),
                "max_vorticity_relative_l2": summary.get("max_vorticity_relative_l2"),
                "mean_vorticity_linf": summary.get("mean_vorticity_linf"),
            }
        )
    return rows


def _manifest_record_for_result_dir(manifest_by_output: dict[object, dict[str, object]], result_dir: Path) -> dict[str, object]:
    record = manifest_by_output.get(str(result_dir))
    if record is not None:
        return record
    manifest_path = result_dir.parents[1] / "manifest.json"
    if not manifest_path.exists():
        return {}
    for previous in json.loads(manifest_path.read_text(encoding="utf-8")):
        if previous.get("output_dir") == str(result_dir):
            return previous
    return {}


def _read_metrics(result_dir: Path) -> dict[str, object]:
    return json.loads((result_dir / "metrics.json").read_text())


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, object]], headers: list[str] | None = None) -> None:
    headers = headers or list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _paper_table_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    paper_rows = []
    fixed_by_mesh = {}
    for row in rows:
        run = str(row["run"])
        if run != "fine/fixed" and run.endswith("/fixed"):
            mesh, _mode = run.split("/", 1)
            fixed_by_mesh[mesh] = row

    for row in rows:
        run = str(row["run"])
        if run == "fine/fixed":
            continue
        mesh, mode = run.split("/", 1)
        fixed = fixed_by_mesh.get(mesh, {})
        paper_rows.append(
            {
                "mesh": mesh,
                "mode": mode,
                "wall_s": row.get("wall_time_s"),
                "wall_overhead_pct": _percent_change(row.get("wall_time_s"), fixed.get("wall_time_s")),
                "mean_rel_l2_vorticity": row.get("mean_vorticity_relative_l2"),
                "mean_rel_l2_vorticity_improvement_pct": _error_improvement_pct(
                    row.get("mean_vorticity_relative_l2"), fixed.get("mean_vorticity_relative_l2")
                ),
                "mean_rel_l2_velocity": row.get("mean_velocity_relative_l2"),
                "mean_rel_l2_velocity_improvement_pct": _error_improvement_pct(
                    row.get("mean_velocity_relative_l2"), fixed.get("mean_velocity_relative_l2")
                ),
                "max_rel_l2_vorticity": row.get("max_vorticity_relative_l2"),
                "max_rel_l2_vorticity_improvement_pct": _error_improvement_pct(
                    row.get("max_vorticity_relative_l2"), fixed.get("max_vorticity_relative_l2")
                ),
                "max_rel_l2_velocity": row.get("max_velocity_relative_l2"),
                "max_rel_l2_velocity_improvement_pct": _error_improvement_pct(
                    row.get("max_velocity_relative_l2"), fixed.get("max_velocity_relative_l2")
                ),
            }
        )
    paper_rows.sort(key=lambda row: (("coarse", "medium").index(str(row["mesh"])), 0 if row["mode"] == "adapted" else 1))
    return paper_rows


def _error_improvement_pct(value: object, fixed_value: object) -> float | None:
    current = _float_or_none(value)
    fixed = _float_or_none(fixed_value)
    if current is None or fixed is None or fixed == 0.0:
        return None
    return 100.0 * (fixed - current) / fixed


def _percent_change(value: object, baseline: object) -> float | None:
    current = _float_or_none(value)
    base = _float_or_none(baseline)
    if current is None or base is None or base == 0.0:
        return None
    return 100.0 * (current - base) / base


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _write_paper_table_png(path: Path, paper_rows: list[dict[str, object]], full_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = [
        "mesh",
        "mode",
        "wall, s",
        "slower,\n%",
        "mean rel L2\nvorticity",
        "mean rel L2\nvelocity",
        "mean improve,\n% vort/vel",
        "max rel L2\nvorticity",
        "max rel L2\nvelocity",
        "max improve,\n% vort/vel",
    ]
    table_values = [
        [
            row["mesh"],
            row["mode"],
            _format_short_float(row["wall_s"], 1),
            _format_signed_percent(row["wall_overhead_pct"]),
            _format_short_float(row["mean_rel_l2_vorticity"], 4),
            _format_short_float(row["mean_rel_l2_velocity"], 4),
            _format_percent_pair(row["mean_rel_l2_vorticity_improvement_pct"], row["mean_rel_l2_velocity_improvement_pct"]),
            _format_short_float(row["max_rel_l2_vorticity"], 4),
            _format_short_float(row["max_rel_l2_velocity"], 4),
            _format_percent_pair(row["max_rel_l2_vorticity_improvement_pct"], row["max_rel_l2_velocity_improvement_pct"]),
        ]
        for row in paper_rows
    ]
    mesh_lines = []
    for row in full_rows:
        if row["run"] == "fine/fixed" or str(row["run"]).endswith("/fixed"):
            mesh = str(row["run"]).split("/", 1)[0]
            mesh_lines.append(f"{mesh}: {row['vertices']} points, {row['cells']} triangles")

    fig, ax = plt.subplots(figsize=(17.0, 7.0))
    ax.axis("off")
    table = ax.table(cellText=table_values, colLabels=headers, cellLoc="left", colLoc="left", loc="upper left", bbox=[0.02, 0.32, 0.96, 0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(1.0 if row in (0, len(table_values)) else 0.0)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_fontsize(15)
        else:
            cell.set_fontsize(15)
    ax.text(0.03, 0.16, "\n".join(mesh_lines), transform=ax.transAxes, fontsize=18, color="#5b5b5b", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_short_float(value: object, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _format_signed_percent(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):+.1f}"


def _format_percent_pair(first: object, second: object) -> str:
    if first is None or second is None:
        return ""
    return f"{float(first):+.1f} / {float(second):+.1f}"


def _write_paraview_readme(path: Path, pvd_paths: dict[str, Path]) -> None:
    path.write_text(
        "\n".join(
            [
                "# ParaView Series",
                "",
                *[f"- {label}: `{pvd_path}`" for label, pvd_path in pvd_paths.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_comparison_plot(
    path: Path,
    fine_pvd: Path,
    fixed_pvd: Path,
    adapted_pvd: Path,
    *,
    cross_sections: tuple[float, ...],
    case_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine_series = _load_series(fine_pvd)
    fixed_series = _load_series(fixed_pvd)
    adapted_series = _load_series(adapted_pvd)
    common_times = sorted(set(fine_series) & set(fixed_series) & set(adapted_series))
    if not common_times:
        raise ValueError("No common output times for comparison plot")
    time_value = common_times[-1]

    fine_points, fine_triangles, fine_data = _read_vtu(fine_series[time_value])
    fixed_points, _fixed_triangles, _fixed_data = _read_vtu(fixed_series[time_value])
    adapted_points, adapted_triangles, adapted_data = _read_vtu(adapted_series[time_value])
    fixed_error = _interpolate_point_data(fixed_series[time_value], fine_points, "vorticity") - fine_data["vorticity"]
    adapted_error = _interpolate_point_data(adapted_series[time_value], fine_points, "vorticity") - fine_data["vorticity"]

    figure = plt.figure(figsize=(24, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 1.25))
    ax_mesh = figure.add_subplot(grid[0, :2])
    ax_fixed_err = figure.add_subplot(grid[0, 2:])
    ax_vort = figure.add_subplot(grid[1, :2])
    ax_adapted_err = figure.add_subplot(grid[1, 2:])
    section_axes = [figure.add_subplot(grid[2, index]) for index in range(4)]

    _plot_mesh(ax_mesh, adapted_points, adapted_triangles, "Adapted Mesh", case_name=case_name)
    error_limit = UM2N_ERROR_PLOT_LIMIT
    _plot_scalar_on_mesh(
        ax_fixed_err,
        fine_points,
        fine_triangles,
        fixed_error,
        "Vorticity Error Map (Original Mesh)",
        "coolwarm",
        -error_limit,
        error_limit,
        case_name=case_name,
    )
    vorticity_limit = UM2N_VORTICITY_PLOT_LIMIT
    _plot_scalar_on_mesh(
        ax_vort,
        adapted_points,
        adapted_triangles,
        adapted_data["vorticity"],
        "Vorticity (Adapted)",
        "coolwarm",
        -vorticity_limit,
        vorticity_limit,
        case_name=case_name,
    )
    _plot_scalar_on_mesh(
        ax_adapted_err,
        fine_points,
        fine_triangles,
        adapted_error,
        "Vorticity Error Map (Adapted)",
        "coolwarm",
        -error_limit,
        error_limit,
        case_name=case_name,
    )

    for x_value in cross_sections:
        ax_vort.axvline(x_value, color="black", linestyle="--", linewidth=2.0)

    y_values = np.linspace(0.0, CHANNEL_HEIGHT, 500)
    for axis, x_value in zip(section_axes, cross_sections, strict=False):
        fine_line = _sample_vorticity_line(fine_series[time_value], x_value, y_values, case_name=case_name)
        fixed_line = _sample_vorticity_line(fixed_series[time_value], x_value, y_values, case_name=case_name)
        adapted_line = _sample_vorticity_line(adapted_series[time_value], x_value, y_values, case_name=case_name)
        axis.plot(y_values, fine_line, "--", label=f"High Res ({fine_points.shape[0]} vertices)", linewidth=1.8)
        axis.plot(y_values, fixed_line, label=f"Original ({fixed_points.shape[0]} vertices)", linewidth=1.6)
        axis.plot(y_values, adapted_line, label=f"Adapted ({adapted_points.shape[0]} vertices)", linewidth=1.6)
        axis.set_title(f"Vorticity at cross-section (x={x_value:g})")
        axis.set_xlabel("y")
        axis.grid(True, linestyle="--", alpha=0.65)
    section_axes[0].set_ylabel("Vorticity")
    handles, labels = section_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _fine_mesh_error_row(time_value: float, current_vtu: Path, reference_vtu: Path) -> dict[str, float]:
    reference_points, reference_triangles, reference_data = _read_vtu(reference_vtu)
    row: dict[str, float] = {"time": float(time_value)}
    areas = _triangle_areas(reference_points, reference_triangles)
    for name in ("vorticity", "velocity"):
        if name not in reference_data:
            continue
        current = _interpolate_point_data(current_vtu, reference_points, name)
        reference = reference_data[name]
        if current.ndim == 2:
            diff = current[:, :2] - reference[:, :2]
            ref_values = reference[:, :2]
        else:
            diff = current - reference
            ref_values = reference
        diff_integral = _integrate_p1_squared(reference_triangles, areas, diff)
        ref_integral = _integrate_p1_squared(reference_triangles, areas, ref_values)
        valid = np.isfinite(diff).all(axis=-1) if diff.ndim == 2 else np.isfinite(diff)
        row[f"{name}_l2"] = float(np.sqrt(diff_integral))
        row[f"{name}_relative_l2"] = float(np.sqrt(diff_integral) / max(np.sqrt(ref_integral), 1.0e-14))
        row[f"{name}_linf"] = float(np.nanmax(np.linalg.norm(diff[valid], axis=1)) if diff.ndim == 2 else np.nanmax(np.abs(diff[valid])))
    return row


def _interpolate_point_data(vtu_path: Path, target_points: np.ndarray, name: str) -> np.ndarray:
    import matplotlib.tri as tri

    points, triangles, point_data = _read_vtu(vtu_path)
    values = point_data[name]
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)
    if values.ndim == 1:
        interpolator = tri.LinearTriInterpolator(triangulation, values)
        return np.asarray(interpolator(target_points[:, 0], target_points[:, 1]).filled(np.nan))
    components = []
    for component in range(values.shape[1]):
        interpolator = tri.LinearTriInterpolator(triangulation, values[:, component])
        components.append(np.asarray(interpolator(target_points[:, 0], target_points[:, 1]).filled(np.nan)))
    return np.column_stack(components)


def _triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0], :2]
    b = points[triangles[:, 1], :2]
    c = points[triangles[:, 2], :2]
    ab = b - a
    ac = c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def _integrate_p1_squared(triangles: np.ndarray, areas: np.ndarray, values: np.ndarray) -> float:
    tri_values = values[triangles]
    if values.ndim == 1:
        valid = np.isfinite(tri_values).all(axis=1)
        f0, f1, f2 = tri_values[valid].T
        integral = areas[valid] / 6.0 * (f0 * f0 + f1 * f1 + f2 * f2 + f0 * f1 + f1 * f2 + f2 * f0)
    else:
        valid = np.isfinite(tri_values).all(axis=(1, 2))
        f0, f1, f2 = tri_values[valid, 0, :], tri_values[valid, 1, :], tri_values[valid, 2, :]
        integral = areas[valid] / 6.0 * (
            np.einsum("ij,ij->i", f0, f0)
            + np.einsum("ij,ij->i", f1, f1)
            + np.einsum("ij,ij->i", f2, f2)
            + np.einsum("ij,ij->i", f0, f1)
            + np.einsum("ij,ij->i", f1, f2)
            + np.einsum("ij,ij->i", f2, f0)
        )
    return float(np.sum(integral))


def _sample_vorticity_line(vtu_path: Path, x_value: float, y_values: np.ndarray, *, case_name: str) -> np.ndarray:
    import matplotlib.tri as tri

    sample_points = np.column_stack([np.full_like(y_values, x_value), y_values])
    points, triangles, point_data = _read_vtu(vtu_path)
    interpolator = tri.LinearTriInterpolator(tri.Triangulation(points[:, 0], points[:, 1], triangles), point_data["vorticity"])
    values = np.asarray(interpolator(sample_points[:, 0], sample_points[:, 1]).filled(np.nan))
    values[_inside_cylinders(sample_points, case_name=case_name)] = np.nan
    return values


def _write_adapted_diagnostics(path: Path, adapted_pvd: Path, *, case_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _load_series(adapted_pvd)
    times = sorted(series)
    if not times:
        raise ValueError(f"No frames in {adapted_pvd}")
    initial_points, initial_triangles, _initial_data = _read_vtu(series[times[0]])
    points, triangles, data = _read_vtu(series[times[-1]])
    displacement = np.linalg.norm(points[:, :2] - initial_points[:, :2], axis=1)
    monitor = data.get("monitor")
    if monitor is None:
        monitor = np.zeros(points.shape[0], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.5), constrained_layout=True)
    vort = axes[0].tripcolor(
        points[:, 0],
        points[:, 1],
        triangles,
        data["vorticity"],
        shading="gouraud",
        cmap="coolwarm",
        vmin=-UM2N_VORTICITY_PLOT_LIMIT,
        vmax=UM2N_VORTICITY_PLOT_LIMIT,
    )
    fig.colorbar(vort, ax=axes[0], shrink=0.82)
    _draw_cylinders_and_format(axes[0], "Vorticity", case_name)

    mon = axes[1].tripcolor(points[:, 0], points[:, 1], triangles, monitor, shading="gouraud", cmap="magma")
    axes[1].triplot(points[:, 0], points[:, 1], triangles, color="white", linewidth=0.05, alpha=0.18)
    fig.colorbar(mon, ax=axes[1], shrink=0.82)
    _draw_cylinders_and_format(axes[1], "Monitor", case_name)

    disp = axes[2].tripcolor(points[:, 0], points[:, 1], triangles, displacement, shading="gouraud", cmap="viridis")
    fig.colorbar(disp, ax=axes[2], shrink=0.82)
    _draw_cylinders_and_format(axes[2], "Mesh displacement", case_name)

    axes[3].triplot(points[:, 0], points[:, 1], triangles, color="black", linewidth=0.09, alpha=0.85)
    _draw_cylinders_and_format(axes[3], "Differentiable adapted mesh", case_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=280)
    plt.close(fig)


def _draw_cylinders_and_format(ax, title: str, case_name: str) -> None:
    import matplotlib.pyplot as plt

    for cx, cy in CASES[case_name]:
        ax.add_patch(plt.Circle((cx, cy), CYLINDER_RADIUS, edgecolor="black", facecolor="white", linewidth=0.8, zorder=10))
    _format_channel_axis(ax, title)


def _inside_cylinders(points: np.ndarray, *, case_name: str) -> np.ndarray:
    mask = np.zeros(points.shape[0], dtype=bool)
    for cx, cy in CASES[case_name]:
        mask |= (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 <= (CYLINDER_RADIUS * 1.02) ** 2
    return mask


def _plot_mesh(ax, points: np.ndarray, triangles: np.ndarray, title: str, *, case_name: str) -> None:
    import matplotlib.pyplot as plt

    ax.triplot(points[:, 0], points[:, 1], triangles, color="black", linewidth=0.12)
    for cx, cy in CASES[case_name]:
        ax.add_patch(plt.Circle((cx, cy), CYLINDER_RADIUS, edgecolor="black", facecolor="white", linewidth=1.0, zorder=10))
    _format_channel_axis(ax, title)


def _plot_scalar_on_mesh(
    ax,
    points: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    *,
    case_name: str,
) -> None:
    import matplotlib.pyplot as plt

    ax.tripcolor(
        points[:, 0],
        points[:, 1],
        triangles,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    for cx, cy in CASES[case_name]:
        ax.add_patch(plt.Circle((cx, cy), CYLINDER_RADIUS, edgecolor="black", facecolor="white", linewidth=1.0, zorder=10))
    _format_channel_axis(ax, title)


def _format_channel_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(0.0, CHANNEL_LENGTH)
    ax.set_ylim(0.0, CHANNEL_HEIGHT)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


if __name__ == "__main__":
    main()
