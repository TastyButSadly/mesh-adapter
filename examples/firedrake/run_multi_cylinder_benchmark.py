from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path

from examples.firedrake.multi_cylinder_meshes import CASES, RESOLUTIONS, write_multi_cylinder_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed/adapted multi-cylinder benchmark cases.")
    parser.add_argument("--mesh-dir", type=Path, default=Path("outputs/multi_cylinder_mesh_family"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_cylinder_benchmark"))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--resolutions", nargs="+", choices=[r.name for r in RESOLUTIONS], default=[r.name for r in RESOLUTIONS])
    parser.add_argument("--modes", nargs="+", choices=("fixed", "adapted"), default=("fixed", "adapted"))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--monitor-scale", type=float, default=3.0)
    parser.add_argument("--adapter-steps", type=int, default=12)
    parser.add_argument("--adapter-lr", type=float, default=8.0e-4)
    parser.add_argument("--max-grid-speed", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent runs to execute in parallel.")
    parser.add_argument("--run", action="store_true", help="Execute runs. Without this, only writes the manifest.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_meshes(args.mesh_dir, args.cases)

    manifest: list[dict[str, object]] = []
    for case_name in args.cases:
        for resolution_name in args.resolutions:
            mesh_path = args.mesh_dir / case_name / f"{case_name}_{resolution_name}.msh"
            for mode in args.modes:
                run_dir = args.output_dir / case_name / resolution_name / mode
                command = _run_command(
                    mesh_path=mesh_path,
                    output_dir=run_dir,
                    mode=mode,
                    dt=args.dt,
                    steps=args.steps,
                    save_every=args.save_every,
                    adapt_every=args.adapt_every,
                    monitor_scale=args.monitor_scale,
                    adapter_steps=args.adapter_steps,
                    adapter_lr=args.adapter_lr,
                    max_grid_speed=args.max_grid_speed,
                )
                record: dict[str, object] = {
                    "case": case_name,
                    "resolution": resolution_name,
                    "mode": mode,
                    "mesh": str(mesh_path),
                    "output_dir": str(run_dir),
                    "command": command,
                }
                manifest.append(record)

    if args.run:
        _run_manifest(manifest, jobs=max(args.jobs, 1))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


def _run_manifest(manifest: list[dict[str, object]], *, jobs: int) -> None:
    if jobs == 1:
        for record in manifest:
            _run_record(record)
        return
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_run_record, record) for record in manifest]
        for future in as_completed(futures):
            future.result()


def _run_record(record: dict[str, object]) -> None:
    command = [str(part) for part in record["command"]]
    label = f"{record['case']} {record['resolution']} {record['mode']}"
    start = time.perf_counter()
    print(f"RUN {label}", flush=True)
    completed = subprocess.run(command, check=False)
    record["returncode"] = completed.returncode
    record["wall_time_s"] = time.perf_counter() - start
    if completed.returncode == 0:
        print(f"DONE {label}: {record['wall_time_s']:.3f}s", flush=True)
    else:
        print(f"FAILED {label}: {completed.returncode}", flush=True)


def _ensure_meshes(mesh_dir: Path, case_names: list[str]) -> None:
    resolution_by_name = {resolution.name: resolution for resolution in RESOLUTIONS}
    for case_name in case_names:
        for resolution_name, resolution in resolution_by_name.items():
            mesh_path = mesh_dir / case_name / f"{case_name}_{resolution_name}.msh"
            if not mesh_path.exists():
                write_multi_cylinder_mesh(mesh_path, CASES[case_name], resolution, preview_path=mesh_path.with_suffix(".png"))


def _run_command(
    *,
    mesh_path: Path,
    output_dir: Path,
    mode: str,
    dt: float,
    steps: int,
    save_every: int,
    adapt_every: int,
    monitor_scale: float,
    adapter_steps: int,
    adapter_lr: float,
    max_grid_speed: float,
) -> list[str]:
    effective_adapt_every = adapt_every if mode == "adapted" else steps + 1
    return [
        sys.executable,
        "-m",
        "examples.firedrake.um2n_diff_adapter_cylinder",
        "--mesh",
        mesh_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--dt",
        str(dt),
        "--steps",
        str(steps),
        "--adapt-every",
        str(effective_adapt_every),
        "--save-every",
        str(save_every),
        "--monitor-frame",
        "current",
        "--monitor-scale",
        str(monitor_scale),
        "--adapter-profile",
        "regularized",
        "--adapter-steps",
        str(adapter_steps),
        "--adapter-lr",
        str(adapter_lr),
        "--max-grid-speed",
        str(max_grid_speed),
    ]


if __name__ == "__main__":
    main()
