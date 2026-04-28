from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


DEFAULT_MESH = Path("examples/firedrake/um2n_reference/meshes/cylinder_015.msh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UM2N cylinder setup with differentiable fixed-topology adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/um2n_diff_adapter_cylinder"))
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--firedrake-run", default="firedrake-run")
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
    parser.add_argument("--adapter-poll-interval", type=float, default=0.02)
    args = parser.parse_args()
    adapter_poll_interval = max(args.adapter_poll_interval, 0.001)

    cwd = Path.cwd().resolve()
    output_dir = args.output_dir
    try:
        command_output_dir = _container_visible_path(output_dir, cwd, "--output-dir")
        command_mesh = _container_visible_path(args.mesh, cwd, "--mesh")
    except ValueError as exc:
        parser.error(str(exc))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    exchange_dir = output_dir / "adapter_exchange"
    exchange_dir.mkdir(parents=True)
    command_exchange_dir = command_output_dir / "adapter_exchange"

    command = [
        args.firedrake_run,
        "python3",
        "-m",
        "examples.firedrake._um2n_diff_adapter_cylinder_firedrake",
        "--mesh",
        command_mesh.as_posix(),
        "--output-dir",
        command_output_dir.as_posix(),
        "--dt",
        str(args.dt),
        "--steps",
        str(args.steps),
        "--adapt-every",
        str(args.adapt_every),
        "--save-every",
        str(args.save_every),
        "--monitor-kind",
        args.monitor_kind,
        "--monitor-frame",
        args.monitor_frame,
        "--monitor-scale",
        str(args.monitor_scale),
        "--adaptation-relaxation",
        str(args.adaptation_relaxation),
        "--max-grid-speed",
        str(args.max_grid_speed),
        "--adapter-steps",
        str(args.adapter_steps),
        "--adapter-lr",
        str(args.adapter_lr),
        "--adapter-profile",
        args.adapter_profile,
        "--adapter-exchange-dir",
        command_exchange_dir.as_posix(),
        "--adapter-poll-interval",
        str(adapter_poll_interval),
    ]
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_serve_adapter_requests,
        args=(exchange_dir, stop_event, adapter_poll_interval),
        daemon=True,
    )
    worker.start()
    process = subprocess.Popen(command)
    try:
        return_code = process.wait()
    finally:
        stop_event.set()
        worker.join(timeout=5.0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"mesh: {args.mesh}")
    print(f"ParaView series: {output_dir / 'um2n_diff_adapter_cylinder.pvd'}")
    print(f"final plot: {output_dir / 'final_vorticity_mesh.png'}")
    print(f"metrics: {output_dir / 'metrics.json'}")


def _serve_adapter_requests(exchange_dir: Path, stop_event: threading.Event, poll_interval: float) -> None:
    adapt_coordinates = None
    processed: set[Path] = set()
    while not stop_event.is_set():
        for request_path in sorted(exchange_dir.glob("request_*.npz")):
            if request_path in processed:
                continue
            response_path = exchange_dir / request_path.name.replace("request_", "response_", 1)
            if response_path.exists():
                processed.add(request_path)
                continue
            try:
                service_start = time.perf_counter()
                if adapt_coordinates is None:
                    from examples.firedrake._diff_adapter_subprocess import adapt_coordinates as loaded_adapt_coordinates

                    adapt_coordinates = loaded_adapt_coordinates
                data = np.load(request_path)
                cells_np = data["cells"] if "cells" in data else data["triangles"]
                points, info = adapt_coordinates(
                    points_np=data["points"],
                    cells_np=cells_np,
                    monitor_np=data["monitor"],
                    steps=int(data["steps"]),
                    lr=float(data["lr"]),
                    profile=str(data["profile"]) if "profile" in data else "regularized",
                )
                info["service_s"] = time.perf_counter() - service_start
                _write_adapter_response(response_path, points, info)
            except Exception as exc:
                _write_error_response(response_path, f"{type(exc).__name__}: {exc}")
            processed.add(request_path)
        time.sleep(poll_interval)


def _container_visible_path(path: Path, cwd: Path, argument_name: str) -> Path:
    if not path.is_absolute():
        return path
    try:
        return path.resolve().relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} must be relative to the repository or inside {cwd}; "
            "firedrake-run only mounts this workspace into the container."
        ) from exc


def _write_adapter_response(path: Path, points: np.ndarray, info: dict[str, float | int | bool]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            points=points,
            initial_loss=np.array(info["initial_loss"], dtype=np.float64),
            final_loss=np.array(info["final_loss"], dtype=np.float64),
            steps_completed=np.array(info["steps_completed"], dtype=np.int64),
            early_stopped=np.array(info["early_stopped"], dtype=bool),
            service_s=np.array(info.get("service_s", np.nan), dtype=np.float64),
        )
    tmp_path.rename(path)


def _write_error_response(path: Path, message: str) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, error=np.array(message))
    tmp_path.rename(path)


if __name__ == "__main__":
    main()
