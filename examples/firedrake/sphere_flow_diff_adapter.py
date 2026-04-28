from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
from pathlib import Path

from examples.firedrake.sphere_flow_mesh import DEMO_RESOLUTION, write_sphere_channel_mesh
from examples.firedrake.um2n_diff_adapter_cylinder import _container_visible_path, _serve_adapter_requests

DEFAULT_MESH = Path("outputs/sphere_flow_meshes/sphere_channel_demo.msh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3D sphere-flow setup with differentiable fixed-topology adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sphere_flow_demo"))
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--firedrake-run", default="firedrake-run")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--monitor-kind", choices=("velocity-gradient", "vorticity-magnitude"), default="velocity-gradient")
    parser.add_argument("--monitor-scale", type=float, default=3.0)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=1.0)
    parser.add_argument("--adapter-steps", type=int, default=8)
    parser.add_argument("--adapter-lr", type=float, default=5.0e-4)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only"), default="regularized")
    parser.add_argument("--adapter-poll-interval", type=float, default=0.02)
    args = parser.parse_args()

    if args.mesh == DEFAULT_MESH and not args.mesh.exists():
        stats = write_sphere_channel_mesh(args.mesh, DEMO_RESOLUTION)
        print(f"generated mesh: {args.mesh} ({stats['points']} points, {stats['tetrahedra']} tetrahedra)")

    adapter_poll_interval = max(args.adapter_poll_interval, 0.001)
    cwd = Path.cwd().resolve()
    try:
        command_output_dir = _container_visible_path(args.output_dir, cwd, "--output-dir")
        command_mesh = _container_visible_path(args.mesh, cwd, "--mesh")
    except ValueError as exc:
        parser.error(str(exc))

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    exchange_dir = args.output_dir / "adapter_exchange"
    exchange_dir.mkdir(parents=True)
    command_exchange_dir = command_output_dir / "adapter_exchange"

    command = [
        args.firedrake_run,
        "python3",
        "-m",
        "examples.firedrake._sphere_flow_diff_adapter_firedrake",
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
    print(f"ParaView series: {args.output_dir / 'sphere_flow_diff_adapter.pvd'}")
    print(f"metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
