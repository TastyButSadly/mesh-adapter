from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from examples.firedrake.monge_ampere_cylinder_adaptation import _write_cylinder_channel_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cylinder Navier-Stokes with Monge-Ampere mesh adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/firedrake_ns_monge_ampere_cylinder"))
    parser.add_argument("--firedrake-run", default="firedrake-run")
    parser.add_argument("--mesh-size", type=float, default=0.045)
    parser.add_argument("--cylinder-size", type=float, default=0.015)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--adapt-start", type=float, default=0.4)
    parser.add_argument("--adapt-interval", type=int, default=50)
    parser.add_argument("--motion-substeps", type=int, default=8)
    parser.add_argument("--write-stride", type=int, default=25)
    parser.add_argument("--mean-speed", type=float, default=1.0)
    parser.add_argument("--inlet-perturbation", type=float, default=0.02)
    parser.add_argument("--monitor-kind", choices=("gradient", "vorticity"), default="vorticity")
    parser.add_argument("--monitor-gain", type=float, default=6.0)
    parser.add_argument("--monitor-ceiling", type=float, default=1.0e3)
    parser.add_argument("--adapt-relaxation", type=float, default=0.5)
    parser.add_argument("--max-adapt-displacement", type=float, default=0.006)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    mesh_path = output_dir / "cylinder_channel.msh"
    _write_cylinder_channel_mesh(mesh_path, mesh_size=args.mesh_size, cylinder_size=args.cylinder_size)

    command = [
        args.firedrake_run,
        "python3",
        "-m",
        "examples.firedrake._navier_stokes_monge_ampere_cylinder_firedrake",
        "--mesh",
        mesh_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--dt",
        str(args.dt),
        "--final-time",
        str(args.final_time),
        "--adapt-start",
        str(args.adapt_start),
        "--adapt-interval",
        str(args.adapt_interval),
        "--motion-substeps",
        str(args.motion_substeps),
        "--write-stride",
        str(args.write_stride),
        "--mean-speed",
        str(args.mean_speed),
        "--inlet-perturbation",
        str(args.inlet_perturbation),
        "--monitor-kind",
        args.monitor_kind,
        "--monitor-gain",
        str(args.monitor_gain),
        "--monitor-ceiling",
        str(args.monitor_ceiling),
        "--adapt-relaxation",
        str(args.adapt_relaxation),
        "--max-adapt-displacement",
        str(args.max_adapt_displacement),
        "--rtol",
        str(args.rtol),
        "--maxiter",
        str(args.maxiter),
    ]
    subprocess.run(command, check=True)

    print(f"mesh: {mesh_path}")
    print(f"ParaView series: {output_dir / 'navier_stokes_monge_ampere_cylinder.pvd'}")
    print(f"final plot: {output_dir / 'final_vorticity_mesh.png'}")
    print(f"metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
