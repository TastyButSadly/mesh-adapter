from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_MESH = Path("examples/firedrake/um2n_reference/meshes/cylinder_015.msh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UM2N cylinder setup with Movement Monge-Ampere adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/um2n_monge_ampere_cylinder"))
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
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    command = [
        args.firedrake_run,
        "python3",
        "-m",
        "examples.firedrake._um2n_monge_ampere_cylinder_firedrake",
        "--mesh",
        args.mesh.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
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
        "--rtol",
        str(args.rtol),
        "--maxiter",
        str(args.maxiter),
    ]
    subprocess.run(command, check=True)

    print(f"mesh: {args.mesh}")
    print(f"ParaView series: {output_dir / 'um2n_monge_ampere_cylinder.pvd'}")
    print(f"final plot: {output_dir / 'final_vorticity_mesh.png'}")
    print(f"metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
