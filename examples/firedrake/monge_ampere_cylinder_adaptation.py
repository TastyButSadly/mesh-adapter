from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CHANNEL_LENGTH = 2.2
CHANNEL_HEIGHT = 0.41
CYLINDER_CENTER = (0.2, 0.2)
CYLINDER_RADIUS = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Firedrake/Movement Monge-Ampere cylinder mesh adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/firedrake_monge_ampere_cylinder"))
    parser.add_argument("--firedrake-run", default="firedrake-run")
    parser.add_argument("--mesh-size", type=float, default=0.035)
    parser.add_argument("--cylinder-size", type=float, default=0.012)
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
        "examples.firedrake._monge_ampere_cylinder_firedrake",
        "--mesh",
        _container_relative_path(mesh_path),
        "--output-dir",
        _container_relative_path(output_dir),
        "--rtol",
        str(args.rtol),
        "--maxiter",
        str(args.maxiter),
    ]
    subprocess.run(command, check=True)

    print(f"mesh: {mesh_path}")
    print(f"ParaView series: {output_dir / 'monge_ampere_cylinder_adaptation.pvd'}")
    print(f"metrics: {output_dir / 'metrics.json'}")


def _write_cylinder_channel_mesh(path: Path, *, mesh_size: float, cylinder_size: float) -> None:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("cylinder_channel")

        occ = gmsh.model.occ
        rectangle = occ.addRectangle(0.0, 0.0, 0.0, CHANNEL_LENGTH, CHANNEL_HEIGHT)
        cylinder = occ.addDisk(CYLINDER_CENTER[0], CYLINDER_CENTER[1], 0.0, CYLINDER_RADIUS, CYLINDER_RADIUS)
        occ.cut([(2, rectangle)], [(2, cylinder)], removeObject=True, removeTool=True)
        occ.synchronize()

        surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "fluid")

        boundary_groups = _classify_boundary_curves(gmsh)
        for marker, name in ((10, "inlet"), (11, "outlet"), (12, "walls"), (13, "cylinder")):
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name)

        for dim, tag in gmsh.model.getEntities(0):
            x, y, _ = gmsh.model.getValue(dim, tag, [])
            distance_to_cylinder = ((x - CYLINDER_CENTER[0]) ** 2 + (y - CYLINDER_CENTER[1]) ** 2) ** 0.5
            size = cylinder_size if distance_to_cylinder < 0.14 else mesh_size
            gmsh.model.mesh.setSize([(dim, tag)], size)

        field = gmsh.model.mesh.field
        field.add("Distance", 1)
        field.setNumbers(1, "CurvesList", boundary_groups["cylinder"])
        field.add("Threshold", 2)
        field.setNumber(2, "InField", 1)
        field.setNumber(2, "SizeMin", cylinder_size)
        field.setNumber(2, "SizeMax", mesh_size)
        field.setNumber(2, "DistMin", CYLINDER_RADIUS)
        field.setNumber(2, "DistMax", 0.55)
        field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _classify_boundary_curves(gmsh_module) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"inlet": [], "outlet": [], "walls": [], "cylinder": []}
    for dim, tag in gmsh_module.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh_module.model.getBoundingBox(dim, tag)
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        if abs(center_x) < 1e-8:
            groups["inlet"].append(tag)
        elif abs(center_x - CHANNEL_LENGTH) < 1e-8:
            groups["outlet"].append(tag)
        elif abs(center_y) < 1e-8 or abs(center_y - CHANNEL_HEIGHT) < 1e-8:
            groups["walls"].append(tag)
        else:
            groups["cylinder"].append(tag)
    return groups


def _container_relative_path(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    main()
