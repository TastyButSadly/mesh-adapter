from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import firedrake as fd
import movement as mv

CYLINDER_CENTER = (0.2, 0.2)
CYLINDER_RADIUS = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Firedrake-side Monge-Ampere cylinder adaptation.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()

    paths = run_case(args.mesh, args.output_dir, rtol=args.rtol, maxiter=args.maxiter)
    print(f"before: {paths['before']}")
    print(f"after: {paths['after']}")
    print(f"series: {paths['series']}")
    print(f"metrics: {paths['metrics']}")


def run_case(mesh_path: Path, output_dir: Path, *, rtol: float, maxiter: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = fd.Mesh(str(mesh_path))
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()
    monitor_before = _monitor(mesh)

    before = output_dir / "before.pvd"
    fd.VTKFile(str(before)).write(monitor_before)

    start = time.perf_counter()
    mover = mv.MongeAmpereMover(
        mesh,
        _monitor,
        rtol=rtol,
        maxiter=maxiter,
        fixed_boundary_segments=["on_boundary"],
    )
    iteration_index = mover.move()
    movement_time = time.perf_counter() - start

    adapted_mesh = mover.mesh
    adapted_coordinates = adapted_mesh.coordinates.dat.data_ro.copy()
    displacement = np.linalg.norm(adapted_coordinates - initial_coordinates, axis=1)

    monitor_after = _monitor(adapted_mesh)
    displacement_field = fd.Function(fd.FunctionSpace(adapted_mesh, "CG", 1), name="mesh_displacement")
    displacement_field.dat.data[:] = displacement

    after = output_dir / "after.pvd"
    fd.VTKFile(str(after)).write(monitor_after, displacement_field)

    series = _write_pvd_series(output_dir / "monge_ampere_cylinder_adaptation.pvd", [(0.0, before), (1.0, after)])
    metrics = output_dir / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "adapter": "movement_monge_ampere",
                "mesh": str(mesh_path),
                "num_vertices": int(adapted_mesh.num_vertices()),
                "num_cells": int(adapted_mesh.num_cells()),
                "rtol": rtol,
                "maxiter": maxiter,
                "iterations": int(iteration_index) + 1,
                "movement_time_s": movement_time,
                "monitor_min": float(monitor_before.dat.data_ro.min()),
                "monitor_max": float(monitor_before.dat.data_ro.max()),
                "monitor_mean": float(monitor_before.dat.data_ro.mean()),
                "max_displacement": float(displacement.max()),
                "mean_displacement": float(displacement.mean()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {"before": before, "after": after, "series": series, "metrics": metrics}


def _monitor(mesh) -> fd.Function:
    V = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    cx = fd.Constant(CYLINDER_CENTER[0])
    cy = fd.Constant(CYLINDER_CENTER[1])
    radius = fd.Constant(CYLINDER_RADIUS)

    radial_distance = fd.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    cylinder_layer = fd.exp(-((radial_distance - radius) / 0.030) ** 2)
    wake = fd.exp(-((y - cy) / 0.065) ** 2) * fd.exp(-((x - 0.65) / 0.65) ** 2)

    monitor = fd.Function(V, name="monitor")
    monitor.interpolate(1.0 + 1.3 * cylinder_layer + 0.9 * wake)
    return monitor


def _write_pvd_series(path: Path, datasets: list[tuple[float, Path]]) -> Path:
    root = ElementTree.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ElementTree.SubElement(root, "Collection")
    for time_value, pvd_path in datasets:
        vtu_path = _single_vtu_from_pvd(pvd_path)
        ElementTree.SubElement(
            collection,
            "DataSet",
            timestep=f"{time_value:.12g}",
            group="",
            part="0",
            file=vtu_path.relative_to(path.parent).as_posix(),
        )
    tree = ElementTree.ElementTree(root)
    ElementTree.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _single_vtu_from_pvd(path: Path) -> Path:
    root = ElementTree.parse(path).getroot()
    dataset = root.find("./Collection/DataSet")
    if dataset is None or "file" not in dataset.attrib:
        raise ValueError(f"Could not find a VTU dataset in {path}")
    return path.parent / dataset.attrib["file"]


if __name__ == "__main__":
    main()
