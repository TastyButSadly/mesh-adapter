from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

CHANNEL_LENGTH = 2.2
CHANNEL_HEIGHT = 0.41
CYLINDER_RADIUS = 0.05

WALL_MARKER = 1
INLET_MARKER = 2
OUTLET_MARKER = 3
CYLINDER_MARKER = 4


@dataclass(frozen=True)
class MeshResolution:
    name: str
    mesh_size: float
    cylinder_size: float
    wake_size: float
    wake_distance: float


RESOLUTIONS = (
    MeshResolution("coarse", mesh_size=0.050, cylinder_size=0.020, wake_size=0.035, wake_distance=0.35),
    MeshResolution("medium", mesh_size=0.030, cylinder_size=0.010, wake_size=0.020, wake_distance=0.45),
    MeshResolution("fine", mesh_size=0.016, cylinder_size=0.005, wake_size=0.010, wake_distance=0.55),
)

CASES: dict[str, tuple[tuple[float, float], ...]] = {
    "multiple_cylinders": (
        (0.20, 0.20),
        (0.45, 0.28),
        (0.70, 0.14),
        (0.95, 0.28),
        (1.20, 0.14),
    ),
    "v_formation": (
        (0.34, 0.205),
        (0.62, 0.270),
        (0.62, 0.140),
        (0.90, 0.315),
        (0.90, 0.095),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UM2N-style multi-cylinder channel meshes.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_cylinder_mesh_family"))
    parser.add_argument("--preview", action="store_true", help="Also write PNG previews of generated meshes.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_name, centers in CASES.items():
        for resolution in RESOLUTIONS:
            mesh_path = args.output_dir / case_name / f"{case_name}_{resolution.name}.msh"
            preview_path = mesh_path.with_suffix(".png") if args.preview else None
            stats = write_multi_cylinder_mesh(mesh_path, centers, resolution, preview_path=preview_path)
            print(
                f"{mesh_path}: points={stats['points']} triangles={stats['triangles']} "
                f"mesh_size={resolution.mesh_size} cylinder_size={resolution.cylinder_size}"
            )


def write_multi_cylinder_mesh(
    path: Path,
    centers: tuple[tuple[float, float], ...],
    resolution: MeshResolution,
    *,
    preview_path: Path | None = None,
) -> dict[str, int]:
    import gmsh

    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
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

        for dim, tag in gmsh.model.getEntities(0):
            x, y, _ = gmsh.model.getValue(dim, tag, [])
            distance = min(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for cx, cy in centers)
            size = resolution.cylinder_size if distance < 0.18 else resolution.mesh_size
            gmsh.model.mesh.setSize([(dim, tag)], size)

        field = gmsh.model.mesh.field
        field.add("Distance", 1)
        field.setNumbers(1, "CurvesList", boundary_groups["cylinders"])
        field.add("Threshold", 2)
        field.setNumber(2, "InField", 1)
        field.setNumber(2, "SizeMin", resolution.cylinder_size)
        field.setNumber(2, "SizeMax", resolution.mesh_size)
        field.setNumber(2, "DistMin", CYLINDER_RADIUS)
        field.setNumber(2, "DistMax", resolution.wake_distance)
        field.add("Min", 3)
        field.setNumbers(3, "FieldsList", [2])
        field.setAsBackgroundMesh(3)

        gmsh.model.mesh.generate(2)
        stats = _mesh_stats(gmsh)
        gmsh.write(str(path))
        if preview_path is not None:
            _write_preview(gmsh, centers, preview_path)
        return stats
    finally:
        gmsh.finalize()


def _classify_boundary_curves(gmsh_module) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"walls": [], "inlet": [], "outlet": [], "cylinders": []}
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
            groups["cylinders"].append(tag)
    return groups


def _mesh_stats(gmsh_module) -> dict[str, int]:
    node_tags, _, _ = gmsh_module.model.mesh.getNodes()
    element_types, _, element_node_tags = gmsh_module.model.mesh.getElements(2)
    triangles = 0
    for element_type, nodes in zip(element_types, element_node_tags, strict=True):
        name, _dim, _order, num_nodes, _local_coords, _num_primary = gmsh_module.model.mesh.getElementProperties(element_type)
        if name == "Triangle 3":
            triangles += len(nodes) // num_nodes
    return {"points": len(node_tags), "triangles": triangles}


def _write_preview(gmsh_module, centers: tuple[tuple[float, float], ...], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    node_tags, coordinates, _ = gmsh_module.model.mesh.getNodes()
    tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
    points = np.asarray(coordinates, dtype=float).reshape(-1, 3)

    triangles: list[list[int]] = []
    element_types, _, element_node_tags = gmsh_module.model.mesh.getElements(2)
    for element_type, nodes in zip(element_types, element_node_tags, strict=True):
        name, _dim, _order, num_nodes, _local_coords, _num_primary = gmsh_module.model.mesh.getElementProperties(element_type)
        if name != "Triangle 3":
            continue
        for start in range(0, len(nodes), num_nodes):
            triangles.append([tag_to_index[int(tag)] for tag in nodes[start : start + num_nodes]])

    triangulation = tri.Triangulation(points[:, 0], points[:, 1], np.asarray(triangles, dtype=int))
    fig, ax = plt.subplots(figsize=(13, 3.2), constrained_layout=True)
    ax.triplot(triangulation, color="black", linewidth=0.15)
    for cx, cy in centers:
        circle = plt.Circle((cx, cy), CYLINDER_RADIUS, edgecolor="tab:red", facecolor="white", linewidth=1.2)
        ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, CHANNEL_LENGTH)
    ax.set_ylim(0.0, CHANNEL_HEIGHT)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
