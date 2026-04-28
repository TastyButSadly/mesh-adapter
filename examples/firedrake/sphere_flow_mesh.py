from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

CHANNEL_LENGTH = 2.4
CHANNEL_HALF_WIDTH = 0.4
SPHERE_CENTER = (0.45, 0.0, 0.0)
SPHERE_RADIUS = 0.1

WALL_MARKER = 1
INLET_MARKER = 2
OUTLET_MARKER = 3
SPHERE_MARKER = 4
FLUID_MARKER = 1


@dataclass(frozen=True)
class SphereMeshResolution:
    name: str
    mesh_size: float
    sphere_size: float
    wake_size: float
    wake_distance: float


DEMO_RESOLUTION = SphereMeshResolution(
    "demo",
    mesh_size=0.24,
    sphere_size=0.085,
    wake_size=0.12,
    wake_distance=0.25,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3D channel-with-sphere tetrahedral mesh.")
    parser.add_argument("--output", type=Path, default=Path("outputs/sphere_flow_meshes/sphere_channel_demo.msh"))
    parser.add_argument("--mesh-size", type=float, default=DEMO_RESOLUTION.mesh_size)
    parser.add_argument("--sphere-size", type=float, default=DEMO_RESOLUTION.sphere_size)
    parser.add_argument("--wake-size", type=float, default=DEMO_RESOLUTION.wake_size)
    parser.add_argument("--wake-distance", type=float, default=DEMO_RESOLUTION.wake_distance)
    args = parser.parse_args()

    resolution = SphereMeshResolution(
        "custom",
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        wake_size=args.wake_size,
        wake_distance=args.wake_distance,
    )
    stats = write_sphere_channel_mesh(args.output, resolution)
    print(
        f"{args.output}: points={stats['points']} tetrahedra={stats['tetrahedra']} "
        f"mesh_size={resolution.mesh_size} sphere_size={resolution.sphere_size}"
    )


def write_sphere_channel_mesh(path: Path, resolution: SphereMeshResolution = DEMO_RESOLUTION) -> dict[str, int]:
    import gmsh

    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add(path.stem)

        occ = gmsh.model.occ
        box = occ.addBox(
            0.0,
            -CHANNEL_HALF_WIDTH,
            -CHANNEL_HALF_WIDTH,
            CHANNEL_LENGTH,
            2.0 * CHANNEL_HALF_WIDTH,
            2.0 * CHANNEL_HALF_WIDTH,
        )
        sphere = occ.addSphere(*SPHERE_CENTER, SPHERE_RADIUS)
        cut_result, _ = occ.cut([(3, box)], [(3, sphere)], removeObject=True, removeTool=True)
        occ.synchronize()

        volumes = [tag for dim, tag in cut_result if dim == 3]
        gmsh.model.addPhysicalGroup(3, volumes, FLUID_MARKER)
        gmsh.model.setPhysicalName(3, FLUID_MARKER, "fluid")

        boundary_groups = _classify_boundary_surfaces(gmsh)
        for marker, name in (
            (WALL_MARKER, "walls"),
            (INLET_MARKER, "inlet"),
            (OUTLET_MARKER, "outlet"),
            (SPHERE_MARKER, "sphere"),
        ):
            if boundary_groups[name]:
                gmsh.model.addPhysicalGroup(2, boundary_groups[name], marker)
                gmsh.model.setPhysicalName(2, marker, name)

        for dim, tag in gmsh.model.getEntities(0):
            x, y, z = gmsh.model.getValue(dim, tag, [])
            distance = ((x - SPHERE_CENTER[0]) ** 2 + (y - SPHERE_CENTER[1]) ** 2 + (z - SPHERE_CENTER[2]) ** 2) ** 0.5
            size = resolution.sphere_size if distance < 0.22 else resolution.mesh_size
            gmsh.model.mesh.setSize([(dim, tag)], size)

        field = gmsh.model.mesh.field
        field.add("Distance", 1)
        field.setNumbers(1, "FacesList", boundary_groups["sphere"])
        field.add("Threshold", 2)
        field.setNumber(2, "InField", 1)
        field.setNumber(2, "SizeMin", resolution.sphere_size)
        field.setNumber(2, "SizeMax", resolution.mesh_size)
        field.setNumber(2, "DistMin", SPHERE_RADIUS)
        field.setNumber(2, "DistMax", resolution.wake_distance)
        field.add("Min", 3)
        field.setNumbers(3, "FieldsList", [2])
        field.setAsBackgroundMesh(3)

        gmsh.model.mesh.generate(3)
        stats = _mesh_stats(gmsh)
        gmsh.write(str(path))
        return stats
    finally:
        gmsh.finalize()


def _classify_boundary_surfaces(gmsh_module) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"walls": [], "inlet": [], "outlet": [], "sphere": []}
    tol = 1.0e-5
    for dim, tag in gmsh_module.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh_module.model.getBoundingBox(dim, tag)
        if abs(xmin) < tol and abs(xmax) < tol:
            groups["inlet"].append(tag)
        elif abs(xmin - CHANNEL_LENGTH) < tol and abs(xmax - CHANNEL_LENGTH) < tol:
            groups["outlet"].append(tag)
        elif (
            abs(ymin + CHANNEL_HALF_WIDTH) < tol
            or abs(ymax - CHANNEL_HALF_WIDTH) < tol
            or abs(zmin + CHANNEL_HALF_WIDTH) < tol
            or abs(zmax - CHANNEL_HALF_WIDTH) < tol
        ):
            groups["walls"].append(tag)
        else:
            groups["sphere"].append(tag)
    return groups


def _mesh_stats(gmsh_module) -> dict[str, int]:
    node_tags, _, _ = gmsh_module.model.mesh.getNodes()
    element_types, _, element_node_tags = gmsh_module.model.mesh.getElements(3)
    tetrahedra = 0
    for element_type, nodes in zip(element_types, element_node_tags, strict=True):
        name, _dim, _order, num_nodes, _local_coords, _num_primary = gmsh_module.model.mesh.getElementProperties(element_type)
        if name == "Tetrahedron 4":
            tetrahedra += len(nodes) // num_nodes
    return {"points": len(node_tags), "tetrahedra": tetrahedra}


if __name__ == "__main__":
    main()
