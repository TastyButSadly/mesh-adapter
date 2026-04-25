from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diff_mesh_adapter.geometry import signed_polygon_areas
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


def generate_unit_square_gmsh(
    path: str | Path,
    *,
    boundary_size: float = 0.22,
    refined_size: float = 0.055,
    refined_point: tuple[float, float] = (0.28, 0.52),
) -> Path:
    """Generate a unit-square 2D Gmsh mesh with one embedded refinement point."""
    import gmsh

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("unit_square")

        geo = gmsh.model.geo
        p0 = geo.addPoint(0.0, 0.0, 0.0, boundary_size)
        p1 = geo.addPoint(1.0, 0.0, 0.0, boundary_size)
        p2 = geo.addPoint(1.0, 1.0, 0.0, boundary_size)
        p3 = geo.addPoint(0.0, 1.0, 0.0, boundary_size)
        pc = geo.addPoint(refined_point[0], refined_point[1], 0.0, refined_size)

        l0 = geo.addLine(p0, p1)
        l1 = geo.addLine(p1, p2)
        l2 = geo.addLine(p2, p3)
        l3 = geo.addLine(p3, p0)
        loop = geo.addCurveLoop([l0, l1, l2, l3])
        surface = geo.addPlaneSurface([loop])
        geo.synchronize()

        gmsh.model.mesh.embed(0, [pc], 2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()

    return output


def generate_nonconvex_hole_gmsh(
    path: str | Path,
    *,
    boundary_size: float = 0.12,
    refined_size: float = 0.035,
    hole_size: float = 0.045,
) -> Path:
    """Generate an L-shaped nonconvex 2D mesh with a circular hole."""
    import gmsh

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("nonconvex_hole")

        geo = gmsh.model.geo
        outer_coords = [
            (0.0, 0.0),
            (1.4, 0.0),
            (1.4, 1.1),
            (0.85, 1.1),
            (0.85, 0.45),
            (0.0, 0.45),
        ]
        outer_points = [geo.addPoint(x, y, 0.0, boundary_size) for x, y in outer_coords]
        outer_lines = [
            geo.addLine(outer_points[i], outer_points[(i + 1) % len(outer_points)])
            for i in range(len(outer_points))
        ]
        outer_loop = geo.addCurveLoop(outer_lines)

        cx, cy, radius = 1.07, 0.25, 0.13
        center = geo.addPoint(cx, cy, 0.0, hole_size)
        hole_points = [
            geo.addPoint(cx + radius, cy, 0.0, hole_size),
            geo.addPoint(cx, cy + radius, 0.0, hole_size),
            geo.addPoint(cx - radius, cy, 0.0, hole_size),
            geo.addPoint(cx, cy - radius, 0.0, hole_size),
        ]
        arcs = [
            geo.addCircleArc(hole_points[i], center, hole_points[(i + 1) % len(hole_points)])
            for i in range(len(hole_points))
        ]
        hole_loop = geo.addCurveLoop(arcs)
        surface = geo.addPlaneSurface([outer_loop, hole_loop])

        refined_points = [
            geo.addPoint(0.62, 0.34, 0.0, refined_size),
            geo.addPoint(0.92, 0.78, 0.0, refined_size * 1.2),
        ]
        geo.synchronize()

        gmsh.model.mesh.embed(0, refined_points, 2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()

    return output


def read_gmsh_mesh(path: str | Path, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> MeshState:
    import meshio

    mesh = meshio.read(path)
    points_np = np.asarray(mesh.points[:, :2], dtype=np.float64)
    points = torch.tensor(points_np, dtype=dtype, device=device)

    triangle_blocks = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangle_blocks.append(torch.tensor(cell_block.data, dtype=torch.long, device=points.device))

    if not triangle_blocks:
        raise ValueError(f"No triangle cells found in {path}")

    oriented_blocks = tuple(_orient_block_positive(points, block) for block in triangle_blocks)
    boundary_nodes = _boundary_nodes_from_cell_adjacency(points.shape[0], oriented_blocks, points.device)
    return MeshState(points=points, cell_blocks=oriented_blocks, boundary_nodes=boundary_nodes)


def _orient_block_positive(points: Tensor, block: Tensor) -> Tensor:
    oriented = block.clone()
    signed = signed_polygon_areas(points, oriented)
    negative = signed < 0.0
    if bool(negative.any()):
        oriented[negative] = torch.flip(oriented[negative], dims=[1])
    return oriented


def _boundary_nodes_from_cell_adjacency(
    num_points: int,
    cell_blocks: tuple[Tensor, ...],
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    edge_count: dict[tuple[int, int], int] = {}
    for block in cell_blocks:
        block_cpu = block.detach().cpu()
        for cell in block_cpu.tolist():
            for i, a in enumerate(cell):
                b = cell[(i + 1) % len(cell)]
                key = tuple(sorted((int(a), int(b))))
                edge_count[key] = edge_count.get(key, 0) + 1

    boundary_indices = sorted({idx for edge, count in edge_count.items() if count == 1 for idx in edge})
    if boundary_indices:
        mask[torch.tensor(boundary_indices, dtype=torch.long, device=device)] = True
    return mask
