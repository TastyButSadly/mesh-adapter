from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diff_mesh_adapter.geometry import signed_polygon_areas
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


def read_gmsh_mesh(path: str | Path, *, dtype: torch.dtype = torch.float64,
                   device: torch.device | str | None = None) -> MeshState:
    import meshio

    mesh = meshio.read(path)

    tetra_blocks = []
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tetra_blocks.append(torch.tensor(cell_block.data, dtype=torch.long, device=device))

    if tetra_blocks:
        points_np = np.asarray(mesh.points[:, :3], dtype=np.float64)
        points = torch.tensor(points_np, dtype=dtype, device=device)
        oriented_blocks = tuple(
            _orient_tetra_block_positive(points, block.to(device=points.device)) for block in tetra_blocks
        )
        boundary_nodes = _boundary_nodes_from_tetra_faces(points.shape[0], oriented_blocks, points.device)
        return MeshState(points=points, cell_blocks=oriented_blocks, boundary_nodes=boundary_nodes)

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


def _orient_tetra_block_positive(points: Tensor, block: Tensor) -> Tensor:
    oriented = block.clone()
    vertices = points[oriented]
    matrices = torch.stack(
        [
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ],
        dim=2,
    )
    negative = torch.linalg.det(matrices) < 0.0
    if bool(negative.any()):
        oriented[negative] = oriented[negative][:, [1, 0, 2, 3]]
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


def _boundary_nodes_from_tetra_faces(
        num_points: int,
        cell_blocks: tuple[Tensor, ...],
        device: torch.device,
) -> Tensor:
    mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    face_count: dict[tuple[int, int, int], int] = {}
    for block in cell_blocks:
        block_cpu = block.detach().cpu()
        for cell in block_cpu.tolist():
            faces = (
                (cell[0], cell[1], cell[2]),
                (cell[0], cell[1], cell[3]),
                (cell[0], cell[2], cell[3]),
                (cell[1], cell[2], cell[3]),
            )
            for face in faces:
                key = tuple(sorted(int(idx) for idx in face))
                face_count[key] = face_count.get(key, 0) + 1

    boundary_indices = sorted({idx for face, count in face_count.items() if count == 1 for idx in face})
    if boundary_indices:
        mask[torch.tensor(boundary_indices, dtype=torch.long, device=device)] = True
    return mask
