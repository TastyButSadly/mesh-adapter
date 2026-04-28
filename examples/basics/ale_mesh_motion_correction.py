from __future__ import annotations

import torch

from diff_mesh_adapter import AdaptationConfig, MeshState, adapt_cell_area_equalization
from diff_mesh_adapter.ale import (
    geometric_conservation_residual,
    update_constant_cell_average_with_mesh_flux,
    update_constant_cell_average_without_mesh_flux,
)


def main() -> None:
    mesh = _four_triangle_mesh()
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=120, lr=3e-2, movement_weight=0.0, shape_weight=0.0),
    )
    constant_state = torch.full((mesh.num_cells,), 2.5, dtype=mesh.points.dtype)

    corrected = update_constant_cell_average_with_mesh_flux(
        constant_state,
        mesh.points,
        result.mesh.points,
        mesh.cell_blocks,
    )
    naive = update_constant_cell_average_without_mesh_flux(
        constant_state,
        mesh.points,
        result.mesh.points,
        mesh.cell_blocks,
    )
    residual = geometric_conservation_residual(mesh.points, result.mesh.points, mesh.cell_blocks)

    print(f"geometric conservation residual max: {float(residual.abs().max()):.3e}")
    print(f"constant-state error with mesh flux: {float((corrected - constant_state).abs().max()):.3e}")
    print(f"constant-state error without mesh flux: {float((naive - constant_state).abs().max()):.3e}")


def _four_triangle_mesh() -> MeshState:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.22, 0.37],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=torch.long,
    )
    boundary_nodes = torch.tensor([True, True, True, True, False])
    return MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)


if __name__ == "__main__":
    main()
