import torch

from diff_mesh_adapter import AdaptationConfig, MeshState, adapt_cell_area_equalization
from diff_mesh_adapter.ale import (
    geometric_conservation_residual,
    update_constant_cell_average_with_mesh_flux,
    update_constant_cell_average_without_mesh_flux,
)


def test_swept_mesh_flux_satisfies_geometric_conservation_for_adapted_mesh():
    mesh = _four_triangle_mesh()
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=120, lr=3e-2, movement_weight=0.0, shape_weight=0.0),
    )

    residual = geometric_conservation_residual(mesh.points, result.mesh.points, mesh.cell_blocks)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)


def test_mesh_flux_preserves_constant_cell_average_under_adaptation():
    mesh = _four_triangle_mesh()
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=120, lr=3e-2, movement_weight=0.0, shape_weight=0.0),
    )
    initial_values = torch.full((mesh.num_cells,), 2.5, dtype=mesh.points.dtype)

    corrected = update_constant_cell_average_with_mesh_flux(
        initial_values,
        mesh.points,
        result.mesh.points,
        mesh.cell_blocks,
    )
    naive = update_constant_cell_average_without_mesh_flux(
        initial_values,
        mesh.points,
        result.mesh.points,
        mesh.cell_blocks,
    )

    assert torch.allclose(corrected, initial_values, atol=1e-12)
    assert torch.max(torch.abs(naive - initial_values)) > 0.1


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
