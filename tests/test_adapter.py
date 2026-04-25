import torch

from diff_mesh_adapter import AdaptationConfig, MeshState, adapt_cell_area_equalization, cell_abs_areas, cell_signed_areas


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


def test_adapter_equalizes_synthetic_triangle_areas_and_keeps_boundary_fixed():
    mesh = _four_triangle_mesh()
    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    initial_cells = mesh.cell_blocks[0].clone()

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=200, lr=3e-2, movement_weight=0.0, shape_weight=0.0),
    )

    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)

    assert torch.equal(result.mesh.cell_blocks[0], initial_cells)
    assert torch.allclose(result.mesh.points[mesh.boundary_nodes], mesh.points[mesh.boundary_nodes])
    assert torch.all(final_signed > 0.0)
    assert final_areas.std(unbiased=False) < initial_areas.std(unbiased=False)
    assert result.final_loss < result.initial_loss


def test_adapter_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        return

    mesh = _four_triangle_mesh().to(device="cuda")
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=10, lr=1e-2, movement_weight=0.0, shape_weight=0.0),
    )

    assert result.mesh.points.is_cuda
    assert result.min_area_history[-1] > 0.0


def test_adapter_preserves_clockwise_connectivity_and_uses_relative_orientation():
    mesh = _four_triangle_mesh()
    clockwise_cells = torch.flip(mesh.cell_blocks[0], dims=[1])
    clockwise_mesh = MeshState(points=mesh.points, cell_blocks=(clockwise_cells,), boundary_nodes=mesh.boundary_nodes)

    result = adapt_cell_area_equalization(
        clockwise_mesh,
        AdaptationConfig(steps=20, lr=1e-2, movement_weight=0.0, shape_weight=0.0),
    )

    assert torch.equal(result.mesh.cell_blocks[0], clockwise_cells)
    assert result.min_area_history[-1] > 0.0


def test_adapter_stops_after_patience_without_significant_improvement():
    mesh = _four_triangle_mesh()

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=100,
            lr=1e-2,
            movement_weight=0.0,
            shape_weight=0.0,
            early_stopping_patience=3,
            early_stopping_min_delta=0.99,
        ),
    )

    assert result.early_stopped
    assert result.steps_completed < 100
    assert len(result.loss_history) == result.steps_completed + 1
