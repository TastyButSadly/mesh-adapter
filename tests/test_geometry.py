import torch

from diff_mesh_adapter.geometry import (
    cell_signed_areas,
    cell_signed_measures_and_shape_energy,
    triangle_shape_energy,
)


def test_triangle_and_quad_area_are_correct():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    quads = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    areas = cell_signed_areas(points, (triangles, quads))

    assert torch.allclose(areas, torch.tensor([0.5, 1.0], dtype=torch.float64))


def test_area_gradients_flow_to_points():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 0.8],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.long)

    loss = cell_signed_areas(points, (cells,)).square().sum()
    loss.backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert float(points.grad.abs().sum()) > 0.0


def test_combined_measure_shape_helper_matches_area_kernel():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)

    measures, shape_energy = cell_signed_measures_and_shape_energy(points, (cells,))

    assert torch.allclose(measures, cell_signed_areas(points, (cells,)))
    assert shape_energy.shape == measures.shape


def test_triangle_shape_energy_penalizes_skinny_triangles():
    regular_points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8660254037844386],
        ],
        dtype=torch.float64,
    )
    skinny_points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.05, 0.05],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.long)

    assert triangle_shape_energy(regular_points, cells).mean() < 1e-12
    assert triangle_shape_energy(skinny_points, cells).mean() > 5.0
