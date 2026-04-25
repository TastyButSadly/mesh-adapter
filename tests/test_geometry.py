import torch

from diff_mesh_adapter.geometry import cell_signed_areas


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
