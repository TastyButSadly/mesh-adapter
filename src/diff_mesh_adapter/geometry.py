from __future__ import annotations

import torch

Tensor = torch.Tensor


def signed_polygon_areas(points: Tensor, cells: Tensor) -> Tensor:
    """Signed areas for one ordered 2D polygon connectivity block."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("signed_polygon_areas expects points with shape (n_points, 2)")
    if cells.ndim != 2 or cells.shape[1] < 3:
        raise ValueError("cells must have shape (n_cells, nodes_per_cell >= 3)")

    vertices = points[cells]
    x = vertices[..., 0]
    y = vertices[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    return 0.5 * torch.sum(x * y_next - y * x_next, dim=1)


def cell_signed_areas(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return torch.cat([signed_polygon_areas(points, block) for block in cell_blocks], dim=0)


def cell_abs_areas(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return cell_signed_areas(points, cell_blocks).abs()


def polygon_centroids(points: Tensor, cells: Tensor, eps: float = 1e-12) -> Tensor:
    """Area-weighted centroids for one ordered 2D polygon connectivity block."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("polygon_centroids expects points with shape (n_points, 2)")
    vertices = points[cells]
    x = vertices[..., 0]
    y = vertices[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    cross = x * y_next - y * x_next
    area2 = cross.sum(dim=1)
    denom = area2 + eps * torch.where(area2 >= 0.0, torch.ones_like(area2), -torch.ones_like(area2))
    centroid = torch.stack(
        [
            ((x + x_next) * cross).sum(dim=1),
            ((y + y_next) * cross).sum(dim=1),
        ],
        dim=1,
    ) / (3.0 * denom.unsqueeze(1))
    fallback = vertices.mean(dim=1)
    return torch.where(area2.abs().unsqueeze(1) > eps, centroid, fallback)


def cell_centroids(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return torch.cat([polygon_centroids(points, block) for block in cell_blocks], dim=0)


def polygon_edge_lengths(points: Tensor, cells: Tensor) -> Tensor:
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, dim)")
    vertices = points[cells]
    next_vertices = torch.roll(vertices, shifts=-1, dims=1)
    return torch.linalg.norm(next_vertices - vertices, dim=2)


def normalized_edge_length_variance(points: Tensor, cell_blocks: tuple[Tensor, ...], eps: float = 1e-12) -> Tensor:
    """Mean per-cell variance of edge lengths normalized by each cell's mean edge length."""
    losses = []
    for block in cell_blocks:
        lengths = polygon_edge_lengths(points, block)
        mean = lengths.mean(dim=1, keepdim=True).clamp_min(eps)
        losses.append(((lengths / mean - 1.0) ** 2).mean(dim=1))
    return torch.cat(losses, dim=0).mean()


def triangle_angles(points: Tensor, triangles: Tensor, eps: float = 1e-12) -> Tensor:
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangle_angles expects triangles with shape (n_cells, 3)")

    vertices = points[triangles]
    a = torch.linalg.norm(vertices[:, 1] - vertices[:, 2], dim=1).clamp_min(eps)
    b = torch.linalg.norm(vertices[:, 2] - vertices[:, 0], dim=1).clamp_min(eps)
    c = torch.linalg.norm(vertices[:, 0] - vertices[:, 1], dim=1).clamp_min(eps)

    angle0 = torch.acos(((b.square() + c.square() - a.square()) / (2.0 * b * c)).clamp(-1.0, 1.0))
    angle1 = torch.acos(((a.square() + c.square() - b.square()) / (2.0 * a * c)).clamp(-1.0, 1.0))
    angle2 = torch.pi - angle0 - angle1
    return torch.stack([angle0, angle1, angle2], dim=1)


def min_triangle_angle_degrees(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    angles = []
    for block in cell_blocks:
        if block.shape[1] == 3:
            angles.append(triangle_angles(points, block).reshape(-1))
    if not angles:
        raise ValueError("min_triangle_angle_degrees requires at least one triangle block")
    return torch.rad2deg(torch.cat(angles).min())
