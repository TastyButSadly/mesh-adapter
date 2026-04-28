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


def triangle_signed_areas_and_shape_energy(points: Tensor, cells: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Signed areas and scale-invariant equilateral-triangle energy for one triangle block."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("triangle_signed_areas_and_shape_energy expects points with shape (n_points, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("triangle_signed_areas_and_shape_energy expects cells with shape (n_cells, 3)")

    vertices = points[cells]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    signed_areas = 0.5 * (edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])

    physical = torch.stack([edge1, edge2], dim=2)
    jacobian = physical @ _equilateral_triangle_inverse(points.dtype, points.device)
    det = (signed_areas * _equilateral_triangle_inverse_det(points.dtype, points.device) * 2.0).abs().clamp_min(eps)
    frobenius_sq = jacobian.square().sum(dim=(1, 2))
    shape_energy = frobenius_sq / (2.0 * det) - 1.0
    return signed_areas, shape_energy


def triangle_shape_energy(points: Tensor, cells: Tensor, eps: float = 1e-12) -> Tensor:
    return triangle_signed_areas_and_shape_energy(points, cells, eps)[1]


def tetra_signed_volumes(points: Tensor, cells: Tensor) -> Tensor:
    """Signed volumes for one 3D tetrahedral connectivity block."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tetra_signed_volumes expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tetra_signed_volumes expects cells with shape (n_cells, 4)")

    vertices = points[cells]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    edge3 = vertices[:, 3] - vertices[:, 0]
    return (torch.cross(edge1, edge2, dim=1) * edge3).sum(dim=1) / 6.0


def tet_signed_volumes(points: Tensor, cells: Tensor) -> Tensor:
    return tetra_signed_volumes(points, cells)


def tet_signed_volumes_and_shape_energy(points: Tensor, cells: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Signed volumes and scale-invariant shape energy for one tetrahedral block."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tet_signed_volumes_and_shape_energy expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tet_signed_volumes_and_shape_energy expects cells with shape (n_cells, 4)")

    vertices = points[cells]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    edge3 = vertices[:, 3] - vertices[:, 0]
    signed_volumes = (torch.cross(edge1, edge2, dim=1) * edge3).sum(dim=1) / 6.0

    physical = torch.stack([edge1, edge2, edge3], dim=2)
    jacobian = physical @ _regular_tetra_inverse(points.dtype, points.device)
    det = (signed_volumes * _regular_tetra_inverse_det(points.dtype, points.device) * 6.0).abs().clamp_min(eps)
    frobenius_sq = jacobian.square().sum(dim=(1, 2))
    shape_energy = frobenius_sq / (3.0 * det.pow(2.0 / 3.0)) - 1.0
    return signed_volumes, shape_energy


def cell_signed_measures(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    """Signed 2D areas or 3D tetra volumes for all fixed-topology cell blocks."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, dim)")
    if points.shape[1] == 2:
        return torch.cat([signed_polygon_areas(points, block) for block in cell_blocks], dim=0)
    if points.shape[1] == 3:
        volumes = []
        for block in cell_blocks:
            if block.shape[1] != 4:
                raise NotImplementedError("V1 3D geometry supports tetrahedral cell blocks only")
            volumes.append(tetra_signed_volumes(points, block))
        return torch.cat(volumes, dim=0)
    raise NotImplementedError("Only 2D polygon and 3D tetrahedral meshes are supported")


def cell_abs_measures(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return cell_signed_measures(points, cell_blocks).abs()


def cell_signed_measures_and_shape_energy(
    points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Signed measures and per-cell shape energy in one geometry pass per block."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, dim)")
    if points.shape[1] == 2:
        signed = []
        energy = []
        for block in cell_blocks:
            if block.shape[1] == 3:
                block_signed, block_energy = triangle_signed_areas_and_shape_energy(points, block, eps)
                signed.append(block_signed)
                energy.append(block_energy)
            else:
                signed.append(signed_polygon_areas(points, block))
                energy.append(_normalized_polygon_edge_length_variance(points, block, eps))
        return torch.cat(signed, dim=0), torch.cat(energy, dim=0)
    if points.shape[1] == 3:
        signed = []
        energy = []
        for block in cell_blocks:
            if block.shape[1] != 4:
                raise NotImplementedError("V1 3D geometry supports tetrahedral cell blocks only")
            block_signed, block_energy = tet_signed_volumes_and_shape_energy(points, block, eps)
            signed.append(block_signed)
            energy.append(block_energy)
        return torch.cat(signed, dim=0), torch.cat(energy, dim=0)
    raise NotImplementedError("Only 2D polygon and 3D tetrahedral meshes are supported")


def cell_signed_areas(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return cell_signed_measures(points, cell_blocks)


def cell_abs_areas(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    return cell_abs_measures(points, cell_blocks)


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


def tetra_centroids(points: Tensor, cells: Tensor) -> Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tetra_centroids expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tetra_centroids expects cells with shape (n_cells, 4)")
    return points[cells].mean(dim=1)


def cell_centroids(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    if points.shape[1] == 2:
        return torch.cat([polygon_centroids(points, block) for block in cell_blocks], dim=0)
    if points.shape[1] == 3:
        centroids = []
        for block in cell_blocks:
            if block.shape[1] != 4:
                raise NotImplementedError("V1 3D centroids support tetrahedral cell blocks only")
            centroids.append(tetra_centroids(points, block))
        return torch.cat(centroids, dim=0)
    raise NotImplementedError("Only 2D polygon and 3D tetrahedral meshes are supported")


def polygon_edge_lengths(points: Tensor, cells: Tensor) -> Tensor:
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, dim)")
    vertices = points[cells]
    next_vertices = torch.roll(vertices, shifts=-1, dims=1)
    return torch.linalg.norm(next_vertices - vertices, dim=2)


def _normalized_polygon_edge_length_variance(points: Tensor, cells: Tensor, eps: float = 1e-12) -> Tensor:
    lengths = polygon_edge_lengths(points, cells)
    mean = lengths.mean(dim=1, keepdim=True).clamp_min(eps)
    return ((lengths / mean - 1.0) ** 2).mean(dim=1)


def tetra_edge_lengths(points: Tensor, cells: Tensor) -> Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tetra_edge_lengths expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tetra_edge_lengths expects cells with shape (n_cells, 4)")
    pairs = torch.tensor(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=torch.long,
        device=cells.device,
    )
    vertices = points[cells]
    return torch.linalg.norm(vertices[:, pairs[:, 0]] - vertices[:, pairs[:, 1]], dim=2)


def _equilateral_triangle_inverse(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(
        [
            [1.0, -0.5773502691896258],
            [0.0, 1.1547005383792517],
        ],
        dtype=dtype,
        device=device,
    )


def _equilateral_triangle_inverse_det(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(1.1547005383792517, dtype=dtype, device=device)


def _regular_tetra_inverse(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(
        [
            [1.0, -0.5773502691896258, -0.408248290463863],
            [0.0, 1.1547005383792517, -0.408248290463863],
            [0.0, 0.0, 1.224744871391589],
        ],
        dtype=dtype,
        device=device,
    )


def _regular_tetra_inverse_det(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(1.4142135623730951, dtype=dtype, device=device)


def tet_shape_energy(points: Tensor, cells: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale-invariant regular-tetrahedron energy for each tetrahedral cell."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tet_shape_energy expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tet_shape_energy expects cells with shape (n_cells, 4)")

    return tet_signed_volumes_and_shape_energy(points, cells, eps)[1]


def cell_shape_energy(points: Tensor, cell_blocks: tuple[Tensor, ...], eps: float = 1e-12) -> Tensor:
    if points.shape[1] == 2:
        energies = []
        for block in cell_blocks:
            if block.shape[1] == 3:
                energies.append(triangle_shape_energy(points, block, eps))
            else:
                energies.append(_normalized_polygon_edge_length_variance(points, block, eps))
        return torch.cat(energies, dim=0)
    if points.shape[1] == 3:
        return cell_signed_measures_and_shape_energy(points, cell_blocks, eps)[1]
    raise NotImplementedError("Only 2D polygon and 3D tetrahedral meshes are supported")


def cell_edge_lengths(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    edge_lengths = []
    if points.shape[1] == 2:
        for block in cell_blocks:
            edge_lengths.append(polygon_edge_lengths(points, block).reshape(-1))
        return torch.cat(edge_lengths, dim=0)
    if points.shape[1] == 3:
        for block in cell_blocks:
            if block.shape[1] != 4:
                raise NotImplementedError("V1 3D edge lengths support tetrahedral cell blocks only")
            edge_lengths.append(tetra_edge_lengths(points, block).reshape(-1))
        return torch.cat(edge_lengths, dim=0)
    raise NotImplementedError("Only 2D polygon and 3D tetrahedral meshes are supported")


def normalized_edge_length_variance(points: Tensor, cell_blocks: tuple[Tensor, ...], eps: float = 1e-12) -> Tensor:
    """Mean 2D edge variance or 3D tetra shape energy over all cells."""
    return cell_shape_energy(points, cell_blocks, eps).mean()


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


def tet_dihedral_angles(points: Tensor, cells: Tensor, eps: float = 1e-12) -> Tensor:
    """Interior dihedral angles in radians for each tetrahedral cell."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tet_dihedral_angles expects points with shape (n_points, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tet_dihedral_angles expects cells with shape (n_cells, 4)")

    vertices = points[cells]
    edge_specs = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1))
    angles = []
    for i, j, k, l in edge_specs:
        edge = vertices[:, j] - vertices[:, i]
        normal_a = torch.cross(edge, vertices[:, k] - vertices[:, i], dim=1)
        normal_b = torch.cross(edge, vertices[:, l] - vertices[:, i], dim=1)
        denom = (torch.linalg.norm(normal_a, dim=1) * torch.linalg.norm(normal_b, dim=1)).clamp_min(eps)
        cosine = (normal_a * normal_b).sum(dim=1) / denom
        angles.append(torch.acos(cosine.clamp(-1.0, 1.0)))
    return torch.stack(angles, dim=1)


def tet_min_dihedral_angles_degrees(points: Tensor, cells: Tensor) -> Tensor:
    return torch.rad2deg(tet_dihedral_angles(points, cells).min(dim=1).values)


def tet_max_dihedral_angles_degrees(points: Tensor, cells: Tensor) -> Tensor:
    return torch.rad2deg(tet_dihedral_angles(points, cells).max(dim=1).values)


def min_tet_dihedral_angle_degrees(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    angles = []
    for block in cell_blocks:
        if block.shape[1] == 4:
            angles.append(tet_dihedral_angles(points, block).reshape(-1))
    if not angles:
        raise ValueError("min_tet_dihedral_angle_degrees requires at least one tetra block")
    return torch.rad2deg(torch.cat(angles).min())


def max_tet_dihedral_angle_degrees(points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    angles = []
    for block in cell_blocks:
        if block.shape[1] == 4:
            angles.append(tet_dihedral_angles(points, block).reshape(-1))
    if not angles:
        raise ValueError("max_tet_dihedral_angle_degrees requires at least one tetra block")
    return torch.rad2deg(torch.cat(angles).max())
