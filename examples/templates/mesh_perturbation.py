from __future__ import annotations

import torch

from diff_mesh_adapter.geometry import cell_signed_measures
from diff_mesh_adapter.mesh import MeshState


def perturb_internal_nodes(
    mesh: MeshState,
    *,
    amplitude: float = 0.08,
    seed: int = 7,
    min_area: float = 1e-8,
    max_attempts: int = 8,
) -> MeshState:
    """Return a valid but degraded 2D mesh by randomly moving non-boundary nodes."""
    generator = torch.Generator(device=mesh.points.device).manual_seed(seed)
    internal = ~mesh.boundary_nodes
    base_points = mesh.points.detach()
    initial_signed = cell_signed_measures(base_points, mesh.cell_blocks).detach()
    orientation = torch.sign(initial_signed)

    scale = amplitude
    for _ in range(max_attempts):
        noise = torch.zeros_like(base_points)
        noise[internal] = scale * torch.randn(
            (int(internal.sum()), mesh.dim),
            dtype=base_points.dtype,
            device=base_points.device,
            generator=generator,
        )
        candidate = base_points + noise
        signed = cell_signed_measures(candidate, mesh.cell_blocks) * orientation
        if bool(torch.all(signed > min_area)):
            return MeshState(points=candidate, cell_blocks=mesh.cell_blocks, boundary_nodes=mesh.boundary_nodes)
        scale *= 0.5

    return MeshState(points=base_points.clone(), cell_blocks=mesh.cell_blocks, boundary_nodes=mesh.boundary_nodes)


def perturb_internal_tet_nodes(
    mesh: MeshState,
    *,
    amplitude: float = 0.16,
    seed: int = 41,
    min_volume: float = 1e-9,
    max_attempts: int = 10,
) -> MeshState:
    """Return a valid but degraded tetrahedral mesh by randomly moving internal nodes."""
    generator = torch.Generator(device=mesh.points.device).manual_seed(seed)
    internal = ~mesh.boundary_nodes
    base_points = mesh.points.detach()
    initial_signed = cell_signed_measures(base_points, mesh.cell_blocks).detach()
    orientation = torch.sign(initial_signed)

    if not bool(internal.any()):
        raise RuntimeError("mesh has no internal nodes to perturb")

    scale = amplitude
    for _ in range(max_attempts):
        noise = torch.zeros_like(base_points)
        noise[internal] = scale * torch.randn(
            (int(internal.sum()), mesh.dim),
            dtype=base_points.dtype,
            device=base_points.device,
            generator=generator,
        )
        candidate = base_points + noise
        signed = cell_signed_measures(candidate, mesh.cell_blocks) * orientation
        if bool(torch.all(signed > min_volume)):
            return MeshState(points=candidate, cell_blocks=mesh.cell_blocks, boundary_nodes=mesh.boundary_nodes)
        scale *= 0.5

    raise RuntimeError("Could not perturb internal tetra nodes without inverting cells")
