from __future__ import annotations

import torch

from diff_mesh_adapter.geometry import cell_signed_measures

Tensor = torch.Tensor


def cell_swept_measure_fluxes(
        old_points: Tensor,
        new_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Integrated signed mesh fluxes for ordered 2D cells moving linearly in time."""
    _validate_point_motion(old_points, new_points)
    if old_points.shape[1] != 2:
        raise NotImplementedError("cell_swept_measure_fluxes currently supports ordered 2D cells only")
    return tuple(_block_swept_measure_fluxes(old_points, new_points, block) for block in cell_blocks)


def geometric_conservation_residual(
        old_points: Tensor,
        new_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
) -> Tensor:
    """Residual of sum(mesh fluxes) = new cell measure - old cell measure."""
    flux_change = torch.cat(
        [fluxes.sum(dim=1) for fluxes in cell_swept_measure_fluxes(old_points, new_points, cell_blocks)])
    old_measures = cell_signed_measures(old_points, cell_blocks)
    new_measures = cell_signed_measures(new_points, cell_blocks)
    return flux_change - (new_measures - old_measures)


def update_constant_cell_average_with_mesh_flux(
        cell_values: Tensor,
        old_points: Tensor,
        new_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        eps: float = 1e-12,
) -> Tensor:
    """Move a piecewise-constant conserved scalar with only mesh-motion flux.

    This is a geometric-conservation diagnostic, not a full PDE solver. It
    corresponds to the zero-physical-flux part of an ALE finite-volume update.
    """
    old_measures, new_measures, measure_change = _cell_measure_motion(old_points, new_points, cell_blocks, eps)
    if cell_values.shape != old_measures.shape:
        raise ValueError("cell_values must have one value per cell")
    old_content = cell_values * old_measures
    new_content = old_content + cell_values * measure_change
    return new_content / new_measures


def update_constant_cell_average_without_mesh_flux(
        cell_values: Tensor,
        old_points: Tensor,
        new_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        *,
        eps: float = 1e-12,
) -> Tensor:
    """Move cell averages by changing volumes but ignoring mesh-motion flux."""
    old_measures, new_measures, _ = _cell_measure_motion(old_points, new_points, cell_blocks, eps)
    if cell_values.shape != old_measures.shape:
        raise ValueError("cell_values must have one value per cell")
    return cell_values * old_measures / new_measures


def _validate_point_motion(old_points: Tensor, new_points: Tensor) -> None:
    if old_points.ndim != 2:
        raise ValueError("old_points must have shape (n_points, dim)")
    if new_points.shape != old_points.shape:
        raise ValueError("new_points must have the same shape as old_points")


def _block_swept_measure_fluxes(old_points: Tensor, new_points: Tensor, cells: Tensor) -> Tensor:
    old_vertices = old_points[cells]
    new_vertices = new_points[cells]
    old_next = torch.roll(old_vertices, shifts=-1, dims=1)
    new_next = torch.roll(new_vertices, shifts=-1, dims=1)
    swept_quads = torch.stack([old_vertices, new_vertices, new_next, old_next], dim=2)
    return _signed_polygon_areas(swept_quads)


def _signed_polygon_areas(vertices: Tensor) -> Tensor:
    x = vertices[..., 0]
    y = vertices[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=-1)
    y_next = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * torch.sum(x * y_next - y * x_next, dim=-1)


def _cell_measure_motion(
        old_points: Tensor,
        new_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    old_signed = cell_signed_measures(old_points, cell_blocks)
    new_signed = cell_signed_measures(new_points, cell_blocks)
    orientation = torch.sign(old_signed)
    if bool((orientation == 0.0).any()):
        raise ValueError("old mesh contains degenerate cells with near-zero signed measure")

    old_measures = old_signed * orientation
    new_measures = new_signed * orientation
    if bool((new_measures <= eps).any()):
        raise ValueError("new mesh contains inverted or near-degenerate cells")

    signed_change = torch.cat(
        [fluxes.sum(dim=1) for fluxes in cell_swept_measure_fluxes(old_points, new_points, cell_blocks)])
    measure_change = signed_change * orientation
    return old_measures, new_measures, measure_change
