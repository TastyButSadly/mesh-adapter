from __future__ import annotations

from functools import partial

import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_edge_lengths,
    cell_signed_areas,
    min_triangle_angle_degrees,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.monitors import moving_gaussian_ridge_monitor
from diff_mesh_adapter.visualization import save_transient_adaptation_gif
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def test_transient_moving_gaussian_ridge_adapts_each_physical_step(tmp_path):
    mesh_path = generate_unit_square_gmsh(tmp_path / "moving_ridge.msh", boundary_size=0.08, refined_size=0.08)
    initial_mesh = read_gmsh_mesh(mesh_path)
    current_mesh = initial_mesh
    times = [0.0, 0.2, 0.4, 0.6]
    config = AdaptationConfig(
        steps=25,
        lr=8e-4,
        movement_weight=6e-3,
        smoothness_weight=8e-2,
        shape_weight=4e-2,
        quality_barrier_weight=8.0,
        boundary_quality_barrier_weight=30.0,
        min_cell_quality=0.58,
        min_step_cell_quality=0.58,
        edge_length_weight=0.1,
        edge_length_barrier_weight=1.0,
        boundary_edge_length_barrier_weight=10.0,
        max_edge_stretch=1.8,
        min_edge_compression=0.4,
        max_step_edge_stretch=2.3,
        min_step_edge_compression=0.35,
        barrier_weight=1.0,
        grad_clip=0.35,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-3,
        early_stopping_relative=False,
    )

    adapted_meshes = []
    monitor_fns = []
    final_losses = []
    high_monitor_area_ratios = []
    min_angles = []
    reference_edge_lengths = cell_edge_lengths(initial_mesh.points, initial_mesh.cell_blocks).clamp_min(config.area_eps)

    for time in times:
        monitor_fn = partial(
            moving_gaussian_ridge_monitor,
            time=time,
            offset0=0.08,
            speed=0.88,
            normal=(1.0, 0.20),
            width=0.05,
            alpha=20.0,
        )
        threshold = torch.quantile(monitor_fn(current_mesh.points, current_mesh.cell_blocks).detach(), 0.90)
        before_high_area = _mean_area_above_monitor(current_mesh, monitor_fn, threshold)

        result = adapt_monitor_weighted_area(current_mesh, monitor_fn, config, reference_points=initial_mesh.points)
        current_mesh = result.mesh
        after_high_area = _mean_area_above_monitor(current_mesh, monitor_fn, threshold)

        adapted_meshes.append(current_mesh)
        monitor_fns.append(monitor_fn)
        final_losses.append(result.final_loss)
        high_monitor_area_ratios.append(float(after_high_area / before_high_area))

        assert result.steps_completed <= config.steps
        assert torch.all(cell_signed_areas(current_mesh.points, current_mesh.cell_blocks) > 0.0)
        min_angles.append(float(min_triangle_angle_degrees(current_mesh.points, current_mesh.cell_blocks)))
        edge_ratios = cell_edge_lengths(current_mesh.points, current_mesh.cell_blocks) / reference_edge_lengths
        assert float(edge_ratios.max()) <= config.max_step_edge_stretch + 1e-8

    displacement = torch.linalg.norm(current_mesh.points - initial_mesh.points, dim=1)
    assert float(displacement[initial_mesh.boundary_nodes].max()) == 0.0
    assert min(min_angles) > 20.0
    assert sum(high_monitor_area_ratios) / len(high_monitor_area_ratios) < 0.85

    gif_path = save_transient_adaptation_gif(
        adapted_meshes,
        monitor_fns,
        final_losses,
        tmp_path / "moving_ridge.gif",
        times=times,
        fps=3,
    )
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def _mean_area_above_monitor(mesh, monitor_fn, threshold: torch.Tensor) -> torch.Tensor:
    monitor = monitor_fn(mesh.points, mesh.cell_blocks).detach()
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[monitor >= threshold].mean()
