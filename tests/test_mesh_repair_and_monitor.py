from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    adapt_mesh_quality,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_centroids,
    cell_signed_areas,
    gaussian_cell_monitor,
    min_triangle_angle_degrees,
    normalized_edge_length_variance,
)
from diff_mesh_adapter.demo import perturb_internal_nodes
from diff_mesh_adapter.io import generate_unit_square_gmsh, read_gmsh_mesh
from tests.helpers import assert_artifacts_exist, save_demo_artifacts


def test_gmsh_mesh_repair_improves_distorted_mesh_and_saves_gif(tmp_path):
    mesh_path = generate_unit_square_gmsh(tmp_path / "repair.msh", boundary_size=0.16, refined_size=0.16)
    clean_mesh = read_gmsh_mesh(mesh_path)
    degraded_mesh = perturb_internal_nodes(clean_mesh, amplitude=0.075, seed=41)

    initial_quality = normalized_edge_length_variance(degraded_mesh.points, degraded_mesh.cell_blocks)
    initial_min_angle = min_triangle_angle_degrees(degraded_mesh.points, degraded_mesh.cell_blocks)

    result = adapt_mesh_quality(
        degraded_mesh,
        AdaptationConfig(
            steps=90,
            lr=1.5e-2,
            movement_weight=2e-3,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            history_stride=2,
        ),
    )

    final_quality = normalized_edge_length_variance(result.mesh.points, result.mesh.cell_blocks)
    final_min_angle = min_triangle_angle_degrees(result.mesh.points, result.mesh.cell_blocks)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)
    paths = save_demo_artifacts(
        degraded_mesh,
        result,
        Path("outputs/mesh_repair"),
        "mesh_repair",
        "Fixed-topology mesh repair",
    )

    assert torch.all(final_signed > 0.0)
    assert final_quality < initial_quality
    assert final_min_angle > initial_min_angle
    assert result.final_loss < result.initial_loss
    assert_artifacts_exist(paths)


def test_gaussian_monitor_adaptation_concentrates_cells_near_feature_and_saves_gif(tmp_path):
    mesh_path = generate_unit_square_gmsh(tmp_path / "monitor.msh", boundary_size=0.15, refined_size=0.15)
    mesh = read_gmsh_mesh(mesh_path)
    monitor_fn = partial(gaussian_cell_monitor, center=(0.28, 0.52), sigma=0.16, alpha=18.0)

    initial_near_area = _near_feature_mean_area(mesh)
    result = adapt_monitor_weighted_area(
        mesh,
        monitor_fn,
        AdaptationConfig(
            steps=90,
            lr=1.2e-2,
            movement_weight=3e-4,
            shape_weight=1e-4,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            history_stride=2,
        ),
    )

    final_near_area = _near_feature_mean_area(result.mesh)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)
    paths = save_demo_artifacts(
        mesh,
        result,
        Path("outputs/gaussian_monitor"),
        "gaussian_monitor",
        "Gaussian monitor weighted-area adaptation",
    )

    assert torch.all(final_signed > 0.0)
    assert final_near_area < initial_near_area
    assert result.final_loss < result.initial_loss
    assert_artifacts_exist(paths)


def _near_feature_mean_area(mesh, center=(0.28, 0.52), radius=0.18):
    centroids = cell_centroids(mesh.points, mesh.cell_blocks)
    center_tensor = torch.tensor(center, dtype=mesh.points.dtype, device=mesh.points.device)
    near = torch.linalg.norm(centroids - center_tensor, dim=1) < radius
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[near].mean()
