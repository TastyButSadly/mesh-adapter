from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, adapt_monitor_weighted_area, cell_abs_areas, cell_centroids, gaussian_cell_monitor
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_artifacts
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def _near_feature_area(mesh, center=(0.28, 0.52), radius=0.18) -> torch.Tensor:
    centroids = cell_centroids(mesh.points, mesh.cell_blocks)
    center_tensor = torch.tensor(center, dtype=mesh.points.dtype, device=mesh.points.device)
    near = torch.linalg.norm(centroids - center_tensor, dim=1) < radius
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[near].mean()


def main() -> None:
    output_dir = Path("outputs/gaussian_monitor")
    mesh_path = generate_unit_square_gmsh(output_dir / "unit_square_monitor.msh", boundary_size=0.15, refined_size=0.15)
    mesh = read_gmsh_mesh(mesh_path)

    monitor_fn = partial(gaussian_cell_monitor, center=(0.28, 0.52), sigma=0.16, alpha=18.0)
    initial_near_area = _near_feature_area(mesh)
    result = adapt_monitor_weighted_area(
        mesh,
        monitor_fn,
        AdaptationConfig(
            steps=180,
            lr=1.2e-2,
            movement_weight=3e-4,
            shape_weight=1e-4,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            early_stopping_patience=20,
            early_stopping_min_delta=1e-3,
        ),
    )
    final_near_area = _near_feature_area(result.mesh)

    paths = save_adaptation_artifacts(
        mesh,
        result,
        output_dir,
        prefix="gaussian_monitor",
        title="Gaussian monitor weighted-area adaptation",
        fps=14,
    )

    print(f"mesh: {mesh.num_points} points, {mesh.num_cells} cells")
    print(f"near-feature mean area: {float(initial_near_area):.6g} -> {float(final_near_area):.6g}")
    print(f"optimizer loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    print(f"steps completed: {result.steps_completed} (early_stopped={result.early_stopped})")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
