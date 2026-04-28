from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, adapt_monitor_weighted_area, cell_abs_areas
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.monitors import (
    advecting_front_gradient_monitor,
    advecting_gaussian_monitor,
    time_integrated_advecting_gaussian_gradient_monitor,
)
from diff_mesh_adapter.visualization import save_adaptation_artifacts, save_adaptation_gif, save_mesh_cell_scalar_comparison
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def main() -> None:
    output_dir = Path("outputs/advecting_monitor_2d")
    mesh_path = generate_unit_square_gmsh(output_dir / "advecting_monitor.msh", boundary_size=0.09, refined_size=0.09)
    mesh = read_gmsh_mesh(mesh_path)

    cases = [
        (
            "time_integrated_advecting_gaussian_gradient_monitor",
            partial(
                time_integrated_advecting_gaussian_gradient_monitor,
                time_start=0.0,
                time_end=0.85,
                samples=8,
                center0=(0.18, 0.30),
                velocity=(0.65, 0.45),
                sigma=0.075,
                alpha=0.18,
            ),
            AdaptationConfig(
                steps=240,
                lr=4e-3,
                movement_weight=2.5e-3,
                smoothness_weight=4e-2,
                shape_weight=2e-2,
                barrier_weight=7e-1,
                grad_clip=0.5,
                store_history=True,
                history_stride=5,
                early_stopping_patience=30,
                early_stopping_min_delta=1e-3,
            ),
        ),
        (
            "advecting_gaussian_solution_monitor",
            partial(
                advecting_gaussian_monitor,
                time=0.38,
                center0=(0.22, 0.40),
                velocity=(0.8, 0.28),
                sigma=0.09,
                alpha=45.0,
            ),
            AdaptationConfig(
                steps=140,
                lr=1.4e-2,
                movement_weight=2e-4,
                smoothness_weight=8e-4,
                shape_weight=2e-4,
                barrier_weight=1e-1,
                grad_clip=0.8,
                store_history=True,
                history_stride=4,
                early_stopping_patience=30,
            ),
        ),
        (
            "advecting_front_gradient_monitor",
            partial(
                advecting_front_gradient_monitor,
                time=0.36,
                front0=0.16,
                speed=0.82,
                normal=(1.0, 0.25),
                width=0.035,
                alpha=0.18,
            ),
            AdaptationConfig(
                steps=160,
                lr=1.1e-2,
                movement_weight=2e-4,
                smoothness_weight=8e-4,
                shape_weight=2e-4,
                barrier_weight=1e-1,
                grad_clip=0.8,
                store_history=True,
                history_stride=4,
                early_stopping_patience=30,
            ),
        ),
    ]

    print(f"mesh: {mesh.num_points} points, {mesh.num_cells} cells")
    for prefix, monitor_fn, config in cases:
        threshold = torch.quantile(monitor_fn(mesh.points, mesh.cell_blocks).detach(), 0.8)
        initial_high_area = _mean_area_above_monitor(mesh, monitor_fn, threshold)
        result = adapt_monitor_weighted_area(mesh, monitor_fn, config)
        final_high_area = _mean_area_above_monitor(result.mesh, monitor_fn, threshold)

        paths = save_adaptation_artifacts(
            mesh,
            result,
            output_dir,
            prefix=prefix,
            title=prefix.replace("_", " "),
            fps=14,
        )
        paths["monitor"] = save_mesh_cell_scalar_comparison(
            mesh,
            result.mesh,
            monitor_fn,
            output_dir / f"{prefix}_monitor.png",
            title=prefix.replace("_", " "),
            scalar_label="monitor",
        )
        if prefix == "time_integrated_advecting_gaussian_gradient_monitor":
            paths["monitor_gif"] = save_adaptation_gif(
                mesh,
                result,
                output_dir / f"{prefix}_monitor.gif",
                title=prefix.replace("_", " "),
                fps=14,
                cell_scalar_fn=monitor_fn,
                scalar_label="monitor",
                cmap="magma",
            )

        print(prefix)
        print(f"  high-monitor mean area: {float(initial_high_area):.6g} -> {float(final_high_area):.6g}")
        print(f"  optimizer loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
        print(f"  steps completed: {result.steps_completed} (early_stopped={result.early_stopped})")
        for name, path in paths.items():
            print(f"  {name}: {path}")


def _mean_area_above_monitor(mesh, monitor_fn, threshold: torch.Tensor) -> torch.Tensor:
    monitor = monitor_fn(mesh.points, mesh.cell_blocks).detach()
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[monitor >= threshold].mean()


if __name__ == "__main__":
    main()
