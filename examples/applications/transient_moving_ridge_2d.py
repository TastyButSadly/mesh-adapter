from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    MeshState,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_edge_lengths,
    cell_signed_areas,
    min_triangle_angle_degrees,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.monitors import moving_gaussian_ridge_monitor
from diff_mesh_adapter.visualization import save_mesh_cell_scalar_comparison, save_transient_adaptation_gif
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def main() -> None:
    output_dir = Path("outputs/transient_moving_ridge_2d")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, mesh_size in (("same", 0.06), ("fine", 0.03)):
        _run_case(output_dir, name=name, mesh_size=mesh_size)


def _run_case(output_dir: Path, *, name: str, mesh_size: float) -> None:
    mesh_path = generate_unit_square_gmsh(
        output_dir / f"moving_ridge_{name}.msh",
        boundary_size=mesh_size,
        refined_size=mesh_size,
    )
    initial_mesh = read_gmsh_mesh(mesh_path)
    times = torch.linspace(0.0, 0.82, 13).tolist()
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

    current_mesh = initial_mesh
    adapted_meshes: list[MeshState] = []
    monitor_fns = []
    relative_loss_reductions: list[float] = []
    high_monitor_area_ratios: list[float] = []
    min_angles: list[float] = []
    max_edge_ratios: list[float] = []
    reference_edge_lengths = cell_edge_lengths(initial_mesh.points, initial_mesh.cell_blocks).clamp_min(config.area_eps)

    print(f"{name} mesh: {initial_mesh.num_points} points, {initial_mesh.num_cells} cells, h={mesh_size}")
    for step, time in enumerate(times):
        monitor_fn = partial(
            moving_gaussian_ridge_monitor,
            time=float(time),
            offset0=0.08,
            speed=0.88,
            normal=(1.0, 0.20),
            width=0.05,
            alpha=20.0,
        )
        threshold = torch.quantile(monitor_fn(current_mesh.points, current_mesh.cell_blocks).detach(), 0.90)
        before_high_area = _mean_area_above_monitor(current_mesh, monitor_fn, threshold)
        result = adapt_monitor_weighted_area(current_mesh, monitor_fn, config, reference_points=initial_mesh.points)
        after_high_area = _mean_area_above_monitor(result.mesh, monitor_fn, threshold)

        current_mesh = result.mesh
        adapted_meshes.append(current_mesh)
        monitor_fns.append(monitor_fn)
        relative_loss_reductions.append((result.initial_loss - result.final_loss) / max(abs(result.initial_loss), config.area_eps))
        high_monitor_area_ratios.append(float(after_high_area / before_high_area))

        min_signed = float(cell_signed_areas(current_mesh.points, current_mesh.cell_blocks).min())
        min_angle = float(min_triangle_angle_degrees(current_mesh.points, current_mesh.cell_blocks))
        max_edge_ratio = float((cell_edge_lengths(current_mesh.points, current_mesh.cell_blocks) / reference_edge_lengths).max())
        min_angles.append(min_angle)
        max_edge_ratios.append(max_edge_ratio)
        print(
            f"{name} step={step:02d} t={time:.3f} "
            f"high-area-ratio={high_monitor_area_ratios[-1]:.3f} "
            f"relative-loss-reduction={relative_loss_reductions[-1]:.3f} "
            f"loss={result.initial_loss:.5g}->{result.final_loss:.5g} "
            f"opt-steps={result.steps_completed} early={result.early_stopped} "
            f"min-area={min_signed:.5g} min-angle={min_angle:.3f} max-edge-ratio={max_edge_ratio:.3f}"
        )

    gif_path = save_transient_adaptation_gif(
        adapted_meshes,
        monitor_fns,
        high_monitor_area_ratios,
        output_dir / f"moving_ridge_{name}_transient_adaptation.gif",
        times=[float(time) for time in times],
        title=f"Transient moving Gaussian ridge adaptation ({name})",
        scalar_label="monitor",
        series_label="peak area ratio",
        fps=5,
    )
    comparison_path = save_mesh_cell_scalar_comparison(
        initial_mesh,
        adapted_meshes[-1],
        monitor_fns[-1],
        output_dir / f"moving_ridge_{name}_initial_final_monitor.png",
        title=f"Moving Gaussian ridge: initial vs final-time adapted mesh ({name})",
        scalar_label="monitor",
    )

    displacement = torch.linalg.norm(current_mesh.points - initial_mesh.points, dim=1)
    print(f"final min signed area: {float(cell_signed_areas(current_mesh.points, current_mesh.cell_blocks).min()):.6g}")
    print(f"final min triangle angle: {float(min_triangle_angle_degrees(current_mesh.points, current_mesh.cell_blocks)):.6g}")
    print(f"min triangle angle over time: {min(min_angles):.6g}")
    print(f"max edge ratio over time: {max(max_edge_ratios):.6g}")
    print(f"max displacement: {float(displacement.max()):.6g}")
    print(f"boundary max displacement: {float(displacement[initial_mesh.boundary_nodes].max()):.6g}")
    print(f"mean high-monitor area ratio: {sum(high_monitor_area_ratios) / len(high_monitor_area_ratios):.6g}")
    print(f"mean relative loss reduction: {sum(relative_loss_reductions) / len(relative_loss_reductions):.6g}")
    print(f"gif: {gif_path}")
    print(f"comparison: {comparison_path}")


def _mean_area_above_monitor(mesh: MeshState, monitor_fn, threshold: torch.Tensor) -> torch.Tensor:
    monitor = monitor_fn(mesh.points, mesh.cell_blocks).detach()
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[monitor >= threshold].mean()


if __name__ == "__main__":
    main()
