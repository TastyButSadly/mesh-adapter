from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_signed_areas,
    min_triangle_angle_degrees,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.monitors import (
    advecting_front_gradient_monitor,
    advecting_gaussian_monitor,
    time_integrated_advecting_gaussian_gradient_monitor,
)
from diff_mesh_adapter.visualization import save_adaptation_artifacts, save_mesh_cell_scalar_comparison
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def test_advecting_gaussian_exact_solution_monitor_concentrates_cells(tmp_path):
    mesh = _coarse_square_mesh(tmp_path)
    monitor_fn = partial(
        advecting_gaussian_monitor,
        time=0.38,
        center0=(0.22, 0.40),
        velocity=(0.8, 0.28),
        sigma=0.09,
        alpha=45.0,
    )

    result = adapt_monitor_weighted_area(mesh, monitor_fn, _monitor_config(steps=140, lr=1.4e-2))

    _assert_monitor_concentration(mesh, result.mesh, monitor_fn, max_area_ratio=0.35)
    _assert_artifacts(mesh, result, monitor_fn, "advecting_gaussian_solution_monitor")


def test_advecting_front_gradient_monitor_concentrates_cells_near_front(tmp_path):
    mesh = _coarse_square_mesh(tmp_path)
    monitor_fn = partial(
        advecting_front_gradient_monitor,
        time=0.36,
        front0=0.16,
        speed=0.82,
        normal=(1.0, 0.25),
        width=0.035,
        alpha=0.18,
    )

    result = adapt_monitor_weighted_area(mesh, monitor_fn, _monitor_config(steps=160, lr=1.1e-2))

    _assert_monitor_concentration(mesh, result.mesh, monitor_fn, max_area_ratio=0.55)
    _assert_artifacts(mesh, result, monitor_fn, "advecting_front_gradient_monitor")


def test_time_integrated_gaussian_gradient_monitor_concentrates_cells_along_path(tmp_path):
    mesh = _coarse_square_mesh(tmp_path)
    monitor_fn = partial(
        time_integrated_advecting_gaussian_gradient_monitor,
        time_start=0.0,
        time_end=0.85,
        samples=8,
        center0=(0.18, 0.30),
        velocity=(0.65, 0.45),
        sigma=0.075,
        alpha=0.18,
    )

    result = adapt_monitor_weighted_area(
        mesh,
        monitor_fn,
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
    )

    _assert_monitor_concentration(mesh, result.mesh, monitor_fn, max_area_ratio=0.65)
    assert min_triangle_angle_degrees(result.mesh.points, result.mesh.cell_blocks) > 20.0
    _assert_artifacts(mesh, result, monitor_fn, "time_integrated_advecting_gaussian_gradient_monitor")


def _coarse_square_mesh(tmp_path: Path):
    mesh_path = generate_unit_square_gmsh(tmp_path / "advecting_monitor.msh", boundary_size=0.09, refined_size=0.09)
    return read_gmsh_mesh(mesh_path)


def _monitor_config(*, steps: int, lr: float) -> AdaptationConfig:
    return AdaptationConfig(
        steps=steps,
        lr=lr,
        movement_weight=2e-4,
        smoothness_weight=8e-4,
        shape_weight=2e-4,
        barrier_weight=1e-1,
        grad_clip=0.8,
        store_history=True,
        history_stride=4,
        early_stopping_patience=30,
    )


def _assert_monitor_concentration(initial_mesh, adapted_mesh, monitor_fn, *, max_area_ratio: float) -> None:
    threshold = torch.quantile(monitor_fn(initial_mesh.points, initial_mesh.cell_blocks).detach(), 0.8)
    initial_area = _mean_area_above_monitor(initial_mesh, monitor_fn, threshold)
    adapted_area = _mean_area_above_monitor(adapted_mesh, monitor_fn, threshold)
    final_signed = cell_signed_areas(adapted_mesh.points, adapted_mesh.cell_blocks)

    assert torch.all(final_signed > 0.0)
    assert adapted_area < max_area_ratio * initial_area


def _mean_area_above_monitor(mesh, monitor_fn, threshold: torch.Tensor) -> torch.Tensor:
    monitor = monitor_fn(mesh.points, mesh.cell_blocks).detach()
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[monitor >= threshold].mean()


def _assert_artifacts(initial_mesh, result, monitor_fn, prefix: str) -> None:
    output_dir = Path("outputs/advecting_monitor_2d")
    paths = save_adaptation_artifacts(
        initial_mesh,
        result,
        output_dir,
        prefix=prefix,
        title=prefix.replace("_", " "),
        fps=14,
    )
    paths["monitor"] = save_mesh_cell_scalar_comparison(
        initial_mesh,
        result.mesh,
        monitor_fn,
        output_dir / f"{prefix}_monitor.png",
        title=prefix.replace("_", " "),
        scalar_label="monitor",
    )

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
