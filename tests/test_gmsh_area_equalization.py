from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, adapt_cell_area_equalization, cell_abs_areas, cell_signed_areas
from diff_mesh_adapter.io import generate_nonconvex_hole_gmsh, generate_unit_square_gmsh, read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_artifacts


def test_gmsh_mesh_import_and_area_equalization(tmp_path):
    mesh_path = generate_unit_square_gmsh(tmp_path / "unit_square.msh")
    mesh = read_gmsh_mesh(mesh_path)

    initial_points = mesh.points.clone()
    initial_cells = tuple(block.clone() for block in mesh.cell_blocks)
    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=250, lr=2e-2, movement_weight=1e-4, shape_weight=0.0),
    )
    paths = save_adaptation_artifacts(
        mesh,
        result,
        Path("outputs/gmsh_area_equalization"),
        prefix="gmsh_area_equalization",
        title="Gmsh cell area equalization",
        include_gif=False,
    )

    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)

    assert result.mesh.num_points == mesh.num_points
    assert result.mesh.num_cells == mesh.num_cells
    for before, after in zip(initial_cells, result.mesh.cell_blocks):
        assert torch.equal(before, after)
    assert torch.allclose(result.mesh.points[mesh.boundary_nodes], initial_points[mesh.boundary_nodes])
    assert torch.all(final_signed > 0.0)
    assert final_areas.std(unbiased=False) < initial_areas.std(unbiased=False)
    assert result.final_loss < result.initial_loss
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_nonconvex_hole_gmsh_mesh_area_equalization_outputs_plots(tmp_path):
    mesh_path = generate_nonconvex_hole_gmsh(tmp_path / "nonconvex_hole.msh")
    mesh = read_gmsh_mesh(mesh_path)

    initial_points = mesh.points.clone()
    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=60,
            lr=1.4e-2,
            movement_weight=2e-4,
            shape_weight=1e-4,
            grad_clip=0.8,
            store_history=True,
            early_stopping_patience=20,
            early_stopping_min_delta=1e-3,
        ),
    )
    paths = save_adaptation_artifacts(
        mesh,
        result,
        Path("outputs/nonconvex_hole"),
        prefix="nonconvex_hole",
        title="Nonconvex Gmsh mesh with hole: cell area equalization",
        fps=12,
    )

    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)

    assert torch.allclose(result.mesh.points[mesh.boundary_nodes], initial_points[mesh.boundary_nodes])
    assert torch.all(final_signed > 0.0)
    assert final_areas.std(unbiased=False) < initial_areas.std(unbiased=False)
    assert result.final_loss < result.initial_loss
    assert result.steps_completed <= 60
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
