from __future__ import annotations

import math
from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, adapt_mesh_quality
from diff_mesh_adapter.geometry import (
    cell_signed_measures_and_shape_energy,
    max_tet_dihedral_angle_degrees,
    min_tet_dihedral_angle_degrees,
    tet_shape_energy,
    tet_signed_volumes,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_vtu_artifacts
from examples.templates.gmsh_meshes import generate_unit_cube_gmsh
from examples.templates.mesh_perturbation import perturb_internal_tet_nodes


def test_synthetic_tetra_geometry_metrics_distinguish_regular_and_sliver():
    regular_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0, 0.0],
            [0.5, math.sqrt(3.0) / 6.0, math.sqrt(2.0 / 3.0)],
        ],
        dtype=torch.float64,
    )
    sliver_points = regular_points.clone()
    sliver_points[3, 2] = 0.045

    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    inverted_cells = torch.tensor([[1, 0, 2, 3]], dtype=torch.long)

    regular_volumes = tet_signed_volumes(regular_points, cells)
    sliver_volumes = tet_signed_volumes(sliver_points, cells)
    inverted_volumes = tet_signed_volumes(regular_points, inverted_cells)

    expected_volume = torch.tensor([math.sqrt(2.0) / 12.0], dtype=regular_points.dtype)
    expected_dihedral = torch.tensor(math.degrees(math.acos(1.0 / 3.0)), dtype=regular_points.dtype)

    assert torch.allclose(regular_volumes, expected_volume, rtol=1e-12, atol=1e-12)
    assert torch.all(sliver_volumes > 0.0)
    assert torch.all(inverted_volumes < 0.0)
    assert tet_shape_energy(regular_points, cells).mean() < tet_shape_energy(sliver_points, cells).mean()
    combined_volumes, combined_energy = cell_signed_measures_and_shape_energy(regular_points, (cells,))
    assert torch.allclose(combined_volumes, regular_volumes)
    assert torch.allclose(combined_energy, tet_shape_energy(regular_points, cells))
    assert torch.isclose(min_tet_dihedral_angle_degrees(regular_points, (cells,)), expected_dihedral, atol=1e-10)
    assert torch.isclose(max_tet_dihedral_angle_degrees(regular_points, (cells,)), expected_dihedral, atol=1e-10)
    assert min_tet_dihedral_angle_degrees(regular_points, (cells,)) > min_tet_dihedral_angle_degrees(sliver_points, (cells,))
    assert max_tet_dihedral_angle_degrees(sliver_points, (cells,)) > max_tet_dihedral_angle_degrees(regular_points, (cells,))


def test_gmsh_unit_cube_tet_repair_preserves_topology_and_saves_vtu(tmp_path):
    mesh_path = generate_unit_cube_gmsh(tmp_path / "unit_cube.msh", boundary_size=0.36, refined_size=0.11)
    clean_mesh = read_gmsh_mesh(mesh_path)
    degraded_mesh = perturb_internal_tet_nodes(clean_mesh, amplitude=0.16, seed=41)

    initial_points = degraded_mesh.points.clone()
    initial_cells = tuple(block.clone() for block in degraded_mesh.cell_blocks)
    initial_energy = _mean_tet_shape_energy(degraded_mesh.points, degraded_mesh.cell_blocks)
    initial_min_dihedral = min_tet_dihedral_angle_degrees(degraded_mesh.points, degraded_mesh.cell_blocks)

    result = adapt_mesh_quality(
        degraded_mesh,
        AdaptationConfig(
            steps=90,
            lr=8e-3,
            movement_weight=2e-3,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            history_stride=3,
            early_stopping_patience=30,
            early_stopping_min_delta=1e-4,
        ),
    )

    final_energy = _mean_tet_shape_energy(result.mesh.points, result.mesh.cell_blocks)
    final_min_dihedral = min_tet_dihedral_angle_degrees(result.mesh.points, result.mesh.cell_blocks)
    final_volumes = _tet_signed_volumes(result.mesh.points, result.mesh.cell_blocks)
    paths = save_adaptation_vtu_artifacts(
        degraded_mesh,
        result,
        tmp_path / "vtu",
        prefix="gmsh_tet_mesh_repair",
    )

    assert result.mesh.num_points == degraded_mesh.num_points
    assert result.mesh.num_cells == degraded_mesh.num_cells
    for before, after in zip(initial_cells, result.mesh.cell_blocks):
        assert torch.equal(before, after)
    assert torch.allclose(result.mesh.points[degraded_mesh.boundary_nodes], initial_points[degraded_mesh.boundary_nodes])
    assert torch.all(final_volumes > 0.0)
    assert final_energy < initial_energy
    assert final_min_dihedral > initial_min_dihedral
    _assert_artifacts_exist(paths)


def _tet_signed_volumes(points: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tet_signed_volumes(points, block) for block in cell_blocks], dim=0)


def _mean_tet_shape_energy(points: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tet_shape_energy(points, block) for block in cell_blocks], dim=0).mean()


def _assert_artifacts_exist(paths: dict[str, object]) -> None:
    assert paths
    vtu_count = 0
    for value in paths.values():
        values = value if isinstance(value, list) else [value]
        for path_like in values:
            path = Path(path_like)
            if path.suffix == ".vtu":
                vtu_count += 1
            assert path.exists()
            assert path.stat().st_size > 0
    assert vtu_count >= 2
