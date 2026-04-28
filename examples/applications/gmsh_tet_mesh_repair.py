from __future__ import annotations

from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, adapt_mesh_quality
from diff_mesh_adapter.geometry import (
    max_tet_dihedral_angle_degrees,
    min_tet_dihedral_angle_degrees,
    tet_shape_energy,
    tet_signed_volumes,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_vtu_artifacts
from examples.templates.gmsh_meshes import generate_unit_cube_gmsh
from examples.templates.mesh_perturbation import perturb_internal_tet_nodes


def main() -> None:
    output_dir = Path("outputs/tet_mesh_repair")
    mesh_path = generate_unit_cube_gmsh(output_dir / "unit_cube_repair.msh", boundary_size=0.36, refined_size=0.11)
    clean_mesh = read_gmsh_mesh(mesh_path)
    degraded_mesh = perturb_internal_tet_nodes(clean_mesh, amplitude=0.16, seed=41)

    initial_energy = mean_tet_shape_energy(degraded_mesh.points, degraded_mesh.cell_blocks)
    initial_min_dihedral = min_tet_dihedral_angle_degrees(degraded_mesh.points, degraded_mesh.cell_blocks)
    initial_max_dihedral = max_tet_dihedral_angle_degrees(degraded_mesh.points, degraded_mesh.cell_blocks)
    result = adapt_mesh_quality(
        degraded_mesh,
        AdaptationConfig(
            steps=160,
            lr=8e-3,
            movement_weight=2e-3,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            history_stride=4,
            early_stopping_patience=30,
            early_stopping_min_delta=1e-4,
        ),
    )
    final_energy = mean_tet_shape_energy(result.mesh.points, result.mesh.cell_blocks)
    final_min_dihedral = min_tet_dihedral_angle_degrees(result.mesh.points, result.mesh.cell_blocks)
    final_max_dihedral = max_tet_dihedral_angle_degrees(result.mesh.points, result.mesh.cell_blocks)
    final_min_volume = tet_signed_volumes_for_blocks(result.mesh.points, result.mesh.cell_blocks).min()

    paths = save_adaptation_vtu_artifacts(
        degraded_mesh,
        result,
        output_dir,
        prefix="tet_mesh_repair",
    )

    print(f"mesh: {degraded_mesh.num_points} points, {degraded_mesh.num_cells} tetrahedra")
    print(f"shape energy: {float(initial_energy):.6g} -> {float(final_energy):.6g}")
    print(f"min dihedral: {float(initial_min_dihedral):.6g} -> {float(final_min_dihedral):.6g}")
    print(f"max dihedral: {float(initial_max_dihedral):.6g} -> {float(final_max_dihedral):.6g}")
    print(f"min signed volume: {float(final_min_volume):.6g}")
    print(f"optimizer loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    print(f"steps completed: {result.steps_completed} (early_stopped={result.early_stopped})")
    for name, path in paths.items():
        if isinstance(path, list):
            print(f"{name}: {len(path)} files")
        else:
            print(f"{name}: {path}")


def tet_signed_volumes_for_blocks(points: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tet_signed_volumes(points, block) for block in cell_blocks], dim=0)


def mean_tet_shape_energy(points: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tet_shape_energy(points, block) for block in cell_blocks], dim=0).mean()


if __name__ == "__main__":
    main()
