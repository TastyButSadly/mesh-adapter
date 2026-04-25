from __future__ import annotations

from pathlib import Path

from diff_mesh_adapter import AdaptationConfig, adapt_cell_area_equalization, cell_abs_areas
from diff_mesh_adapter.io import generate_nonconvex_hole_gmsh, read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_artifacts


def main() -> None:
    output_dir = Path("outputs/nonconvex_hole")
    mesh_path = generate_nonconvex_hole_gmsh(output_dir / "nonconvex_hole.msh")
    mesh = read_gmsh_mesh(mesh_path)

    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=180,
            lr=1.4e-2,
            movement_weight=2e-4,
            shape_weight=1e-4,
            grad_clip=0.8,
            store_history=True,
            early_stopping_patience=20,
            early_stopping_min_delta=1e-3,
        ),
    )
    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)

    paths = save_adaptation_artifacts(
        mesh,
        result,
        output_dir,
        prefix="nonconvex_hole",
        title="Nonconvex Gmsh mesh with hole: cell area equalization",
        fps=14,
    )

    print(f"mesh: {mesh.num_points} points, {mesh.num_cells} cells")
    print(f"area std: {float(initial_areas.std(unbiased=False)):.6g} -> {float(final_areas.std(unbiased=False)):.6g}")
    print(f"loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    print(f"steps completed: {result.steps_completed} (early_stopped={result.early_stopped})")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
