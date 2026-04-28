from __future__ import annotations

from pathlib import Path

from diff_mesh_adapter import AdaptationConfig, adapt_cell_area_equalization, cell_abs_areas
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_artifacts
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def main() -> None:
    output_dir = Path("outputs")
    mesh_path = generate_unit_square_gmsh(output_dir / "unit_square.msh")
    mesh = read_gmsh_mesh(mesh_path)

    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=250, lr=2e-2, movement_weight=1e-4, shape_weight=0.0),
    )
    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)
    paths = save_adaptation_artifacts(
        mesh,
        result,
        output_dir,
        prefix="gmsh_area_equalization",
        title="Gmsh cell area equalization",
        include_gif=False,
    )

    print(f"mesh: {mesh.num_points} points, {mesh.num_cells} cells")
    print(f"area std: {float(initial_areas.std(unbiased=False)):.6g} -> {float(final_areas.std(unbiased=False)):.6g}")
    print(f"loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
