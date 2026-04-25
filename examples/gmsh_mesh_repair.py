from __future__ import annotations

from pathlib import Path

from diff_mesh_adapter import AdaptationConfig, adapt_mesh_quality, min_triangle_angle_degrees, normalized_edge_length_variance
from diff_mesh_adapter.demo import perturb_internal_nodes
from diff_mesh_adapter.io import generate_unit_square_gmsh, read_gmsh_mesh
from diff_mesh_adapter.visualization import save_adaptation_artifacts


def main() -> None:
    output_dir = Path("outputs/mesh_repair")
    mesh_path = generate_unit_square_gmsh(output_dir / "unit_square_repair.msh", boundary_size=0.16, refined_size=0.16)
    clean_mesh = read_gmsh_mesh(mesh_path)
    degraded_mesh = perturb_internal_nodes(clean_mesh, amplitude=0.075, seed=41)

    initial_quality = normalized_edge_length_variance(degraded_mesh.points, degraded_mesh.cell_blocks)
    initial_min_angle = min_triangle_angle_degrees(degraded_mesh.points, degraded_mesh.cell_blocks)
    result = adapt_mesh_quality(
        degraded_mesh,
        AdaptationConfig(
            steps=160,
            lr=1.5e-2,
            movement_weight=2e-3,
            barrier_weight=1e-1,
            grad_clip=0.8,
            store_history=True,
            early_stopping_patience=20,
            early_stopping_min_delta=1e-3,
        ),
    )
    final_quality = normalized_edge_length_variance(result.mesh.points, result.mesh.cell_blocks)
    final_min_angle = min_triangle_angle_degrees(result.mesh.points, result.mesh.cell_blocks)

    paths = save_adaptation_artifacts(
        degraded_mesh,
        result,
        output_dir,
        prefix="mesh_repair",
        title="Fixed-topology mesh repair",
        fps=14,
    )

    print(f"mesh: {degraded_mesh.num_points} points, {degraded_mesh.num_cells} cells")
    print(f"quality loss: {float(initial_quality):.6g} -> {float(final_quality):.6g}")
    print(f"min angle: {float(initial_min_angle):.6g} -> {float(final_min_angle):.6g}")
    print(f"optimizer loss: {result.initial_loss:.6g} -> {result.final_loss:.6g}")
    print(f"steps completed: {result.steps_completed} (early_stopped={result.early_stopped})")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
