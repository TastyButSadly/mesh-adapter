from __future__ import annotations

from functools import partial
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    MeshState,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_centroids,
    cell_edge_lengths,
    min_triangle_angle_degrees,
)
from diff_mesh_adapter.io import read_gmsh_mesh
from diff_mesh_adapter.monitors import advecting_gaussian_gradient_monitor, advecting_gaussian_solution
from diff_mesh_adapter.visualization import save_mesh_vtu
from examples.templates.gmsh_meshes import generate_unit_square_gmsh


def main() -> None:
    output_dir = Path("outputs/solution_driven_transient_adaptation_2d")
    paths = run_case(output_dir)
    print(f"ParaView series: {paths['pvd']}")
    print(f"frames: {len(paths['frames'])}")


def run_case(
    output_dir: str | Path,
    *,
    mesh_size: float = 0.075,
    steps_per_time: int = 35,
    time_count: int = 9,
) -> dict[str, Path | list[Path]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mesh_path = generate_unit_square_gmsh(
        output / "initial_unit_square.msh",
        boundary_size=mesh_size,
        refined_size=mesh_size,
    )
    initial_mesh = read_gmsh_mesh(mesh_path)
    current_mesh = initial_mesh
    reference_edge_lengths = cell_edge_lengths(initial_mesh.points, initial_mesh.cell_blocks).clamp_min(1e-12)
    times = torch.linspace(0.0, 0.72, time_count).tolist()

    config = AdaptationConfig(
        steps=steps_per_time,
        lr=1.2e-3,
        movement_weight=5e-3,
        smoothness_weight=7e-2,
        shape_weight=3e-2,
        quality_barrier_weight=6.0,
        boundary_quality_barrier_weight=20.0,
        min_cell_quality=0.54,
        min_step_cell_quality=0.50,
        edge_length_weight=0.08,
        edge_length_barrier_weight=0.8,
        boundary_edge_length_barrier_weight=8.0,
        max_edge_stretch=1.8,
        min_edge_compression=0.4,
        max_step_edge_stretch=2.2,
        min_step_edge_compression=0.34,
        barrier_weight=1.0,
        grad_clip=0.35,
        early_stopping_patience=12,
        early_stopping_min_delta=1e-3,
        early_stopping_relative=False,
    )

    frames: list[Path] = []
    datasets: list[tuple[float, Path]] = []
    high_monitor_area_ratios: list[float] = []

    for step, time in enumerate(times):
        monitor_fn = partial(
            advecting_gaussian_gradient_monitor,
            time=float(time),
            center0=(0.18, 0.30),
            velocity=(0.68, 0.40),
            sigma=0.085,
            alpha=0.18,
        )
        threshold = torch.quantile(monitor_fn(current_mesh.points, current_mesh.cell_blocks).detach(), 0.90)
        before_high_area = _mean_area_above_monitor(current_mesh, monitor_fn, threshold)

        result = adapt_monitor_weighted_area(
            current_mesh,
            monitor_fn,
            config,
            reference_points=initial_mesh.points,
        )
        current_mesh = result.mesh
        after_high_area = _mean_area_above_monitor(current_mesh, monitor_fn, threshold)
        high_monitor_area_ratios.append(float(after_high_area / before_high_area))

        frame_path = output / f"solution_driven_{step:04d}.vtu"
        save_mesh_vtu(
            current_mesh,
            frame_path,
            cell_data=_cell_data(current_mesh, initial_mesh, monitor_fn, float(time), reference_edge_lengths),
            point_data=_point_data(current_mesh, initial_mesh, float(time)),
        )
        frames.append(frame_path)
        datasets.append((float(time), frame_path))

        min_angle = float(min_triangle_angle_degrees(current_mesh.points, current_mesh.cell_blocks))
        max_edge_ratio = float((cell_edge_lengths(current_mesh.points, current_mesh.cell_blocks) / reference_edge_lengths).max())
        print(
            f"step={step:02d} t={time:.3f} "
            f"high-gradient area ratio={high_monitor_area_ratios[-1]:.3f} "
            f"loss={result.initial_loss:.5g}->{result.final_loss:.5g} "
            f"min-angle={min_angle:.3f} max-edge-ratio={max_edge_ratio:.3f}"
        )

    pvd = _write_pvd(output / "solution_driven_transient_adaptation.pvd", datasets)
    print(f"mean high-gradient area ratio: {sum(high_monitor_area_ratios) / len(high_monitor_area_ratios):.6g}")
    return {"pvd": pvd, "frames": frames}


def _cell_data(
    mesh: MeshState,
    initial_mesh: MeshState,
    monitor_fn,
    time: float,
    reference_edge_lengths: torch.Tensor,
) -> dict[str, np.ndarray]:
    centroids = cell_centroids(mesh.points, mesh.cell_blocks)
    solution = advecting_gaussian_solution(
        centroids,
        time=time,
        center0=(0.18, 0.30),
        velocity=(0.68, 0.40),
        sigma=0.085,
    )
    monitor = monitor_fn(mesh.points, mesh.cell_blocks)
    cell_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    initial_areas = cell_abs_areas(initial_mesh.points, initial_mesh.cell_blocks).to(dtype=cell_areas.dtype, device=cell_areas.device)
    edge_ratio = cell_edge_lengths(mesh.points, mesh.cell_blocks) / reference_edge_lengths
    return {
        "solution": _numpy(solution),
        "solution_gradient_monitor": _numpy(monitor),
        "cell_area": _numpy(cell_areas),
        "cell_area_ratio_to_initial": _numpy(cell_areas / initial_areas),
        "max_edge_ratio_to_initial": _numpy(_cell_max_edge_ratio(edge_ratio, mesh.cell_blocks)),
    }


def _point_data(mesh: MeshState, initial_mesh: MeshState, time: float) -> dict[str, np.ndarray]:
    solution = advecting_gaussian_solution(
        mesh.points,
        time=time,
        center0=(0.18, 0.30),
        velocity=(0.68, 0.40),
        sigma=0.085,
    )
    displacement = torch.linalg.norm(mesh.points - initial_mesh.points.to(device=mesh.points.device, dtype=mesh.points.dtype), dim=1)
    return {
        "solution": _numpy(solution),
        "mesh_displacement": _numpy(displacement),
        "is_boundary_node": _numpy(mesh.boundary_nodes.to(dtype=torch.int8)),
    }


def _cell_max_edge_ratio(edge_ratios: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    values = []
    offset = 0
    for block in cell_blocks:
        edge_count = int(block.shape[1])
        block_values = edge_ratios[offset : offset + block.shape[0] * edge_count].reshape(block.shape[0], edge_count)
        values.append(block_values.max(dim=1).values)
        offset += block.shape[0] * edge_count
    return torch.cat(values)


def _mean_area_above_monitor(mesh: MeshState, monitor_fn, threshold: torch.Tensor) -> torch.Tensor:
    monitor = monitor_fn(mesh.points, mesh.cell_blocks).detach()
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    return areas[monitor >= threshold].mean()


def _write_pvd(path: Path, datasets: list[tuple[float, Path]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ElementTree.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ElementTree.SubElement(root, "Collection")
    for time, dataset_path in datasets:
        ElementTree.SubElement(
            collection,
            "DataSet",
            timestep=f"{time:.12g}",
            group="",
            part="0",
            file=dataset_path.relative_to(path.parent).as_posix(),
        )
    tree = ElementTree.ElementTree(root)
    ElementTree.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


if __name__ == "__main__":
    main()
