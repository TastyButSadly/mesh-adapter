from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from diff_mesh_adapter.adapt import AdaptationResult
from diff_mesh_adapter.geometry import (
    cell_abs_areas,
    cell_centroids,
    cell_shape_energy,
    tet_min_dihedral_angles_degrees,
    tetra_signed_volumes,
)
from diff_mesh_adapter.mesh import MeshState


def save_mesh_area_comparison(
    initial_mesh: MeshState,
    adapted_mesh: MeshState,
    path: str | Path,
    *,
    title: str = "Gmsh cell area equalization",
    dpi: int = 180,
) -> Path:
    """Save a before/after mesh plot colored by fixed-topology cell area."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    initial_polys, initial_areas = _polygons_and_areas(initial_mesh)
    adapted_polys, adapted_areas = _polygons_and_areas(adapted_mesh)

    all_areas = np.concatenate([initial_areas, adapted_areas])
    norm = _area_norm(all_areas)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    fig.suptitle(title)

    collections = []
    for ax, polys, areas, panel_title in (
        (axes[0], initial_polys, initial_areas, f"Before: std={initial_areas.std():.4g}"),
        (axes[1], adapted_polys, adapted_areas, f"After: std={adapted_areas.std():.4g}"),
    ):
        collection = PolyCollection(
            polys,
            array=areas,
            cmap="viridis",
            norm=norm,
            edgecolors="#222222",
            linewidths=0.45,
            antialiaseds=True,
        )
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        collections.append(collection)

    colorbar = fig.colorbar(collections[-1], ax=axes, shrink=0.88)
    colorbar.set_label("cell area")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def save_mesh_area_snapshot(
    mesh: MeshState,
    path: str | Path,
    *,
    title: str = "Mesh cell areas",
    dpi: int = 180,
) -> Path:
    """Save one mesh state colored by fixed-topology cell area."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    polys, areas = _polygons_and_areas(mesh)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    collection = PolyCollection(
        polys,
        array=areas,
        cmap="viridis",
        norm=_area_norm(areas),
        edgecolors="#222222",
        linewidths=0.45,
        antialiaseds=True,
    )
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}: std={areas.std():.4g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.9)
    colorbar.set_label("cell area")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def save_mesh_cell_scalar_comparison(
    initial_mesh: MeshState,
    adapted_mesh: MeshState,
    scalar_fn,
    path: str | Path,
    *,
    title: str,
    scalar_label: str = "cell scalar",
    dpi: int = 180,
) -> Path:
    """Save a before/after 2D mesh plot colored by a cell-centered scalar."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    initial_polys, initial_values = _polygons_and_cell_scalar(initial_mesh, scalar_fn)
    adapted_polys, adapted_values = _polygons_and_cell_scalar(adapted_mesh, scalar_fn)
    norm = _area_norm(np.concatenate([initial_values, adapted_values]))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    fig.suptitle(title)
    collections = []
    for ax, polys, values, panel_title in (
        (axes[0], initial_polys, initial_values, "Before"),
        (axes[1], adapted_polys, adapted_values, "After"),
    ):
        collection = PolyCollection(
            polys,
            array=values,
            cmap="magma",
            norm=norm,
            edgecolors="#202020",
            linewidths=0.35,
            antialiaseds=True,
        )
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        collections.append(collection)

    colorbar = fig.colorbar(collections[-1], ax=axes, shrink=0.88)
    colorbar.set_label(scalar_label)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _polygons_and_areas(mesh: MeshState) -> tuple[list[np.ndarray], np.ndarray]:
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks).detach().cpu().numpy()
    polygons: list[np.ndarray] = []
    for block in mesh.cell_blocks:
        block_polys = mesh.points.detach().cpu()[block.detach().cpu()].numpy()
        polygons.extend([poly for poly in block_polys])
    return polygons, areas


def _polygons_and_cell_scalar(mesh: MeshState, scalar_fn) -> tuple[list[np.ndarray], np.ndarray]:
    if mesh.dim != 2:
        raise ValueError("cell scalar comparison supports 2D meshes only")
    polygons, _ = _polygons_and_areas(mesh)
    return polygons, _cell_scalar_values(mesh, scalar_fn)


def _cell_scalar_values(mesh: MeshState, scalar_fn) -> np.ndarray:
    values = scalar_fn(mesh.points, mesh.cell_blocks).detach().cpu().numpy()
    if values.shape != (mesh.num_cells,):
        centroids = cell_centroids(mesh.points, mesh.cell_blocks)
        values = scalar_fn(centroids).detach().cpu().numpy()
    return values


def _area_norm(areas: np.ndarray):
    from matplotlib.colors import Normalize

    vmin = float(areas.min())
    vmax = float(areas.max())
    if vmin == vmax:
        pad = max(abs(vmin), 1.0) * 1e-12
        vmin -= pad
        vmax += pad
    return Normalize(vmin=vmin, vmax=vmax)


def save_mesh_vtu(
    mesh: MeshState,
    path: str | Path,
    cell_data: dict[str, object] | None = None,
    point_data: dict[str, object] | None = None,
) -> Path:
    """Save a fixed-topology 2D triangle or 3D tetra mesh as a VTU file."""
    import meshio

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    vtu_mesh = meshio.Mesh(
        points=_meshio_points(mesh),
        cells=_meshio_cell_blocks(mesh),
        cell_data=_normalize_cell_data(cell_data, mesh.cell_blocks),
        point_data=_normalize_point_data(point_data, mesh.num_points),
    )
    meshio.write(output, vtu_mesh)
    return output


def save_adaptation_vtu_artifacts(
    initial_mesh: MeshState,
    result: AdaptationResult,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path | list[Path]]:
    """Save ParaView VTU/PVD artifacts and a loss plot for an adaptation run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    displacement = torch_displacement_norm(initial_mesh, result.mesh)
    paths: dict[str, Path | list[Path]] = {
        "initial": save_mesh_vtu(
            initial_mesh,
            output / f"{prefix}_initial.vtu",
            cell_data=_mesh_cell_diagnostics(initial_mesh, np.zeros(initial_mesh.num_cells)),
            point_data=_mesh_point_diagnostics(initial_mesh),
        ),
        "final": save_mesh_vtu(
            result.mesh,
            output / f"{prefix}_final.vtu",
            cell_data=_mesh_cell_diagnostics(result.mesh, displacement),
            point_data=_mesh_point_diagnostics(result.mesh),
        ),
        "loss": save_loss_history(
            result.loss_history,
            output / f"{prefix}_loss.png",
            title=f"{prefix} loss",
        ),
    }

    history_datasets = []
    history_paths = []
    for frame_idx, (step, mesh) in enumerate(_adaptation_history_meshes(initial_mesh, result)):
        history_key = f"history_{frame_idx:04d}"
        history_displacement = torch_displacement_norm(initial_mesh, mesh)
        history_path = save_mesh_vtu(
            mesh,
            output / f"{prefix}_{history_key}.vtu",
            cell_data=_mesh_cell_diagnostics(mesh, history_displacement),
            point_data=_mesh_point_diagnostics(mesh),
        )
        history_paths.append(history_path)
        history_datasets.append((step, history_path))

    paths["history"] = history_paths
    paths["pvd"] = _write_pvd(output / f"{prefix}_history.pvd", history_datasets)
    return paths


def _meshio_cell_blocks(mesh: MeshState):
    import meshio

    cell_type = _meshio_cell_type(mesh)
    return [
        meshio.CellBlock(cell_type, _to_numpy(block).astype(np.int64, copy=False))
        for block in mesh.cell_blocks
    ]


def _meshio_points(mesh: MeshState) -> np.ndarray:
    points = _to_numpy(mesh.points)
    if mesh.dim == 2:
        z = np.zeros((points.shape[0], 1), dtype=points.dtype)
        return np.concatenate([points, z], axis=1)
    return points


def _meshio_cell_type(mesh: MeshState) -> str:
    nodes_per_cell = {int(block.shape[1]) for block in mesh.cell_blocks}
    if mesh.dim == 2 and nodes_per_cell == {3}:
        return "triangle"
    if mesh.dim == 3 and nodes_per_cell == {4}:
        return "tetra"
    raise ValueError("VTU export supports 2D triangle meshes and 3D tetra meshes")


def _normalize_point_data(point_data: dict[str, object] | None, num_points: int) -> dict[str, np.ndarray]:
    if point_data is None:
        return {}

    normalized = {}
    for name, values in point_data.items():
        array = _to_numpy(values)
        if array.ndim == 0 or array.shape[0] != num_points:
            raise ValueError(f"point_data[{name!r}] must have one value per mesh point")
        normalized[str(name)] = array
    return normalized


def _normalize_cell_data(cell_data: dict[str, object] | None, cell_blocks: tuple[object, ...]) -> dict[str, list[np.ndarray]]:
    if cell_data is None:
        return {}

    normalized = {}
    for name, values in cell_data.items():
        normalized[str(name)] = _cell_data_by_block(values, cell_blocks)
    return normalized


def _cell_data_by_block(values: object, cell_blocks: tuple[object, ...]) -> list[np.ndarray]:
    block_sizes = [int(block.shape[0]) for block in cell_blocks]
    if isinstance(values, (list, tuple)) and len(values) == len(block_sizes):
        arrays = [_to_numpy(value) for value in values]
        if all(array.ndim > 0 and array.shape[0] == size for array, size in zip(arrays, block_sizes)):
            return arrays

    array = _to_numpy(values)
    if array.ndim == 0 or array.shape[0] != sum(block_sizes):
        raise ValueError("cell_data values must match the total cell count or be split per cell block")

    split_at = np.cumsum(block_sizes[:-1])
    return [part for part in np.split(array, split_at, axis=0)]


def _to_numpy(values: object) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _mesh_cell_diagnostics(mesh: MeshState, displacement_norm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if mesh.dim == 2:
        return {"cell_area": cell_abs_areas(mesh.points, mesh.cell_blocks).detach().cpu().numpy()}
    if mesh.dim == 3:
        signed_volumes = np.concatenate(
            [tetra_signed_volumes(mesh.points, block).detach().cpu().numpy() for block in mesh.cell_blocks],
            axis=0,
        )
        shape_energy = cell_shape_energy(mesh.points, mesh.cell_blocks).detach().cpu().numpy()
        min_dihedral = np.concatenate(
            [tet_min_dihedral_angles_degrees(mesh.points, block).detach().cpu().numpy() for block in mesh.cell_blocks],
            axis=0,
        )
        if displacement_norm is None:
            displacement_norm = np.zeros(mesh.num_cells)
        return {
            "shape_energy": shape_energy,
            "volume": np.abs(signed_volumes),
            "signed_volume": signed_volumes,
            "min_dihedral_deg": min_dihedral,
            "displacement_norm": displacement_norm,
        }
    return {}


def _mesh_point_diagnostics(mesh: MeshState) -> dict[str, np.ndarray]:
    return {"is_boundary_node": mesh.boundary_nodes.detach().cpu().numpy().astype(np.int8)}


def torch_displacement_norm(initial_mesh: MeshState, mesh: MeshState) -> np.ndarray:
    import torch

    point_displacement = torch.linalg.norm(mesh.points - initial_mesh.points.to(device=mesh.points.device, dtype=mesh.points.dtype), dim=1)
    cell_values = []
    for block in mesh.cell_blocks:
        cell_values.append(point_displacement[block].mean(dim=1).detach().cpu().numpy())
    return np.concatenate(cell_values, axis=0)


def _adaptation_history_meshes(initial_mesh: MeshState, result: AdaptationResult) -> list[tuple[int, MeshState]]:
    if result.points_history:
        steps = result.points_history_steps
        if steps is None:
            steps = _history_steps(len(result.loss_history), len(result.points_history))
        return [
            (
                int(step),
                MeshState(
                    points=points.to(dtype=initial_mesh.points.dtype),
                    cell_blocks=initial_mesh.cell_blocks,
                    boundary_nodes=initial_mesh.boundary_nodes,
                ),
            )
            for step, points in zip(steps, result.points_history)
        ]

    final_step = result.stopped_step if result.stopped_step else max(len(result.loss_history) - 1, 0)
    return [(0, initial_mesh), (int(final_step), result.mesh)]


def _write_pvd(path: str | Path, datasets: list[tuple[int, Path]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = ElementTree.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ElementTree.SubElement(root, "Collection")
    for step, dataset_path in datasets:
        ElementTree.SubElement(
            collection,
            "DataSet",
            timestep=str(step),
            group="",
            part="0",
            file=_pvd_relative_path(output.parent, dataset_path),
        )

    tree = ElementTree.ElementTree(root)
    ElementTree.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _pvd_relative_path(base: Path, path: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def save_loss_history(loss_history: list[float], path: str | Path, *, title: str = "Adaptation loss", dpi: int = 180) -> Path:
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(loss_history, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def save_adaptation_artifacts(
    initial_mesh: MeshState,
    result: AdaptationResult,
    output_dir: str | Path,
    *,
    prefix: str,
    title: str,
    include_gif: bool = True,
    fps: int = 12,
) -> dict[str, Path]:
    output = Path(output_dir)
    paths = {
        "initial": save_mesh_area_snapshot(
            initial_mesh,
            output / f"{prefix}_initial.png",
            title=f"{title} initial",
        ),
        "final": save_mesh_area_snapshot(
            result.mesh,
            output / f"{prefix}_final.png",
            title=f"{title} final",
        ),
        "mesh": save_mesh_area_comparison(
            initial_mesh,
            result.mesh,
            output / f"{prefix}_before_after.png",
            title=title,
        ),
        "loss": save_loss_history(
            result.loss_history,
            output / f"{prefix}_loss.png",
            title=f"{title} loss",
        ),
    }
    if include_gif:
        paths["gif"] = save_adaptation_gif(
            initial_mesh,
            result,
            output / f"{prefix}_adaptation.gif",
            title=title,
            fps=fps,
        )
    return paths


def save_adaptation_gif(
    initial_mesh: MeshState,
    result: AdaptationResult,
    path: str | Path,
    *,
    title: str = "Mesh adaptation",
    fps: int = 12,
    dpi: int = 120,
    cell_scalar_fn=None,
    scalar_label: str = "cell area",
    cmap: str = "viridis",
) -> Path:
    """Save a GIF with the current mesh and loss curve at each stored optimization step."""
    if result.points_history is None or not result.points_history:
        raise ValueError("AdaptationResult does not contain points_history; set store_history=True")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    history_meshes = [
        MeshState(points=points.to(dtype=initial_mesh.points.dtype), cell_blocks=initial_mesh.cell_blocks, boundary_nodes=initial_mesh.boundary_nodes)
        for points in result.points_history
    ]
    if cell_scalar_fn is None:
        scalar_history = [cell_abs_areas(mesh.points, mesh.cell_blocks).detach().cpu().numpy() for mesh in history_meshes]
    else:
        scalar_history = [_cell_scalar_values(mesh, cell_scalar_fn) for mesh in history_meshes]
    all_scalar_values = np.concatenate(scalar_history)
    norm = Normalize(vmin=float(all_scalar_values.min()), vmax=float(all_scalar_values.max()))

    all_points = np.concatenate([mesh.points.detach().cpu().numpy() for mesh in history_meshes], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    pad = 0.05 * span

    losses = np.asarray(result.loss_history, dtype=float)
    if result.points_history_steps is not None:
        history_steps = result.points_history_steps
    else:
        history_steps = _history_steps(len(result.loss_history), len(result.points_history))
    loss_ymin = float(losses.min())
    loss_ymax = float(losses.max())
    loss_pad = 0.05 * max(loss_ymax - loss_ymin, 1e-12)

    fig, (mesh_ax, loss_ax) = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.suptitle(title)

    initial_polys, _ = _polygons_and_areas(history_meshes[0])
    collection = PolyCollection(
        initial_polys,
        array=scalar_history[0],
        cmap=cmap,
        norm=norm,
        edgecolors="#222222",
        linewidths=0.42,
        antialiaseds=True,
    )
    mesh_ax.add_collection(collection)
    mesh_ax.set_aspect("equal", adjustable="box")
    mesh_ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    mesh_ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    mesh_ax.set_xlabel("x")
    mesh_ax.set_ylabel("y")
    fig.colorbar(collection, ax=mesh_ax, shrink=0.9, label=scalar_label)

    loss_ax.plot(np.arange(len(losses)), losses, color="#b8b8b8", linewidth=1.4)
    active_line, = loss_ax.plot([], [], color="#1f77b4", linewidth=2.4)
    marker, = loss_ax.plot([], [], marker="o", color="#d62728", markersize=5)
    loss_ax.set_xlim(0, max(len(losses) - 1, 1))
    loss_ax.set_ylim(loss_ymin - loss_pad, loss_ymax + loss_pad)
    loss_ax.set_xlabel("optimization step")
    loss_ax.set_ylabel("loss")
    loss_ax.grid(True, alpha=0.3)

    def update(frame_idx: int):
        mesh = history_meshes[frame_idx]
        polys, _ = _polygons_and_areas(mesh)
        collection.set_verts(polys)
        collection.set_array(scalar_history[frame_idx])

        step = history_steps[frame_idx]
        active_line.set_data(np.arange(step + 1), losses[: step + 1])
        marker.set_data([step], [losses[step]])
        mesh_ax.set_title(f"mesh step {step}")
        loss_ax.set_title(f"loss={losses[step]:.4g}")
        return collection, active_line, marker

    animation = FuncAnimation(fig, update, frames=len(history_meshes), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output


def save_transient_adaptation_gif(
    meshes: list[MeshState],
    cell_scalar_fns: list,
    series_values: list[float],
    path: str | Path,
    *,
    times: list[float] | None = None,
    title: str = "Transient mesh adaptation",
    scalar_label: str = "monitor",
    series_label: str = "final optimization loss",
    fps: int = 8,
    dpi: int = 120,
    cmap: str = "magma",
) -> Path:
    """Save a GIF over physical time for a sequence of already adapted meshes."""
    if not meshes:
        raise ValueError("meshes must contain at least one frame")
    if len(cell_scalar_fns) != len(meshes):
        raise ValueError("cell_scalar_fns must match meshes")
    if len(series_values) != len(meshes):
        raise ValueError("series_values must match meshes")
    if times is not None and len(times) != len(meshes):
        raise ValueError("times must match meshes")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    scalar_history = [_cell_scalar_values(mesh, scalar_fn) for mesh, scalar_fn in zip(meshes, cell_scalar_fns)]
    scalar_values = np.concatenate(scalar_history)
    norm = Normalize(vmin=float(scalar_values.min()), vmax=float(scalar_values.max()))

    all_points = np.concatenate([mesh.points.detach().cpu().numpy() for mesh in meshes], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    pad = 0.05 * span

    series = np.asarray(series_values, dtype=float)
    series_ymin = float(series.min())
    series_ymax = float(series.max())
    series_pad = 0.05 * max(series_ymax - series_ymin, 1e-12)

    fig, (mesh_ax, loss_ax) = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.suptitle(title)

    initial_polys, _ = _polygons_and_areas(meshes[0])
    collection = PolyCollection(
        initial_polys,
        array=scalar_history[0],
        cmap=cmap,
        norm=norm,
        edgecolors="#222222",
        linewidths=0.35,
        antialiaseds=True,
    )
    mesh_ax.add_collection(collection)
    mesh_ax.set_aspect("equal", adjustable="box")
    mesh_ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    mesh_ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    mesh_ax.set_xlabel("x")
    mesh_ax.set_ylabel("y")
    fig.colorbar(collection, ax=mesh_ax, shrink=0.9, label=scalar_label)

    x_values = np.arange(len(series))
    loss_ax.plot(x_values, series, color="#b8b8b8", linewidth=1.4)
    active_line, = loss_ax.plot([], [], color="#1f77b4", linewidth=2.4)
    marker, = loss_ax.plot([], [], marker="o", color="#d62728", markersize=5)
    loss_ax.set_xlim(0, max(len(series) - 1, 1))
    loss_ax.set_ylim(series_ymin - series_pad, series_ymax + series_pad)
    loss_ax.set_xlabel("physical step")
    loss_ax.set_ylabel(series_label)
    loss_ax.grid(True, alpha=0.3)

    def update(frame_idx: int):
        polygons, _ = _polygons_and_areas(meshes[frame_idx])
        collection.set_verts(polygons)
        collection.set_array(scalar_history[frame_idx])
        active_line.set_data(x_values[: frame_idx + 1], series[: frame_idx + 1])
        marker.set_data([x_values[frame_idx]], [series[frame_idx]])
        if times is None:
            mesh_ax.set_title(f"physical step {frame_idx}")
        else:
            mesh_ax.set_title(f"t={times[frame_idx]:.3f}")
        loss_ax.set_title(f"{series_label}={series[frame_idx]:.4g}")
        return collection, active_line, marker

    animation = FuncAnimation(fig, update, frames=len(meshes), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output


def _history_steps(loss_count: int, history_count: int) -> list[int]:
    if history_count == loss_count:
        return list(range(loss_count))
    if history_count == 1:
        return [loss_count - 1]
    return [round(i * (loss_count - 1) / (history_count - 1)) for i in range(history_count)]
