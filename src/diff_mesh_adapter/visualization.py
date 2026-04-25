from __future__ import annotations

from pathlib import Path

import numpy as np

from diff_mesh_adapter.adapt import AdaptationResult
from diff_mesh_adapter.geometry import cell_abs_areas
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


def _polygons_and_areas(mesh: MeshState) -> tuple[list[np.ndarray], np.ndarray]:
    areas = cell_abs_areas(mesh.points, mesh.cell_blocks).detach().cpu().numpy()
    polygons: list[np.ndarray] = []
    for block in mesh.cell_blocks:
        block_polys = mesh.points.detach().cpu()[block.detach().cpu()].numpy()
        polygons.extend([poly for poly in block_polys])
    return polygons, areas


def _area_norm(areas: np.ndarray):
    from matplotlib.colors import Normalize

    vmin = float(areas.min())
    vmax = float(areas.max())
    if vmin == vmax:
        pad = max(abs(vmin), 1.0) * 1e-12
        vmin -= pad
        vmax += pad
    return Normalize(vmin=vmin, vmax=vmax)


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
    area_history = [cell_abs_areas(mesh.points, mesh.cell_blocks).detach().cpu().numpy() for mesh in history_meshes]
    all_areas = np.concatenate(area_history)
    norm = Normalize(vmin=float(all_areas.min()), vmax=float(all_areas.max()))

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
        array=area_history[0],
        cmap="viridis",
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
    fig.colorbar(collection, ax=mesh_ax, shrink=0.9, label="cell area")

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
        collection.set_array(area_history[frame_idx])

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


def _history_steps(loss_count: int, history_count: int) -> list[int]:
    if history_count == loss_count:
        return list(range(loss_count))
    if history_count == 1:
        return [loss_count - 1]
    return [round(i * (loss_count - 1) / (history_count - 1)) for i in range(history_count)]
