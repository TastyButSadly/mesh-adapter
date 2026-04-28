from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

from diff_mesh_adapter import AdaptationConfig, MeshState, adapt_monitor_weighted_area


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Torch fixed-topology adapter for one Firedrake exchange request.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        data = np.load(args.input)
        cells_np = data["cells"] if "cells" in data else data["triangles"]
        points, info = adapt_coordinates(
            points_np=data["points"],
            cells_np=cells_np,
            monitor_np=data["monitor"],
            steps=int(data["steps"]),
            lr=float(data["lr"]),
            profile=str(data["profile"]) if "profile" in data else "regularized",
        )
        _write_npz_atomic(
            args.output,
            points=points,
            initial_loss=np.array(info["initial_loss"], dtype=np.float64),
            final_loss=np.array(info["final_loss"], dtype=np.float64),
            steps_completed=np.array(info["steps_completed"], dtype=np.int64),
            early_stopped=np.array(info["early_stopped"], dtype=bool),
        )
    except Exception:
        _write_npz_atomic(args.output, error=np.array(traceback.format_exc()))
        raise


def adapt_coordinates(
    *,
    points_np: np.ndarray,
    cells_np: np.ndarray | None = None,
    triangles_np: np.ndarray | None = None,
    monitor_np: np.ndarray,
    steps: int,
    lr: float,
    profile: str = "regularized",
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if cells_np is None:
        if triangles_np is None:
            raise ValueError("adapt_coordinates requires cells_np or triangles_np")
        cells_np = triangles_np
    if cells_np.ndim != 2 or cells_np.shape[1] not in (3, 4):
        raise ValueError("cells_np must have shape (n_cells, 3) or (n_cells, 4)")

    cell_monitor_np = np.maximum(monitor_np[cells_np].mean(axis=1), 1.0e-12)
    points = torch.as_tensor(points_np, dtype=torch.float64)
    cells = torch.as_tensor(cells_np, dtype=torch.long)
    boundary_nodes = torch.as_tensor(_boundary_nodes_from_cells(points_np.shape[0], cells_np), dtype=torch.bool)
    cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch.float64)
    mesh_state = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)

    def monitor_fn(query_points: torch.Tensor, _cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return cell_monitor.to(device=query_points.device, dtype=query_points.dtype)

    config = _adaptation_config(profile, steps=steps, lr=lr)
    result = adapt_monitor_weighted_area(mesh_state, monitor_fn, config, reference_points=points)
    return result.mesh.points.cpu().numpy(), {
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "steps_completed": int(result.steps_completed),
        "early_stopped": bool(result.early_stopped),
    }


def _adaptation_config(profile: str, *, steps: int, lr: float) -> AdaptationConfig:
    if profile == "regularized":
        return AdaptationConfig(
            steps=steps,
            lr=lr,
            movement_weight=2.0e-3,
            smoothness_weight=5.0e-2,
            shape_weight=2.0e-2,
            quality_barrier_weight=4.0,
            boundary_quality_barrier_weight=20.0,
            min_cell_quality=0.35,
            edge_length_weight=5.0e-2,
            edge_length_barrier_weight=0.5,
            boundary_edge_length_barrier_weight=5.0,
            max_edge_stretch=1.8,
            min_edge_compression=0.4,
            max_step_edge_stretch=2.2,
            min_step_edge_compression=0.3,
            barrier_weight=1.0,
            grad_clip=0.25,
            early_stopping_patience=12,
            early_stopping_min_delta=1.0e-3,
            early_stopping_relative=False,
        )
    if profile == "monitor-only":
        return AdaptationConfig(
            steps=steps,
            lr=lr,
            movement_weight=0.0,
            smoothness_weight=0.0,
            shape_weight=0.0,
            quality_barrier_weight=0.0,
            boundary_quality_barrier_weight=0.0,
            edge_length_weight=0.0,
            edge_length_barrier_weight=0.0,
            boundary_edge_length_barrier_weight=0.0,
            max_step_edge_stretch=None,
            min_step_edge_compression=None,
            barrier_weight=0.0,
            grad_clip=0.25,
            early_stopping_patience=12,
            early_stopping_min_delta=1.0e-3,
            early_stopping_relative=False,
        )
    raise ValueError(f"Unknown adapter profile: {profile}")


def _boundary_nodes_from_cells(num_points: int, cells: np.ndarray) -> np.ndarray:
    if cells.shape[1] == 3:
        return _boundary_nodes_from_triangles(num_points, cells)
    if cells.shape[1] == 4:
        return _boundary_nodes_from_tetrahedra(num_points, cells)
    raise ValueError("Only triangle and tetrahedron cells are supported")


def _boundary_nodes_from_triangles(num_points: int, triangles: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_points, dtype=bool)
    edge_count: dict[tuple[int, int], int] = {}
    for cell in triangles.tolist():
        edges = ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))
        for a, b in edges:
            key = tuple(sorted((int(a), int(b))))
            edge_count[key] = edge_count.get(key, 0) + 1
    for edge, count in edge_count.items():
        if count == 1:
            mask[list(edge)] = True
    return mask


def _boundary_nodes_from_tetrahedra(num_points: int, tetrahedra: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_points, dtype=bool)
    face_count: dict[tuple[int, int, int], int] = {}
    for cell in tetrahedra.tolist():
        faces = (
            (cell[0], cell[1], cell[2]),
            (cell[0], cell[1], cell[3]),
            (cell[0], cell[2], cell[3]),
            (cell[1], cell[2], cell[3]),
        )
        for face in faces:
            key = tuple(sorted(int(idx) for idx in face))
            face_count[key] = face_count.get(key, 0) + 1
    for face, count in face_count.items():
        if count == 1:
            mask[list(face)] = True
    return mask


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.rename(path)


if __name__ == "__main__":
    main()
