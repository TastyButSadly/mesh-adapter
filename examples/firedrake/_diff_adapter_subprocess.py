from __future__ import annotations

import argparse
from dataclasses import dataclass
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
        dtype = _npz_string(data, "dtype", "float64")
        cell_monitor_np = data["cell_monitor"] if "cell_monitor" in data else None
        points, info = adapt_coordinates(
            points_np=data["points"],
            cells_np=cells_np,
            monitor_np=data["monitor"] if "monitor" in data else None,
            cell_monitor_np=cell_monitor_np,
            steps=int(data["steps"]),
            lr=float(data["lr"]),
            profile=str(data["profile"]) if "profile" in data else "regularized",
            dtype=dtype,
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


@dataclass
class AdapterTopologyCache:
    num_points: int | None = None
    cells_source_id: int | None = None
    cells_np: np.ndarray | None = None
    cells: torch.Tensor | None = None
    boundary_nodes: torch.Tensor | None = None
    hits: int = 0
    misses: int = 0

    def tensors(self, num_points: int, cells_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.num_points == num_points
            and self.cells_np is not None
            and self.cells_np.shape == cells_np.shape
            and ((self.cells_source_id == id(cells_np) and not cells_np.flags.writeable) or np.array_equal(self.cells_np, cells_np))
            and self.cells is not None
            and self.boundary_nodes is not None
        ):
            self.hits += 1
            return self.cells, self.boundary_nodes

        cells_copy = np.ascontiguousarray(cells_np).copy()
        boundary = _boundary_nodes_from_cells(num_points, cells_copy)
        self.num_points = num_points
        self.cells_source_id = id(cells_np)
        self.cells_np = cells_copy
        self.cells = torch.as_tensor(cells_copy, dtype=torch.long)
        self.boundary_nodes = torch.as_tensor(boundary, dtype=torch.bool)
        self.misses += 1
        return self.cells, self.boundary_nodes


def adapt_coordinates(
    *,
    points_np: np.ndarray,
    cells_np: np.ndarray | None = None,
    triangles_np: np.ndarray | None = None,
    monitor_np: np.ndarray | None = None,
    cell_monitor_np: np.ndarray | None = None,
    steps: int,
    lr: float,
    profile: str = "regularized",
    dtype: str | np.dtype | torch.dtype = "float64",
    topology_cache: AdapterTopologyCache | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if cells_np is None:
        if triangles_np is None:
            raise ValueError("adapt_coordinates requires cells_np or triangles_np")
        cells_np = triangles_np
    if cells_np.ndim != 2 or cells_np.shape[1] not in (3, 4):
        raise ValueError("cells_np must have shape (n_cells, 3) or (n_cells, 4)")

    if cell_monitor_np is None:
        if monitor_np is None:
            raise ValueError("adapt_coordinates requires monitor_np or cell_monitor_np")
        cell_monitor_np = monitor_np[cells_np].mean(axis=1)
    cell_monitor_np = np.maximum(cell_monitor_np, 1.0e-12)
    if not points_np.flags.writeable:
        points_np = points_np.copy()
    if not cell_monitor_np.flags.writeable:
        cell_monitor_np = cell_monitor_np.copy()
    torch_dtype = _torch_dtype(dtype)
    points = torch.as_tensor(points_np, dtype=torch_dtype)
    if topology_cache is None:
        cells = torch.as_tensor(cells_np, dtype=torch.long)
        boundary_nodes = torch.as_tensor(_boundary_nodes_from_cells(points_np.shape[0], cells_np), dtype=torch.bool)
    else:
        cells, boundary_nodes = topology_cache.tensors(points_np.shape[0], cells_np)
    cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)
    mesh_state = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)

    def monitor_fn(query_points: torch.Tensor, _cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return cell_monitor

    config = _adaptation_config(profile, steps=steps, lr=lr)
    result = adapt_monitor_weighted_area(mesh_state, monitor_fn, config, reference_points=points)
    adapted = result.mesh.points.cpu().numpy().astype(points_np.dtype, copy=False)
    return adapted, {
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
            collect_step_diagnostics=False,
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
            collect_step_diagnostics=False,
            early_stopping_patience=12,
            early_stopping_min_delta=1.0e-3,
            early_stopping_relative=False,
        )
    raise ValueError(f"Unknown adapter profile: {profile}")


def _torch_dtype(dtype: str | np.dtype | torch.dtype) -> torch.dtype:
    if dtype is torch.float64 or str(dtype) in {"float64", "torch.float64"}:
        return torch.float64
    if dtype is torch.float32 or str(dtype) in {"float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported adapter dtype: {dtype}")


def _npz_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


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
