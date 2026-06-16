from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    AdaptationTopologyCache,
    MeshState,
    ReplicatorLaplacianConfig,
    ReplicatorLaplacianTopologyCache,
    SobolevTransportConfig,
    SobolevTransportTopologyCache,
    adapt_monitor_weighted_area,
    adapt_monitor_weighted_area_analytic_fast,
    adapt_replicator_laplacian,
    adapt_sobolev_transport_map,
)
from scripts.learned_monitor_v4 import CellMonitorV4, TopologyV4, compute_v4_features, feature_dim_for_mode
from scripts.learned_monitor_v4.monitor import message_passing_mean_v4


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
            learned_u_nodes_np=data["learned_u_nodes"] if "learned_u_nodes" in data else None,
            learned_weights_path=_npz_string(data, "learned_weights_path", None),
            learned_feature_mode=_npz_string(data, "learned_feature_mode", "full7"),
            learned_recalibrate=bool(data["learned_recalibrate"]) if "learned_recalibrate" in data else True,
            learned_smoothing_steps=int(data["learned_smoothing_steps"]) if "learned_smoothing_steps" in data else 0,
            learned_smoothing_weight=float(data["learned_smoothing_weight"]) if "learned_smoothing_weight" in data else 0.5,
            steps=int(data["steps"]),
            lr=float(data["lr"]),
            profile=str(data["profile"]) if "profile" in data else "regularized",
            dtype=dtype,
            movement_weight=_npz_float(data, "adapter_movement_weight"),
            smoothness_weight=_npz_float(data, "adapter_smoothness_weight"),
            max_step_edge_stretch=_npz_float(data, "adapter_max_step_edge_stretch"),
            min_step_edge_compression=_npz_float(data, "adapter_min_step_edge_compression"),
        )
        _write_npz_atomic(
            args.output,
            points=points,
            initial_loss=np.array(info["initial_loss"], dtype=np.float64),
            final_loss=np.array(info["final_loss"], dtype=np.float64),
            steps_completed=np.array(info["steps_completed"], dtype=np.int64),
            early_stopped=np.array(info["early_stopped"], dtype=bool),
            adapter_forward_loss=np.array(info.get("adapter_forward_loss", 0.0), dtype=np.float64),
            adapter_backward_or_grad=np.array(info.get("adapter_backward_or_grad", 0.0), dtype=np.float64),
            adapter_sobolev_solve=np.array(info.get("adapter_sobolev_solve", 0.0), dtype=np.float64),
            adapter_validation=np.array(info.get("adapter_validation", 0.0), dtype=np.float64),
            adapter_step=np.array(info.get("adapter_step", 0.0), dtype=np.float64),
            adapter_alpha=np.array(info.get("adapter_alpha", 0.0), dtype=np.float64),
            adapter_cg_iters=np.array(info.get("adapter_cg_iters", 0.0), dtype=np.float64),
            adapter_cg_residual=np.array(info.get("adapter_cg_residual", 0.0), dtype=np.float64),
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
    adaptation: AdaptationTopologyCache | None = None
    replicator: ReplicatorLaplacianTopologyCache | None = None
    sobolev: SobolevTransportTopologyCache | None = None
    learned_topology_key: tuple[object, ...] | None = None
    learned_topology: TopologyV4 | None = None
    learned_model_key: tuple[object, ...] | None = None
    learned_model: CellMonitorV4 | None = None
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
        self.adaptation = None
        self.replicator = None
        self.sobolev = None
        self.learned_topology_key = None
        self.learned_topology = None
        self.misses += 1
        return self.cells, self.boundary_nodes

    def adaptation_cache(self) -> AdaptationTopologyCache:
        if self.adaptation is None:
            self.adaptation = AdaptationTopologyCache()
        return self.adaptation

    def replicator_cache(self) -> ReplicatorLaplacianTopologyCache:
        if self.replicator is None:
            self.replicator = ReplicatorLaplacianTopologyCache()
        return self.replicator

    def sobolev_cache(self) -> SobolevTransportTopologyCache:
        if self.sobolev is None:
            self.sobolev = SobolevTransportTopologyCache()
        return self.sobolev

    def learned_topology_for(self, points: torch.Tensor, cells: torch.Tensor, boundary_nodes: torch.Tensor) -> TopologyV4:
        key = (
            int(points.shape[0]),
            tuple(cells.shape),
            str(cells.device),
            str(points.dtype),
        )
        if self.learned_topology_key == key and self.learned_topology is not None:
            return self.learned_topology
        self.learned_topology_key = key
        self.learned_topology = TopologyV4.build(points, (cells,), boundary_nodes, dtype=points.dtype)
        return self.learned_topology

    def learned_model_for(
            self,
            *,
            weights_path: str,
            feature_mode: str,
            dtype: torch.dtype,
            device: torch.device,
    ) -> CellMonitorV4:
        key = (weights_path, feature_mode, str(dtype), str(device))
        if self.learned_model_key == key and self.learned_model is not None:
            return self.learned_model
        model = CellMonitorV4(in_dim=feature_dim_for_mode(feature_mode), hidden=64, depth=2)
        state = torch.load(weights_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(dtype=dtype, device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.learned_model_key = key
        self.learned_model = model
        return model


def adapt_coordinates(
    *,
    points_np: np.ndarray,
    cells_np: np.ndarray | None = None,
    triangles_np: np.ndarray | None = None,
    monitor_np: np.ndarray | None = None,
    cell_monitor_np: np.ndarray | None = None,
    learned_u_nodes_np: np.ndarray | None = None,
    learned_weights_path: str | None = None,
    learned_feature_mode: str = "full7",
    learned_recalibrate: bool = True,
    learned_blend: float = 1.0,
    learned_smoothing_steps: int = 0,
    learned_smoothing_weight: float = 0.5,
    steps: int,
    lr: float,
    profile: str = "regularized",
    dtype: str | np.dtype | torch.dtype = "float64",
    movement_weight: float | None = None,
    smoothness_weight: float | None = None,
    max_step_edge_stretch: float | None = None,
    min_step_edge_compression: float | None = None,
    topology_cache: AdapterTopologyCache | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if cells_np is None:
        if triangles_np is None:
            raise ValueError("adapt_coordinates requires cells_np or triangles_np")
        cells_np = triangles_np
    if cells_np.ndim != 2 or cells_np.shape[1] not in (3, 4):
        raise ValueError("cells_np must have shape (n_cells, 3) or (n_cells, 4)")

    if learned_u_nodes_np is not None and profile in {"replicator-laplace", "analytic-fast", "analytic-fast-v2", "sobolev-transport"}:
        raise ValueError(f"learned-v4 monitor requires the regularized gradient-descent adapter profile, not {profile}")
    if learned_u_nodes_np is not None and learned_weights_path is None:
        raise ValueError("learned_weights_path is required when learned_u_nodes_np is provided")

    if cell_monitor_np is None and learned_u_nodes_np is None:
        if monitor_np is None:
            raise ValueError("adapt_coordinates requires monitor_np or cell_monitor_np")
        cell_monitor_np = monitor_np[cells_np].mean(axis=1)
    if not points_np.flags.writeable:
        points_np = points_np.copy()
    if cell_monitor_np is not None and not cell_monitor_np.flags.writeable:
        cell_monitor_np = cell_monitor_np.copy()
    if learned_u_nodes_np is not None and not learned_u_nodes_np.flags.writeable:
        learned_u_nodes_np = learned_u_nodes_np.copy()
    torch_dtype = _torch_dtype(dtype)
    points = torch.as_tensor(points_np, dtype=torch_dtype)
    if topology_cache is None:
        cells = torch.as_tensor(cells_np, dtype=torch.long)
        boundary_nodes = torch.as_tensor(_boundary_nodes_from_cells(points_np.shape[0], cells_np), dtype=torch.bool)
    else:
        cells, boundary_nodes = topology_cache.tensors(points_np.shape[0], cells_np)
    mesh_state = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)

    if learned_u_nodes_np is not None:
        u_nodes = torch.as_tensor(learned_u_nodes_np, dtype=torch_dtype)
        topology = (
            TopologyV4.build(points, (cells,), boundary_nodes, dtype=torch_dtype)
            if topology_cache is None
            else topology_cache.learned_topology_for(points, cells, boundary_nodes)
        )
        model = (
            _load_learned_model(learned_weights_path, learned_feature_mode, torch_dtype, points.device)
            if topology_cache is None
            else topology_cache.learned_model_for(
                weights_path=learned_weights_path,
                feature_mode=learned_feature_mode,
                dtype=torch_dtype,
                device=points.device,
            )
        )
        if learned_recalibrate:
            with torch.no_grad():
                model.calibrate(compute_v4_features(points.detach(), topology, u_nodes, feature_mode=learned_feature_mode))

        baseline_monitor = None
        if cell_monitor_np is not None:
            cell_monitor_np = np.maximum(cell_monitor_np, 1.0e-12)
            baseline_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)
        blend = min(max(float(learned_blend), 0.0), 1.0)

        def monitor_fn(query_points: torch.Tensor, _cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
            features = compute_v4_features(query_points, topology, u_nodes, feature_mode=learned_feature_mode)
            learned_monitor = model(features, topology)
            learned_monitor = _smooth_learned_monitor(
                learned_monitor, topology,
                steps=learned_smoothing_steps,
                weight=learned_smoothing_weight,
            )
            if baseline_monitor is None or blend >= 1.0:
                return learned_monitor
            baseline = baseline_monitor.to(dtype=query_points.dtype, device=query_points.device)
            if blend <= 0.0:
                return baseline
            return (1.0 - blend) * baseline + blend * learned_monitor
    else:
        cell_monitor_np = np.maximum(cell_monitor_np, 1.0e-12)
        cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)

        def monitor_fn(query_points: torch.Tensor, _cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return cell_monitor

    if profile == "replicator-laplace":
        cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)
        result = adapt_replicator_laplacian(
            mesh_state,
            cell_monitor,
            ReplicatorLaplacianConfig(steps=steps),
            topology_cache=topology_cache.replicator_cache() if topology_cache is not None else None,
        )
    elif profile == "sobolev-transport":
        cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)
        result = adapt_sobolev_transport_map(
            mesh_state,
            cell_monitor,
            SobolevTransportConfig(
                steps=1 if steps > 0 else 0,
                max_step_edge_stretch=max_step_edge_stretch if max_step_edge_stretch is not None else 2.2,
                min_step_edge_compression=min_step_edge_compression if min_step_edge_compression is not None else 0.3,
                max_cg_iters=_env_int("DIFF_MESH_SOBOLV_MAX_CG_ITERS", 8),
                collect_step_diagnostics=False,
                freeze_metric=_env_bool("DIFF_MESH_SOBOLEV_FREEZE_METRIC")
                              or _env_bool("DIFF_MESH_SOBOLV_FREEZE_METRIC"),
            ),
            reference_points=points,
            topology_cache=topology_cache.sobolev_cache() if topology_cache is not None else None,
        )
    elif profile in {"analytic-fast", "analytic-fast-v2"}:
        cell_monitor = torch.as_tensor(cell_monitor_np, dtype=torch_dtype)
        config = _adaptation_config(profile, steps=steps, lr=lr)
        config = _apply_config_overrides(
            config,
            movement_weight=movement_weight,
            smoothness_weight=smoothness_weight,
            max_step_edge_stretch=max_step_edge_stretch,
            min_step_edge_compression=min_step_edge_compression,
        )
        result = adapt_monitor_weighted_area_analytic_fast(
            mesh_state,
            cell_monitor,
            config,
            reference_points=points,
            topology_cache=topology_cache.adaptation_cache() if topology_cache is not None else None,
        )
    else:
        config = _adaptation_config(profile, steps=steps, lr=lr)
        config = _apply_config_overrides(
            config,
            movement_weight=movement_weight,
            smoothness_weight=smoothness_weight,
            max_step_edge_stretch=max_step_edge_stretch,
            min_step_edge_compression=min_step_edge_compression,
        )
        result = adapt_monitor_weighted_area(
            mesh_state,
            monitor_fn,
            config,
            reference_points=points,
            topology_cache=topology_cache.adaptation_cache() if topology_cache is not None else None,
        )
    adapted = result.mesh.points.cpu().numpy().astype(points_np.dtype, copy=False)
    info = {
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "steps_completed": int(result.steps_completed),
        "early_stopped": bool(result.early_stopped),
    }
    if result.timings is not None:
        info.update({key: float(value) for key, value in result.timings.items()})
    return adapted, info


def _load_learned_model(weights_path: str, feature_mode: str, dtype: torch.dtype, device: torch.device) -> CellMonitorV4:
    model = CellMonitorV4(in_dim=feature_dim_for_mode(feature_mode), hidden=64, depth=2)
    state = torch.load(weights_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(dtype=dtype, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


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
    if profile in {"analytic-fast", "analytic-fast-v2"}:
        return AdaptationConfig(
            steps=steps,
            lr=lr,
            movement_weight=2.0e-3,
            smoothness_weight=5.0e-2,
            shape_weight=0.0,
            quality_barrier_weight=0.0,
            boundary_quality_barrier_weight=0.0,
            min_cell_quality=0.35,
            min_step_cell_quality=None,
            edge_length_weight=0.0,
            edge_length_barrier_weight=0.0,
            boundary_edge_length_barrier_weight=0.0,
            max_edge_stretch=1.8,
            min_edge_compression=0.4,
            max_step_edge_stretch=1.8 if profile == "analytic-fast-v2" else 2.2,
            min_step_edge_compression=0.45 if profile == "analytic-fast-v2" else 0.3,
            barrier_weight=0.0,
            grad_clip=0.25,
            collect_step_diagnostics=False,
            early_stopping_patience=12,
            early_stopping_min_delta=1.0e-3,
            early_stopping_relative=False,
        )
    raise ValueError(f"Unknown adapter profile: {profile}")


def _apply_config_overrides(
        config: AdaptationConfig,
        *,
        movement_weight: float | None,
        smoothness_weight: float | None,
        max_step_edge_stretch: float | None,
        min_step_edge_compression: float | None,
) -> AdaptationConfig:
    overrides = {}
    if movement_weight is not None:
        overrides["movement_weight"] = movement_weight
    if smoothness_weight is not None:
        overrides["smoothness_weight"] = smoothness_weight
    if max_step_edge_stretch is not None:
        overrides["max_step_edge_stretch"] = max_step_edge_stretch
    if min_step_edge_compression is not None:
        overrides["min_step_edge_compression"] = min_step_edge_compression
    return replace(config, **overrides) if overrides else config


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


def _npz_float(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in data:
        return None
    value = data[key]
    return float(value.item() if value.shape == () else value)


def _env_bool(name: str) -> bool:
    return os.environ.get(name) == "1"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _smooth_learned_monitor(monitor: torch.Tensor, topology: TopologyV4, *, steps: int, weight: float) -> torch.Tensor:
    steps = max(int(steps), 0)
    weight = min(max(float(weight), 0.0), 1.0)
    if steps == 0 or weight == 0.0:
        return monitor
    values = monitor
    for _ in range(steps):
        neighbor_mean = message_passing_mean_v4(values.unsqueeze(-1), topology).squeeze(-1)
        values = (1.0 - weight) * values + weight * neighbor_mean
    return values


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
