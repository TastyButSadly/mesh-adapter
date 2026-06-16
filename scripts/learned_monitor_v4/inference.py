"""Production inference helpers for learned-monitor v4.

The production path predicts a cell monitor once on the fixed input mesh, then
passes that frozen monitor to a robust mesh mover. This matches the real adapter
use case better than recomputing neural features inside every mesh-optimization
step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from diff_mesh_adapter import AdaptationConfig, MeshState
from diff_mesh_adapter.adapt import AdaptationResult, adapt_monitor_weighted_area
from diff_mesh_adapter.geometry import cell_signed_measures
from diff_mesh_adapter.replicator import (
    ReplicatorLaplacianConfig,
    ReplicatorLaplacianTopologyCache,
    adapt_replicator_laplacian,
)

from .features import compute_v4_features, feature_dim_for_mode
from .monitor import CellMonitorV4, gauge_normalize
from .topology import TopologyV4

Tensor = torch.Tensor


@dataclass
class FrozenLearnedMonitorResult:
    result: AdaptationResult
    cell_monitor: Tensor
    predict_wall_s: float
    adapt_wall_s: float


def predict_cell_monitor(
        *,
        topology: TopologyV4,
        points: Tensor,
        u_nodes: Tensor,
        weights_path: str | Path | None = None,
        model: CellMonitorV4 | None = None,
        feature_mode: str = "full7",
        with_boundary: bool = True,
        hidden: int = 64,
        depth: int = 2,
        cap: float = 20.0,
        recalibrate: bool = True,
        dtype: torch.dtype | None = None,
        monitor_min: float = 1.0e-6,
        monitor_max: float | None = None,
        smoothing_steps: int = 0,
        smoothing_relax: float = 0.25,
) -> Tensor:
    """Predict a frozen per-cell monitor on the initial mesh."""
    if model is None:
        if weights_path is None:
            raise ValueError("Either model or weights_path is required")
        in_dim = feature_dim_for_mode(feature_mode, with_boundary=with_boundary)
        model = CellMonitorV4(in_dim=in_dim, hidden=hidden, depth=depth, cap=cap)
        state = torch.load(Path(weights_path), weights_only=True, map_location="cpu")
        model.load_state_dict(state)
    if dtype is None:
        dtype = points.dtype
    model = model.to(dtype=dtype, device=points.device)
    model.eval()
    points_i = points.detach().to(dtype=dtype)
    u_i = u_nodes.detach().to(dtype=dtype, device=points.device)

    with torch.no_grad():
        features = compute_v4_features(
            points_i, topology, u_i,
            with_boundary=with_boundary,
            feature_mode=feature_mode,
        )
        if recalibrate:
            model.calibrate(features)
        monitor = model(features, topology)
        areas = cell_signed_measures(points_i, topology.cell_blocks_tuple).abs().detach()
        monitor = gauge_normalize(monitor.clamp_min(monitor_min), areas)
        if monitor_max is not None:
            monitor = monitor.clamp(max=monitor_max)
            monitor = gauge_normalize(monitor.clamp_min(monitor_min), areas)
        if smoothing_steps > 0:
            monitor = _smooth_monitor(
                monitor, topology,
                steps=smoothing_steps,
                relax=smoothing_relax,
                monitor_min=monitor_min,
                monitor_max=monitor_max,
            )
            monitor = gauge_normalize(monitor.clamp_min(monitor_min), areas)
    return monitor.detach()


def adapt_with_learned_frozen_monitor(
        mesh: MeshState,
        topology: TopologyV4,
        u_nodes: Tensor,
        *,
        weights_path: str | Path | None = None,
        model: CellMonitorV4 | None = None,
        mover: str = "replicator-laplace",
        feature_mode: str = "full7",
        recalibrate: bool = True,
        dtype: torch.dtype | None = None,
        monitor_max: float | None = None,
        smoothing_steps: int = 0,
        replicator_config: ReplicatorLaplacianConfig | None = None,
        adaptation_config: AdaptationConfig | None = None,
        replicator_cache: ReplicatorLaplacianTopologyCache | None = None,
) -> FrozenLearnedMonitorResult:
    """Predict once, then adapt with a frozen monitor."""
    start = time.perf_counter()
    cell_monitor = predict_cell_monitor(
        topology=topology,
        points=mesh.points,
        u_nodes=u_nodes,
        weights_path=weights_path,
        model=model,
        feature_mode=feature_mode,
        recalibrate=recalibrate,
        dtype=dtype,
        monitor_max=monitor_max,
        smoothing_steps=smoothing_steps,
    )
    predict_wall = time.perf_counter() - start

    start = time.perf_counter()
    if mover == "replicator-laplace":
        result = adapt_replicator_laplacian(
            mesh,
            cell_monitor,
            replicator_config or ReplicatorLaplacianConfig(),
            topology_cache=replicator_cache,
        )
    elif mover == "regularized":
        frozen = cell_monitor.detach()

        def monitor_fn(points: Tensor, cell_blocks=None) -> Tensor:
            return frozen.to(dtype=points.dtype, device=points.device)

        result = adapt_monitor_weighted_area(
            mesh,
            monitor_fn,
            adaptation_config or AdaptationConfig(),
        )
    else:
        raise ValueError(f"Unknown mover {mover!r}; expected 'replicator-laplace' or 'regularized'")
    adapt_wall = time.perf_counter() - start
    return FrozenLearnedMonitorResult(
        result=result,
        cell_monitor=cell_monitor,
        predict_wall_s=predict_wall,
        adapt_wall_s=adapt_wall,
    )


def _smooth_monitor(
        monitor: Tensor,
        topology: TopologyV4,
        *,
        steps: int,
        relax: float,
        monitor_min: float,
        monitor_max: float | None,
) -> Tensor:
    src = topology.adj_src_undirected
    dst = topology.adj_dst_undirected
    if src.numel() == 0:
        return monitor
    out = monitor
    for _ in range(steps):
        accum = torch.zeros_like(out)
        accum.index_add_(0, dst, out[src])
        counts = topology.adj_degree.squeeze(-1).to(dtype=out.dtype, device=out.device).clamp_min(1.0)
        nbr = accum / counts
        out = (1.0 - relax) * out + relax * nbr
        out = out.clamp_min(monitor_min)
        if monitor_max is not None:
            out = out.clamp(max=monitor_max)
    return out
