"""Learned monitor v4 — optimized for large meshes.

Key design changes vs v2/v3:

  * Vectorized topology build (numpy, no dict-of-strings).
  * Vectorized features (no Python-loop over cell_blocks).
  * Pre-computed scatter-mean indices stored as buffers (no per-step rebuild).
  * Optional fp32 inference path with no autograd graph.
  * Topology cache (TopologyV4) so the heavy graph-construction work is done
    once per (mesh.points, cell_blocks) pair and reused.
  * Backward-compat: same 7 features as v3 (G + I) by default; same MPNN
    architecture (mean aggregation, sigmoid output, gauge-normalized).

Public API:
    TopologyV4(points, cell_blocks, boundary_mask)
        -> .adjacency, .scatter_mean_target_idx, .scatter_mean_source_idx, ...

    compute_v4_features(points, topology, u_nodes, *, with_boundary=True) -> (C, F)

    CellMonitorV4(in_dim, hidden, depth, cap)
        -> nn.Module with .forward(features, topology) -> (C,)

    LearnedMonitorAdapterV4(points, cell_blocks, boundary_mask, u_nodes,
                            weights_path, *, dtype=fp64, recalibrate=True)
        -> callable adapter compatible with adapt_monitor_weighted_area.
"""

from .topology import TopologyV4, build_cell_adjacency_fast
from .features import (
    compute_v4_features,
    feature_dim_for_mode,
    per_cell_p1_gradient_v4,
    recover_cell_hessian_frobenius_v4,
)
from .monitor import CellMonitorV4, gauge_normalize, smoothness_loss_on_log
from .adapter import LearnedMonitorAdapterV4
from .inference import FrozenLearnedMonitorResult, adapt_with_learned_frozen_monitor, predict_cell_monitor
from .quality import QualityGate, passes_quality_gate, triangle_mesh_quality
from .snapshots import ReferenceSnapshotDataset, SnapshotDataset, SnapshotP1Field, SnapshotSample
from .training import (
    UnrollConfigV4,
    snapshot_interp_error,
    sobolev_loss_v4,
    unroll_inner_adaptation_v4,
)

__all__ = [
    "TopologyV4",
    "build_cell_adjacency_fast",
    "compute_v4_features",
    "feature_dim_for_mode",
    "per_cell_p1_gradient_v4",
    "recover_cell_hessian_frobenius_v4",
    "CellMonitorV4",
    "gauge_normalize",
    "smoothness_loss_on_log",
    "LearnedMonitorAdapterV4",
    "FrozenLearnedMonitorResult",
    "adapt_with_learned_frozen_monitor",
    "predict_cell_monitor",
    "QualityGate",
    "passes_quality_gate",
    "triangle_mesh_quality",
    "SnapshotDataset",
    "ReferenceSnapshotDataset",
    "SnapshotP1Field",
    "SnapshotSample",
    "UnrollConfigV4",
    "snapshot_interp_error",
    "sobolev_loss_v4",
    "unroll_inner_adaptation_v4",
]
