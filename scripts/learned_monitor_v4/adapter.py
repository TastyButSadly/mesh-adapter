"""Inference-side adapter compatible with diff_mesh_adapter.adapt_monitor_weighted_area.

Differences vs v2/v3 LearnedMonitorAdapter*:
  * Uses pre-built TopologyV4 (no per-call adjacency rebuild).
  * Inference path runs under no_grad and (optionally) fp32.
  * `topology` is the heavy precomputed object; pass it in to skip the build
    cost on every adapter instantiation.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .features import compute_v4_features
from .monitor import CellMonitorV4
from .topology import TopologyV4

Tensor = torch.Tensor


class LearnedMonitorAdapterV4:
    """Adapter object usable as the `monitor` argument of adapt_monitor_weighted_area.

    The training-time path expects an autograd-tracked u_nodes through features
    and the network. The inference path (default) detaches both and runs under
    no_grad.
    """

    def __init__(
            self,
            *,
            topology: TopologyV4,
            u_nodes: Tensor,
            weights_path: str | Path,
            in_dim: int = 7,
            hidden: int = 64,
            depth: int = 2,
            cap: float = 20.0,
            recalibrate: bool = True,
            calibration_points: Tensor | None = None,
            with_boundary: bool = True,
            feature_mode: str = "full7",
            track_monitor_grad: bool = True,
            dtype: torch.dtype | None = None,
    ) -> None:
        self.topology = topology
        self.u_nodes = u_nodes.detach().clone()
        self.with_boundary = with_boundary
        self.feature_mode = feature_mode
        self.track_monitor_grad = track_monitor_grad

        # Default to the dtype of the topology (and hence the mesh).
        if dtype is None:
            dtype = self.u_nodes.dtype
        self.u_nodes = self.u_nodes.to(dtype=dtype)

        self.model = CellMonitorV4(in_dim=in_dim, hidden=hidden, depth=depth, cap=cap).to(dtype=dtype)
        state = torch.load(Path(weights_path), weights_only=True, map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(dtype=dtype)

        if recalibrate:
            if calibration_points is None:
                raise ValueError("calibration_points is required when recalibrate=True")
            self.recalibrate(calibration_points)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def recalibrate(self, points: Tensor) -> None:
        """Recompute feat_mean/feat_std on the current u_nodes + points.

        Must be called BEFORE adaptation starts if recalibration is desired.
        """
        with torch.no_grad():
            feats0 = compute_v4_features(
                points.detach(), self.topology, self.u_nodes,
                with_boundary=self.with_boundary,
                feature_mode=self.feature_mode,
            )
            self.model.calibrate(feats0)

    def __call__(self, points: Tensor, cell_blocks=None) -> Tensor:
        """adapt_monitor_weighted_area calls this with (points, cell_blocks).
        cell_blocks arg is ignored — topology is already cached.
        """
        if self.track_monitor_grad:
            feats = compute_v4_features(
                points, self.topology, self.u_nodes,
                with_boundary=self.with_boundary,
                feature_mode=self.feature_mode,
            )
            return self.model(feats, self.topology)
        with torch.no_grad():
            feats = compute_v4_features(
                points.detach(), self.topology, self.u_nodes,
                with_boundary=self.with_boundary,
                feature_mode=self.feature_mode,
            )
            return self.model(feats, self.topology)
