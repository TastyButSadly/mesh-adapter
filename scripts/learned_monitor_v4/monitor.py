"""v4 monitor network and helpers.

Same architecture as v2/v3:
  features (C, F) -> concat with neighbor-mean -> MLP -> 1 + cap * sigmoid

Difference vs v2: scatter-mean indices come pre-built from TopologyV4
(no rebuild per call).
"""
from __future__ import annotations

import torch
from torch import nn

from .topology import TopologyV4

Tensor = torch.Tensor


def message_passing_mean_v4(features: Tensor, topology: TopologyV4) -> Tensor:
    """Mean aggregation over neighbors using cached topology.

    Falls back to own features for isolated cells.
    """
    src_u = topology.adj_src_undirected
    dst_u = topology.adj_dst_undirected
    n_cells = topology.n_cells
    n_feat = features.shape[1]
    accum = torch.zeros((n_cells, n_feat), dtype=features.dtype, device=features.device)
    if src_u.numel() > 0:
        accum.index_add_(0, dst_u, features[src_u])
    counts = topology.adj_degree.to(dtype=features.dtype)
    has_nbr = counts > 0
    mean_nbr = accum / counts.clamp_min(1.0)
    return torch.where(has_nbr, mean_nbr, features)


class CellMonitorV4(nn.Module):
    """Per-cell monitor network. Same shape as CellMonitorMPNN(v2/v3),
    but takes a TopologyV4 directly in forward (no separate adjacency arg)."""

    def __init__(self, in_dim: int = 7, hidden: int = 64, depth: int = 2,
                 cap: float = 20.0):
        super().__init__()
        self.in_dim = in_dim
        self.cap = cap
        mp_in = 2 * in_dim
        layers: list[nn.Module] = [nn.Linear(mp_in, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("feat_mean", torch.zeros(in_dim, dtype=torch.float64))
        self.register_buffer("feat_std", torch.ones(in_dim, dtype=torch.float64))

    def calibrate(self, sample_features: Tensor) -> None:
        with torch.no_grad():
            self.feat_mean.copy_(sample_features.mean(dim=0).to(self.feat_mean.dtype))
            self.feat_std.copy_(sample_features.std(dim=0).clamp_min(1e-6).to(self.feat_std.dtype))

    def normalize(self, features: Tensor) -> Tensor:
        mean = self.feat_mean.to(dtype=features.dtype, device=features.device)
        std = self.feat_std.to(dtype=features.dtype, device=features.device)
        return (features - mean) / std

    def forward(self, features: Tensor, topology: TopologyV4) -> Tensor:
        z = self.normalize(features)
        z_nbr = message_passing_mean_v4(z, topology)
        z_cat = torch.cat([z, z_nbr], dim=-1)
        raw = self.net(z_cat).squeeze(-1)
        return 1.0 + self.cap * torch.sigmoid(raw)


def gauge_normalize(monitor: Tensor, cell_areas: Tensor, eps: float = 1e-12) -> Tensor:
    weighted = (monitor * cell_areas).sum().clamp_min(eps)
    domain = cell_areas.sum().clamp_min(eps)
    return monitor * (domain / weighted)


def smoothness_loss_on_log(monitor: Tensor, topology: TopologyV4) -> Tensor:
    adj = topology.adjacency
    if adj.shape[0] == 0:
        return monitor.new_zeros(())
    a = monitor[adj[:, 0]]
    b = monitor[adj[:, 1]]
    return (torch.log(a) - torch.log(b)).square().mean()
