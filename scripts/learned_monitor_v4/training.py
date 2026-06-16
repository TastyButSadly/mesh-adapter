"""Training utilities for learned-monitor v4 on snapshot fields."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from diff_mesh_adapter.geometry import (
    cell_signed_measures,
    cell_signed_measures_and_shape_energy,
)

from .features import compute_v4_features
from .monitor import CellMonitorV4, gauge_normalize, message_passing_mean_v4
from .snapshots import SnapshotP1Field
from .topology import TopologyV4

Tensor = torch.Tensor

_QUAD_BARY = torch.tensor(
    [[0.5, 0.5, 0.0],
     [0.0, 0.5, 0.5],
     [0.5, 0.0, 0.5]],
    dtype=torch.float64,
)
_QUAD_W = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)


@dataclass(frozen=True)
class UnrollConfigV4:
    steps: int = 50
    lr: float = 8e-3
    movement_weight: float = 3e-4
    shape_weight: float = 1e-4
    barrier_weight: float = 1e-1
    grad_clip: float = 0.8
    area_eps: float = 1e-12
    feature_mode: str = "full7"


def sobolev_loss_v4(model: CellMonitorV4, features: Tensor, topology: TopologyV4,
                    sigma_pert: float = 0.05) -> Tensor:
    z = model.normalize(features)
    eps = sigma_pert * torch.randn_like(z)
    z_pert = z + eps
    z_nbr = message_passing_mean_v4(z, topology)
    z_pert_nbr = message_passing_mean_v4(z_pert, topology)
    raw = model.net(torch.cat([z, z_nbr], dim=-1)).squeeze(-1)
    raw_pert = model.net(torch.cat([z_pert, z_pert_nbr], dim=-1)).squeeze(-1)
    monitor = 1.0 + model.cap * torch.sigmoid(raw)
    monitor_pert = 1.0 + model.cap * torch.sigmoid(raw_pert)
    return (monitor_pert - monitor).square().mean()


def snapshot_interp_error(points: Tensor, topology: TopologyV4, field: SnapshotP1Field) -> Tensor:
    cells = topology.cell_blocks_flat
    if cells.shape[1] != 3:
        raise ValueError("snapshot_interp_error currently supports triangle meshes only")
    bary = _QUAD_BARY.to(dtype=points.dtype, device=points.device)
    weights = _QUAD_W.to(dtype=points.dtype, device=points.device)
    verts = points[cells]
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    area = 0.5 * cross.abs().clamp_min(1e-30)

    u_nodes = field.evaluate(points)
    u_at = u_nodes[cells]
    quad_points = torch.einsum("qi,cid->cqd", bary, verts)
    u_true = field.evaluate(quad_points.reshape(-1, points.shape[1])).reshape(cells.shape[0], -1)
    u_interp = torch.einsum("qi,ci->cq", bary, u_at)
    return (area * (weights.view(1, -1) * (u_true - u_interp).square()).sum(dim=-1)).sum()


def _inner_step_v4(
        points: Tensor,
        initial_points: Tensor,
        topology: TopologyV4,
        orientation: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        step: int,
        u_nodes: Tensor,
        monitor_model: CellMonitorV4,
        cfg: UnrollConfigV4,
        *,
        create_graph: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    signed_measures, shape_energy = cell_signed_measures_and_shape_energy(
        points, topology.cell_blocks_tuple, cfg.area_eps,
    )
    oriented = signed_measures * orientation
    measures = oriented.abs()
    target_measure = measures.detach().mean().clamp_min(cfg.area_eps)

    feats = compute_v4_features(points, topology, u_nodes, feature_mode=cfg.feature_mode)
    monitor_raw = monitor_model(feats, topology).clamp_min(cfg.area_eps)
    monitor = gauge_normalize(monitor_raw, measures.detach(), cfg.area_eps)

    weighted = monitor * measures
    eq_loss = ((weighted / weighted.detach().mean().clamp_min(cfg.area_eps) - 1.0) ** 2).mean()
    movement = ((points - initial_points) ** 2).sum(dim=1).mean()
    shape_loss = shape_energy.mean()
    barrier = (torch.relu(cfg.area_eps - oriented) / target_measure).square().mean()
    inner_loss = (
        eq_loss
        + cfg.movement_weight * movement
        + cfg.shape_weight * shape_loss
        + cfg.barrier_weight * barrier
    )

    grad = torch.autograd.grad(inner_loss, points, create_graph=create_graph)[0].clone()
    grad[topology.boundary_mask] = 0.0
    gnorm = torch.linalg.vector_norm(grad)
    scale = torch.clamp(cfg.grad_clip / (gnorm + 1e-6), max=1.0)
    grad = grad * scale

    exp_avg = 0.9 * exp_avg + 0.1 * grad
    exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad * grad
    bc1 = 1.0 - 0.9 ** step
    bc2_sqrt = math.sqrt(1.0 - 0.999 ** step)
    denom = torch.sqrt(exp_avg_sq + 1e-24) / bc2_sqrt + 1e-8
    update = (cfg.lr / bc1) * exp_avg / denom
    update = update.clone()
    update[topology.boundary_mask] = 0.0
    return points - update, exp_avg, exp_avg_sq


def unroll_inner_adaptation_v4(
        mesh_points: Tensor,
        topology: TopologyV4,
        u_nodes: Tensor,
        monitor_model: CellMonitorV4,
        cfg: UnrollConfigV4,
        *,
        backprop_last_k: int = 10,
) -> Tensor:
    initial_points = mesh_points.detach()
    u_nodes = u_nodes.detach().to(dtype=mesh_points.dtype, device=mesh_points.device)
    orientation = torch.sign(cell_signed_measures(initial_points, topology.cell_blocks_tuple).detach())
    points = mesh_points.detach().clone()
    exp_avg = torch.zeros_like(points)
    exp_avg_sq = torch.zeros_like(points)
    warmup = max(0, cfg.steps - backprop_last_k)

    for step in range(1, warmup + 1):
        p = points.detach().clone().requires_grad_(True)
        points, exp_avg, exp_avg_sq = _inner_step_v4(
            p, initial_points, topology, orientation,
            exp_avg.detach(), exp_avg_sq.detach(), step,
            u_nodes, monitor_model, cfg, create_graph=False,
        )
        points = points.detach()
        exp_avg = exp_avg.detach()
        exp_avg_sq = exp_avg_sq.detach()

    points = points.detach().clone().requires_grad_(True)
    exp_avg = exp_avg.detach()
    exp_avg_sq = exp_avg_sq.detach()
    for step in range(warmup + 1, cfg.steps + 1):
        points, exp_avg, exp_avg_sq = _inner_step_v4(
            points, initial_points, topology, orientation,
            exp_avg, exp_avg_sq, step,
            u_nodes, monitor_model, cfg, create_graph=True,
        )
    return points
