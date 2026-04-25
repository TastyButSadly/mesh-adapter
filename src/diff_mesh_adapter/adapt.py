from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch

from diff_mesh_adapter.geometry import cell_centroids, cell_signed_areas, normalized_edge_length_variance
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 300
    lr: float = 1e-2
    movement_weight: float = 1e-3
    shape_weight: float = 1e-3
    barrier_weight: float = 1e-2
    area_eps: float = 1e-12
    grad_clip: float | None = 1.0
    lr_shrink: float = 0.5
    min_lr: float = 1e-8
    store_history: bool = False
    history_stride: int = 1
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 1e-3


ObjectiveFn = Callable[[Tensor, Tensor, tuple[Tensor, ...], Tensor, AdaptationConfig], Tensor]


@dataclass
class AdaptationResult:
    mesh: MeshState
    loss_history: list[float]
    area_std_history: list[float]
    min_area_history: list[float]
    lr_history: list[float]
    points_history: list[Tensor] | None = None
    points_history_steps: list[int] | None = None
    stopped_step: int = 0
    early_stopped: bool = False

    @property
    def initial_loss(self) -> float:
        return self.loss_history[0]

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]

    @property
    def steps_completed(self) -> int:
        return self.stopped_step


@dataclass
class _LossParts:
    total: Tensor
    areas: Tensor
    signed_areas: Tensor
    oriented_areas: Tensor


def _movement_loss(points: Tensor, initial_points: Tensor) -> Tensor:
    return ((points - initial_points) ** 2).sum(dim=1).mean()


def _barrier_loss(oriented_areas: Tensor, target_area: Tensor, config: AdaptationConfig) -> Tensor:
    return (torch.relu(config.area_eps - oriented_areas) / target_area).square().mean()


def _optional_shape_loss(points: Tensor, cell_blocks: tuple[Tensor, ...], config: AdaptationConfig) -> Tensor:
    if config.shape_weight != 0.0:
        return normalized_edge_length_variance(points, cell_blocks)
    return torch.zeros((), dtype=points.dtype, device=points.device)


def _area_equalization_objective(
    points: Tensor,
    initial_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    orientation: Tensor,
    config: AdaptationConfig,
) -> Tensor:
    signed_areas = cell_signed_areas(points, cell_blocks)
    oriented_areas = signed_areas * orientation
    areas = oriented_areas.abs()
    target_area = areas.detach().mean().clamp_min(config.area_eps)

    area_loss = ((areas / target_area - 1.0) ** 2).mean()
    return (
        area_loss
        + config.movement_weight * _movement_loss(points, initial_points)
        + config.shape_weight * _optional_shape_loss(points, cell_blocks, config)
        + config.barrier_weight * _barrier_loss(oriented_areas, target_area, config)
    )


def _quality_repair_objective(
    points: Tensor,
    initial_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    orientation: Tensor,
    config: AdaptationConfig,
) -> Tensor:
    oriented_areas = cell_signed_areas(points, cell_blocks) * orientation
    target_area = oriented_areas.detach().abs().mean().clamp_min(config.area_eps)
    quality_loss = normalized_edge_length_variance(points, cell_blocks)
    return (
        quality_loss
        + config.movement_weight * _movement_loss(points, initial_points)
        + config.barrier_weight * _barrier_loss(oriented_areas, target_area, config)
    )


def gaussian_cell_monitor(
    points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    *,
    center: tuple[float, float] = (0.65, 0.55),
    sigma: float = 0.16,
    alpha: float = 12.0,
) -> Tensor:
    centroids = cell_centroids(points, cell_blocks)
    center_tensor = torch.tensor(center, dtype=points.dtype, device=points.device)
    dist2 = ((centroids - center_tensor) ** 2).sum(dim=1)
    return 1.0 + alpha * torch.exp(-dist2 / (2.0 * sigma**2))


def _monitor_weighted_area_objective(
    monitor_fn: Callable[[Tensor, tuple[Tensor, ...]], Tensor],
) -> ObjectiveFn:
    def objective(
        points: Tensor,
        initial_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        orientation: Tensor,
        config: AdaptationConfig,
    ) -> Tensor:
        oriented_areas = cell_signed_areas(points, cell_blocks) * orientation
        areas = oriented_areas.abs()
        target_area = areas.detach().mean().clamp_min(config.area_eps)
        monitor = monitor_fn(points, cell_blocks).clamp_min(config.area_eps)
        weighted_areas = monitor * areas
        weighted_loss = ((weighted_areas / weighted_areas.detach().mean().clamp_min(config.area_eps) - 1.0) ** 2).mean()
        return (
            weighted_loss
            + config.movement_weight * _movement_loss(points, initial_points)
            + config.shape_weight * _optional_shape_loss(points, cell_blocks, config)
            + config.barrier_weight * _barrier_loss(oriented_areas, target_area, config)
        )

    return objective


def _run_fixed_topology_optimization(
    mesh: MeshState,
    config: AdaptationConfig,
    objective_fn: ObjectiveFn,
) -> AdaptationResult:
    if config is None:
        config = AdaptationConfig()
    if mesh.dim != 2:
        raise NotImplementedError("V1 adaptation supports 2D meshes only")

    initial_points = mesh.points.detach().clone()
    points = initial_points.clone().detach().requires_grad_(True)
    cell_blocks = tuple(block.detach().clone().to(device=points.device) for block in mesh.cell_blocks)
    boundary_nodes = mesh.boundary_nodes.detach().clone().to(device=points.device)
    initial_signed = cell_signed_areas(points, cell_blocks).detach()
    if bool((initial_signed.abs() <= config.area_eps).any()):
        raise ValueError("Initial mesh contains degenerate cells with near-zero signed area")
    orientation = torch.sign(initial_signed).detach()

    optimizer = torch.optim.Adam([points], lr=config.lr)
    current_lr = config.lr

    loss_history: list[float] = []
    area_std_history: list[float] = []
    min_area_history: list[float] = []
    lr_history: list[float] = []
    points_history: list[Tensor] | None = [] if config.store_history else None
    points_history_steps: list[int] | None = [] if config.store_history else None
    history_stride = max(int(config.history_stride), 1)
    best_loss = float("inf")
    stale_steps = 0
    stopped_step = config.steps
    early_stopped = False

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = objective_fn(points, initial_points, cell_blocks, orientation, config)
        signed_areas = cell_signed_areas(points, cell_blocks)
        oriented_areas = signed_areas * orientation
        areas = oriented_areas.abs()
        loss_parts = _LossParts(total=total_loss, areas=areas, signed_areas=signed_areas, oriented_areas=oriented_areas)
        loss_value = float(loss_parts.total.detach())
        loss_history.append(loss_value)
        area_std_history.append(float(loss_parts.areas.detach().std(unbiased=False)))
        min_area_history.append(float(loss_parts.oriented_areas.detach().min()))
        lr_history.append(current_lr)
        if points_history is not None and (step % history_stride == 0 or step == config.steps):
            points_history.append(points.detach().cpu().clone())
            if points_history_steps is not None:
                points_history_steps.append(step)

        if loss_value < best_loss * (1.0 - config.early_stopping_min_delta):
            best_loss = loss_value
            stale_steps = 0
        else:
            stale_steps += 1

        if step == config.steps:
            stopped_step = step
            break

        if config.early_stopping_patience is not None and stale_steps >= config.early_stopping_patience:
            stopped_step = step
            early_stopped = True
            if points_history is not None and (points_history_steps is None or points_history_steps[-1] != step):
                points_history.append(points.detach().cpu().clone())
                if points_history_steps is not None:
                    points_history_steps.append(step)
            break

        previous_points = points.detach().clone()
        loss_parts.total.backward()
        if points.grad is not None:
            points.grad[boundary_nodes] = 0.0
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([points], config.grad_clip)

        previous_optimizer_state = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        with torch.no_grad():
            points[boundary_nodes] = initial_points[boundary_nodes]
            oriented_after = cell_signed_areas(points, cell_blocks) * orientation
            invalid = (not bool(torch.isfinite(oriented_after).all())) or bool((oriented_after <= config.area_eps).any())
            if invalid:
                points.copy_(previous_points)
                optimizer.load_state_dict(previous_optimizer_state)
                current_lr = max(current_lr * config.lr_shrink, config.min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

    final_mesh = MeshState(
        points=points.detach(),
        cell_blocks=tuple(block.detach() for block in cell_blocks),
        boundary_nodes=boundary_nodes.detach(),
    )
    return AdaptationResult(
        mesh=final_mesh,
        loss_history=loss_history,
        area_std_history=area_std_history,
        min_area_history=min_area_history,
        lr_history=lr_history,
        points_history=points_history,
        points_history_steps=points_history_steps,
        stopped_step=stopped_step,
        early_stopped=early_stopped,
    )


def adapt_cell_area_equalization(mesh: MeshState, config: AdaptationConfig | None = None) -> AdaptationResult:
    """Equalize fixed-topology 2D cell areas by moving non-boundary nodes."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _area_equalization_objective)


def adapt_mesh_quality(mesh: MeshState, config: AdaptationConfig | None = None) -> AdaptationResult:
    """Improve fixed-topology mesh shape quality without changing connectivity."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _quality_repair_objective)


def adapt_monitor_weighted_area(
    mesh: MeshState,
    monitor_fn: Callable[[Tensor, tuple[Tensor, ...]], Tensor],
    config: AdaptationConfig | None = None,
) -> AdaptationResult:
    """Equidistribute monitor-weighted cell areas."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _monitor_weighted_area_objective(monitor_fn))
