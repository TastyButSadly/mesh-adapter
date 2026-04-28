from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch

from diff_mesh_adapter.geometry import (
    cell_centroids,
    cell_signed_measures,
    cell_signed_measures_and_shape_energy,
)
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 300
    lr: float = 1e-2
    movement_weight: float = 1e-3
    smoothness_weight: float = 0.0
    shape_weight: float = 1e-3
    quality_barrier_weight: float = 0.0
    boundary_quality_barrier_weight: float = 0.0
    min_cell_quality: float = 0.2
    min_step_cell_quality: float | None = None
    edge_length_weight: float = 0.0
    edge_length_barrier_weight: float = 0.0
    boundary_edge_length_barrier_weight: float = 0.0
    max_edge_stretch: float = 2.0
    min_edge_compression: float = 0.35
    max_step_edge_stretch: float | None = None
    min_step_edge_compression: float | None = None
    barrier_weight: float = 1e-2
    area_eps: float = 1e-12
    grad_clip: float | None = 1.0
    lr_shrink: float = 0.5
    min_lr: float = 1e-8
    restore_optimizer_state_on_invalid: bool = False
    store_history: bool = False
    history_stride: int = 1
    early_stopping_patience: int | None = 20
    early_stopping_min_delta: float = 1e-3
    early_stopping_relative: bool = True


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
    measures: Tensor
    signed_measures: Tensor
    oriented_measures: Tensor


ObjectiveFn = Callable[[Tensor, Tensor, Tensor, tuple[Tensor, ...], Tensor, Tensor, Tensor, AdaptationConfig], _LossParts]


def _movement_loss(points: Tensor, initial_points: Tensor) -> Tensor:
    return ((points - initial_points) ** 2).sum(dim=1).mean()


def _cell_displacement_smoothness_loss(points: Tensor, initial_points: Tensor, cell_blocks: tuple[Tensor, ...]) -> Tensor:
    displacement = points - initial_points
    losses = []
    for block in cell_blocks:
        cell_displacement = displacement[block]
        centered = cell_displacement - cell_displacement.mean(dim=1, keepdim=True)
        losses.append(centered.square().sum(dim=2).mean(dim=1))
    return torch.cat(losses, dim=0).mean()


def _regularization_loss(
    points: Tensor,
    initial_points: Tensor,
    reference_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    boundary_nodes: Tensor,
    shape_loss: Tensor,
    config: AdaptationConfig,
) -> Tensor:
    total = config.movement_weight * _movement_loss(points, initial_points) + config.shape_weight * shape_loss
    if config.smoothness_weight != 0.0:
        total = total + config.smoothness_weight * _cell_displacement_smoothness_loss(points, initial_points, cell_blocks)
    total = total + _edge_length_loss(points, reference_points, cell_blocks, boundary_nodes, config)
    return total


def _quality_barrier_loss(points: Tensor, cell_blocks: tuple[Tensor, ...], quality_weights: Tensor, config: AdaptationConfig) -> Tensor:
    if config.quality_barrier_weight == 0.0 and config.boundary_quality_barrier_weight == 0.0:
        return torch.zeros((), dtype=points.dtype, device=points.device)

    quality = _cell_qualities(points, cell_blocks, config.area_eps)
    min_quality = torch.as_tensor(config.min_cell_quality, dtype=points.dtype, device=points.device).clamp_min(config.area_eps)
    deficit = torch.relu(min_quality - quality) / min_quality
    normalized_penalty = deficit.square() / quality.clamp_min(config.area_eps)
    return (quality_weights * normalized_penalty).mean()


def _barrier_loss(oriented_measures: Tensor, target_measure: Tensor, config: AdaptationConfig) -> Tensor:
    return (torch.relu(config.area_eps - oriented_measures) / target_measure).square().mean()


def _make_loss_parts(total_loss: Tensor, signed_measures: Tensor, orientation: Tensor) -> _LossParts:
    oriented_measures = signed_measures * orientation
    return _LossParts(
        total=total_loss,
        measures=oriented_measures.abs(),
        signed_measures=signed_measures,
        oriented_measures=oriented_measures,
    )


def _area_equalization_objective(
    points: Tensor,
    initial_points: Tensor,
    reference_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    orientation: Tensor,
    quality_weights: Tensor,
    boundary_nodes: Tensor,
    config: AdaptationConfig,
) -> _LossParts:
    if config.shape_weight != 0.0:
        signed_measures, shape_energy = cell_signed_measures_and_shape_energy(points, cell_blocks, config.area_eps)
        shape_loss = shape_energy.mean()
    else:
        signed_measures = cell_signed_measures(points, cell_blocks)
        shape_loss = torch.zeros((), dtype=points.dtype, device=points.device)
    oriented_measures = signed_measures * orientation
    measures = oriented_measures.abs()
    target_measure = measures.detach().mean().clamp_min(config.area_eps)

    area_loss = ((measures / target_measure - 1.0) ** 2).mean()
    total_loss = (
        area_loss
        + _regularization_loss(points, initial_points, reference_points, cell_blocks, boundary_nodes, shape_loss, config)
        + _quality_barrier_loss(points, cell_blocks, quality_weights, config)
        + config.barrier_weight * _barrier_loss(oriented_measures, target_measure, config)
    )
    return _make_loss_parts(total_loss, signed_measures, orientation)


def _quality_repair_objective(
    points: Tensor,
    initial_points: Tensor,
    reference_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    orientation: Tensor,
    quality_weights: Tensor,
    boundary_nodes: Tensor,
    config: AdaptationConfig,
) -> _LossParts:
    signed_measures, shape_energy = cell_signed_measures_and_shape_energy(points, cell_blocks, config.area_eps)
    oriented_measures = signed_measures * orientation
    target_measure = oriented_measures.detach().abs().mean().clamp_min(config.area_eps)
    quality_loss = shape_energy.mean()
    total_loss = (
        quality_loss
        + _regularization_loss(
            points,
            initial_points,
            reference_points,
            cell_blocks,
            boundary_nodes,
            torch.zeros((), dtype=points.dtype, device=points.device),
            config,
        )
        + _quality_barrier_loss(points, cell_blocks, quality_weights, config)
        + config.barrier_weight * _barrier_loss(oriented_measures, target_measure, config)
    )
    return _make_loss_parts(total_loss, signed_measures, orientation)


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
        reference_points: Tensor,
        cell_blocks: tuple[Tensor, ...],
        orientation: Tensor,
        quality_weights: Tensor,
        boundary_nodes: Tensor,
        config: AdaptationConfig,
    ) -> _LossParts:
        if config.shape_weight != 0.0:
            signed_measures, shape_energy = cell_signed_measures_and_shape_energy(points, cell_blocks, config.area_eps)
            shape_loss = shape_energy.mean()
        else:
            signed_measures = cell_signed_measures(points, cell_blocks)
            shape_loss = torch.zeros((), dtype=points.dtype, device=points.device)
        oriented_measures = signed_measures * orientation
        measures = oriented_measures.abs()
        target_measure = measures.detach().mean().clamp_min(config.area_eps)
        monitor = monitor_fn(points, cell_blocks).clamp_min(config.area_eps)
        weighted_measures = monitor * measures
        weighted_loss = ((weighted_measures / weighted_measures.detach().mean().clamp_min(config.area_eps) - 1.0) ** 2).mean()
        total_loss = (
            weighted_loss
            + _regularization_loss(points, initial_points, reference_points, cell_blocks, boundary_nodes, shape_loss, config)
            + _quality_barrier_loss(points, cell_blocks, quality_weights, config)
            + config.barrier_weight * _barrier_loss(oriented_measures, target_measure, config)
        )
        return _make_loss_parts(total_loss, signed_measures, orientation)

    return objective


def _run_fixed_topology_optimization(
    mesh: MeshState,
    config: AdaptationConfig,
    objective_fn: ObjectiveFn,
    reference_points: Tensor | None = None,
) -> AdaptationResult:
    if config is None:
        config = AdaptationConfig()
    initial_points = mesh.points.detach().clone()
    if reference_points is None:
        reference_points = initial_points
    reference_points = reference_points.detach().clone().to(device=initial_points.device, dtype=initial_points.dtype)
    if reference_points.shape != initial_points.shape:
        raise ValueError("reference_points must have the same shape as mesh.points")
    points = initial_points.clone().detach().requires_grad_(True)
    cell_blocks = tuple(block.detach().clone().to(device=points.device) for block in mesh.cell_blocks)
    boundary_nodes = mesh.boundary_nodes.detach().clone().to(device=points.device)
    initial_signed = cell_signed_measures(points, cell_blocks).detach()
    if bool((initial_signed.abs() <= config.area_eps).any()):
        raise ValueError("Initial mesh contains degenerate cells with near-zero signed area")
    orientation = torch.sign(initial_signed).detach()
    quality_weights = _cell_quality_weights(cell_blocks, boundary_nodes, config).to(dtype=points.dtype, device=points.device)
    if config.min_step_cell_quality is not None:
        initial_quality = _cell_qualities(points, cell_blocks, config.area_eps).detach()
        if bool((initial_quality <= config.min_step_cell_quality).any()):
            raise ValueError("Initial mesh contains cells below min_step_cell_quality")
    _validate_initial_edge_ratios(initial_points, reference_points, cell_blocks, config)

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
        loss_parts = objective_fn(points, initial_points, reference_points, cell_blocks, orientation, quality_weights, boundary_nodes, config)
        loss_value = float(loss_parts.total.detach())
        loss_history.append(loss_value)
        area_std_history.append(float(loss_parts.measures.detach().std(unbiased=False)))
        min_area_history.append(float(loss_parts.oriented_measures.detach().min()))
        lr_history.append(current_lr)
        if points_history is not None and (step % history_stride == 0 or step == config.steps):
            points_history.append(points.detach().cpu().clone())
            if points_history_steps is not None:
                points_history_steps.append(step)

        if _is_early_stopping_improvement(loss_value, best_loss, config):
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

        previous_optimizer_state = copy.deepcopy(optimizer.state_dict()) if config.restore_optimizer_state_on_invalid else None
        optimizer.step()
        with torch.no_grad():
            points[boundary_nodes] = initial_points[boundary_nodes]
            oriented_after = cell_signed_measures(points, cell_blocks) * orientation
            invalid = (not bool(torch.isfinite(oriented_after).all())) or bool((oriented_after <= config.area_eps).any())
            if not invalid and config.min_step_cell_quality is not None:
                quality_after = _cell_qualities(points, cell_blocks, config.area_eps)
                invalid = (not bool(torch.isfinite(quality_after).all())) or bool((quality_after <= config.min_step_cell_quality).any())
            if not invalid and (config.max_step_edge_stretch is not None or config.min_step_edge_compression is not None):
                edge_ratios_after = _edge_ratios(points, reference_points, cell_blocks, config.area_eps)
                invalid = not bool(torch.isfinite(edge_ratios_after).all())
                if not invalid and config.max_step_edge_stretch is not None:
                    invalid = bool((edge_ratios_after >= config.max_step_edge_stretch).any())
                if not invalid and config.min_step_edge_compression is not None:
                    invalid = bool((edge_ratios_after <= config.min_step_edge_compression).any())
            if invalid:
                points.copy_(previous_points)
                if previous_optimizer_state is not None:
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


def adapt_cell_area_equalization(
    mesh: MeshState,
    config: AdaptationConfig | None = None,
    *,
    reference_points: Tensor | None = None,
) -> AdaptationResult:
    """Equalize fixed-topology 2D cell areas or 3D tetra volumes by moving non-boundary nodes."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _area_equalization_objective, reference_points)


def adapt_mesh_quality(
    mesh: MeshState,
    config: AdaptationConfig | None = None,
    *,
    reference_points: Tensor | None = None,
) -> AdaptationResult:
    """Improve fixed-topology mesh shape quality without changing connectivity."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _quality_repair_objective, reference_points)


def adapt_monitor_weighted_area(
    mesh: MeshState,
    monitor_fn: Callable[[Tensor, tuple[Tensor, ...]], Tensor],
    config: AdaptationConfig | None = None,
    *,
    reference_points: Tensor | None = None,
) -> AdaptationResult:
    """Equidistribute monitor-weighted cell areas."""
    return _run_fixed_topology_optimization(mesh, config or AdaptationConfig(), _monitor_weighted_area_objective(monitor_fn), reference_points)


def _is_early_stopping_improvement(loss_value: float, best_loss: float, config: AdaptationConfig) -> bool:
    if config.early_stopping_relative:
        return loss_value < best_loss * (1.0 - config.early_stopping_min_delta)
    return loss_value < best_loss - config.early_stopping_min_delta


def _edge_length_loss(
    points: Tensor,
    reference_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    boundary_nodes: Tensor,
    config: AdaptationConfig,
) -> Tensor:
    if (
        config.edge_length_weight == 0.0
        and config.edge_length_barrier_weight == 0.0
        and config.boundary_edge_length_barrier_weight == 0.0
    ):
        return torch.zeros((), dtype=points.dtype, device=points.device)

    losses = []
    for block in cell_blocks:
        current = _block_edge_lengths(points, block)
        reference = _block_edge_lengths(reference_points, block).clamp_min(config.area_eps)
        ratio = current / reference
        block_loss = torch.zeros_like(ratio)

        if config.edge_length_weight != 0.0:
            block_loss = block_loss + config.edge_length_weight * torch.log(ratio.clamp_min(config.area_eps)).square()

        if config.edge_length_barrier_weight != 0.0 or config.boundary_edge_length_barrier_weight != 0.0:
            stretch = torch.relu(ratio - config.max_edge_stretch) / max(config.max_edge_stretch, config.area_eps)
            compression = torch.relu(config.min_edge_compression - ratio) / max(config.min_edge_compression, config.area_eps)
            barrier = stretch.square() + compression.square()
            weights = torch.full_like(ratio, float(config.edge_length_barrier_weight))
            if config.boundary_edge_length_barrier_weight != 0.0:
                weights = weights + _block_edge_boundary_mask(boundary_nodes, block, points.shape[1]).to(dtype=ratio.dtype) * float(
                    config.boundary_edge_length_barrier_weight
                )
            block_loss = block_loss + weights * barrier

        losses.append(block_loss.reshape(-1))

    return torch.cat(losses).mean()


def _validate_initial_edge_ratios(
    points: Tensor,
    reference_points: Tensor,
    cell_blocks: tuple[Tensor, ...],
    config: AdaptationConfig,
) -> None:
    if config.max_step_edge_stretch is None and config.min_step_edge_compression is None:
        return

    ratios = _edge_ratios(points, reference_points, cell_blocks, config.area_eps).detach()
    if config.max_step_edge_stretch is not None and bool((ratios >= config.max_step_edge_stretch).any()):
        raise ValueError("Initial mesh contains edges above max_step_edge_stretch")
    if config.min_step_edge_compression is not None and bool((ratios <= config.min_step_edge_compression).any()):
        raise ValueError("Initial mesh contains edges below min_step_edge_compression")


def _edge_ratios(points: Tensor, reference_points: Tensor, cell_blocks: tuple[Tensor, ...], eps: float) -> Tensor:
    ratios = []
    for block in cell_blocks:
        current = _block_edge_lengths(points, block)
        reference = _block_edge_lengths(reference_points, block).clamp_min(eps)
        ratios.append((current / reference).reshape(-1))
    return torch.cat(ratios)


def _block_edge_lengths(points: Tensor, block: Tensor) -> Tensor:
    vertices = points[block]
    if points.shape[1] == 3 and block.shape[1] == 4:
        pairs = _tetra_edge_pairs(block.device)
        return torch.linalg.norm(vertices[:, pairs[:, 0]] - vertices[:, pairs[:, 1]], dim=2)

    next_vertices = torch.roll(vertices, shifts=-1, dims=1)
    return torch.linalg.norm(next_vertices - vertices, dim=2)


def _block_edge_boundary_mask(boundary_nodes: Tensor, block: Tensor, dim: int) -> Tensor:
    node_boundary = boundary_nodes[block]
    if dim == 3 and block.shape[1] == 4:
        pairs = _tetra_edge_pairs(block.device)
        return node_boundary[:, pairs[:, 0]] | node_boundary[:, pairs[:, 1]]
    return node_boundary | torch.roll(node_boundary, shifts=-1, dims=1)


def _tetra_edge_pairs(device: torch.device) -> Tensor:
    return torch.tensor(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=torch.long,
        device=device,
    )


def _cell_quality_weights(cell_blocks: tuple[Tensor, ...], boundary_nodes: Tensor, config: AdaptationConfig) -> Tensor:
    weights = []
    for block in cell_blocks:
        weight = torch.full(
            (block.shape[0],),
            float(config.quality_barrier_weight),
            dtype=torch.float64,
            device=block.device,
        )
        if config.boundary_quality_barrier_weight != 0.0:
            touches_boundary = boundary_nodes[block].any(dim=1)
            weight = weight + touches_boundary.to(dtype=weight.dtype) * float(config.boundary_quality_barrier_weight)
        weights.append(weight)
    return torch.cat(weights, dim=0)


def _cell_qualities(points: Tensor, cell_blocks: tuple[Tensor, ...], eps: float) -> Tensor:
    _, shape_energy = cell_signed_measures_and_shape_energy(points, cell_blocks, eps)
    return 1.0 / (1.0 + shape_energy)
