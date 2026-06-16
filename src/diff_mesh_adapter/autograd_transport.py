from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from diff_mesh_adapter.adapt import (
    AdaptationResult,
    _edge_lengths_for_edges,
    _triangle_weighted_area_loss_state,
    _unique_triangle_edges,
)
from diff_mesh_adapter.geometry import triangle_signed_areas
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


@dataclass(frozen=True)
class AutogradTransportConfig:
    steps: int = 4
    max_control_points: int = 256
    basis: str = "compact-rbf"
    rbf_radius_factor: float = 1.5
    initial_alpha: float = 1.0e-2
    backtrack_factor: float = 0.5
    max_backtracks: int = 10
    area_eps: float = 1.0e-12
    require_loss_decrease: bool = True
    max_step_edge_stretch: float | None = 2.2
    min_step_edge_compression: float | None = 0.3
    collect_step_diagnostics: bool = True


@dataclass
class AutogradTransportTopologyCache:
    num_points: int | None = None
    cells: Tensor | None = None
    boundary_nodes: Tensor | None = None
    graph_edges: Tensor | None = None
    hits: int = 0
    misses: int = 0

    def tensors(self, mesh: MeshState) -> tuple[Tensor, Tensor, Tensor]:
        cells = _require_triangle_mesh(mesh)
        boundary_nodes = mesh.boundary_nodes.detach().to(device=mesh.points.device)
        if (
                self.num_points == mesh.num_points
                and self.cells is not None
                and self.cells.shape == cells.shape
                and self.cells.device == cells.device
                and torch.equal(self.cells, cells)
                and self.boundary_nodes is not None
                and self.boundary_nodes.shape == boundary_nodes.shape
                and torch.equal(self.boundary_nodes, boundary_nodes)
                and self.graph_edges is not None
        ):
            self.hits += 1
            return self.cells, self.boundary_nodes, self.graph_edges

        self.num_points = mesh.num_points
        self.cells = cells
        self.boundary_nodes = boundary_nodes
        self.graph_edges = _unique_triangle_edges(cells)
        self.misses += 1
        return self.cells, self.boundary_nodes, self.graph_edges


def adapt_autograd_transport_map(
        mesh: MeshState,
        cell_monitor: Tensor,
        config: AutogradTransportConfig | None = None,
        *,
        reference_points: Tensor | None = None,
        topology_cache: AutogradTransportTopologyCache | None = None,
) -> AdaptationResult:
    """Adapt a 2D triangle mesh through a low-dimensional autograd transport map."""
    config = config or AutogradTransportConfig()
    points = mesh.points.detach().clone()
    if topology_cache is None:
        cells = _require_triangle_mesh(mesh)
        boundary_nodes = mesh.boundary_nodes.detach().to(device=points.device)
        graph_edges = _unique_triangle_edges(cells)
    else:
        cells, boundary_nodes, graph_edges = topology_cache.tensors(mesh)
    cell_blocks = (cells,)

    if reference_points is None:
        reference_points = points
    else:
        reference_points = reference_points.detach().clone().to(device=points.device, dtype=points.dtype)
    if reference_points.shape != points.shape:
        raise ValueError("reference_points must have the same shape as mesh.points")

    monitor = cell_monitor.detach().to(device=points.device, dtype=points.dtype).reshape(-1).clamp_min(
        config.area_eps)
    if monitor.shape != (cells.shape[0],):
        raise ValueError("cell_monitor must contain one value per triangle")

    initial_signed = triangle_signed_areas(points, cells).detach()
    if bool((initial_signed.abs() <= config.area_eps).any()):
        raise ValueError("Initial mesh contains degenerate cells with near-zero signed area")
    orientation = torch.sign(initial_signed).detach()

    timings = {
        "adapter_forward_loss": 0.0,
        "adapter_backward_or_grad": 0.0,
        "adapter_transport_basis": 0.0,
        "adapter_validation": 0.0,
        "adapter_step": 0.0,
    }

    basis_start = time.perf_counter()
    weights = _transport_weights(points, boundary_nodes, config)
    timings["adapter_transport_basis"] += time.perf_counter() - basis_start
    if weights.shape[1] == 0 or config.steps <= 0:
        initial_loss = _triangle_weighted_area_loss_state(points, cells, orientation, monitor, config.area_eps)[0]
        return AdaptationResult(
            mesh=MeshState(points=points, cell_blocks=cell_blocks, boundary_nodes=boundary_nodes),
            loss_history=[float(initial_loss.detach())],
            area_std_history=[],
            min_area_history=[],
            lr_history=[],
            stopped_step=0,
            early_stopped=config.steps > 0,
            timings=timings,
        )

    forward_start = time.perf_counter()
    initial_loss, oriented_measures = _loss_and_measures(points, cells, orientation, monitor, config)
    timings["adapter_forward_loss"] += time.perf_counter() - forward_start

    loss_history = [float(initial_loss.detach())]
    area_std_history = []
    min_area_history = []
    lr_history = []
    if config.collect_step_diagnostics:
        area_std_history.append(float(oriented_measures.detach().abs().std(unbiased=False)))
        min_area_history.append(float(oriented_measures.detach().min()))
        lr_history.append(0.0)

    theta = torch.zeros(weights.shape[1], points.shape[1], dtype=points.dtype, device=points.device)
    current_points = points
    current_loss = initial_loss.detach()
    accepted_steps = 0
    last_alpha = 0.0
    bbox_scale = _bbox_scale(points, config.area_eps)
    reference_edge_lengths = _edge_lengths_for_edges(reference_points, graph_edges, config.area_eps)

    for _ in range(max(int(config.steps), 0)):
        grad_start = time.perf_counter()
        theta_var = theta.detach().clone().requires_grad_(True)
        trial_points = _apply_transport(points, weights, theta_var, boundary_nodes)
        loss, _ = _loss_and_measures(trial_points, cells, orientation, monitor, config)
        loss.backward()
        grad = theta_var.grad.detach() if theta_var.grad is not None else torch.zeros_like(theta)
        timings["adapter_backward_or_grad"] += time.perf_counter() - grad_start
        if not bool(torch.isfinite(grad).all()) or bool((grad.abs().max() <= torch.finfo(points.dtype).eps)):
            break

        direction = -grad
        node_direction = weights @ direction
        node_direction[boundary_nodes] = 0.0
        max_norm = torch.linalg.norm(node_direction, dim=1).max().clamp_min(torch.finfo(points.dtype).eps)
        direction = direction / max_norm

        validation_start = time.perf_counter()
        accepted_theta = theta
        accepted_points = current_points
        accepted_loss = current_loss
        accepted_alpha = 0.0
        alpha = float(config.initial_alpha) * bbox_scale
        for _backtrack in range(max(int(config.max_backtracks), 0) + 1):
            candidate_theta = theta + alpha * direction
            candidate_points = _apply_transport(points, weights, candidate_theta, boundary_nodes)
            valid = _candidate_is_valid(candidate_points, cells, orientation, graph_edges, reference_edge_lengths,
                                        config)
            if valid:
                candidate_loss = _triangle_weighted_area_loss_state(
                    candidate_points,
                    cells,
                    orientation,
                    monitor,
                    config.area_eps,
                )[0]
                valid = bool(torch.isfinite(candidate_loss))
                if valid and config.require_loss_decrease:
                    valid = bool(candidate_loss <= current_loss)
                if valid:
                    accepted_theta = candidate_theta.detach()
                    accepted_points = candidate_points.detach()
                    accepted_loss = candidate_loss.detach()
                    accepted_alpha = alpha
                    break
            alpha *= float(config.backtrack_factor)
        timings["adapter_validation"] += time.perf_counter() - validation_start

        if accepted_alpha <= 0.0:
            break
        step_start = time.perf_counter()
        theta = accepted_theta
        current_points = accepted_points
        current_loss = accepted_loss
        accepted_steps += 1
        last_alpha = accepted_alpha
        timings["adapter_step"] += time.perf_counter() - step_start

    final_oriented = triangle_signed_areas(current_points, cells) * orientation
    loss_history.append(float(current_loss.detach()))
    if config.collect_step_diagnostics:
        area_std_history.append(float(final_oriented.detach().abs().std(unbiased=False)))
        min_area_history.append(float(final_oriented.detach().min()))
        lr_history.append(float(last_alpha))
    timings["adapter_alpha"] = float(last_alpha)
    timings["adapter_control_points"] = float(weights.shape[1])

    return AdaptationResult(
        mesh=MeshState(points=current_points.detach(), cell_blocks=cell_blocks, boundary_nodes=boundary_nodes),
        loss_history=loss_history,
        area_std_history=area_std_history,
        min_area_history=min_area_history,
        lr_history=lr_history,
        stopped_step=accepted_steps,
        early_stopped=accepted_steps < max(int(config.steps), 0),
        timings=timings,
    )


def _loss_and_measures(
        points: Tensor,
        cells: Tensor,
        orientation: Tensor,
        monitor: Tensor,
        config: AutogradTransportConfig,
) -> tuple[Tensor, Tensor]:
    loss, oriented_measures = _triangle_weighted_area_loss_state(
        points,
        cells,
        orientation,
        monitor,
        config.area_eps,
    )[:2]
    return loss, oriented_measures


def _apply_transport(points: Tensor, weights: Tensor, theta: Tensor, boundary_nodes: Tensor) -> Tensor:
    displacement = weights @ theta
    displacement[boundary_nodes] = 0.0
    return points + displacement


def _transport_weights(points: Tensor, boundary_nodes: Tensor, config: AutogradTransportConfig) -> Tensor:
    interior = torch.nonzero(~boundary_nodes, as_tuple=False).reshape(-1)
    if interior.numel() == 0 or config.max_control_points <= 0:
        return torch.empty(points.shape[0], 0, dtype=points.dtype, device=points.device)
    control_indices = _select_control_indices(points, interior, int(config.max_control_points))
    centers = points[control_indices]
    sigma = _bbox_scale(points, config.area_eps) * float(config.rbf_radius_factor) / max(
        float(control_indices.numel()) ** 0.5,
        1.0,
    )
    sigma = max(sigma, config.area_eps)
    distances = torch.cdist(points, centers)
    if config.basis == "global-rbf":
        weights = torch.exp(-distances.square() / (2.0 * sigma * sigma))
    elif config.basis == "compact-rbf":
        radius = max(sigma, config.area_eps)
        scaled = (distances / radius).clamp_max(1.0)
        weights = (1.0 - scaled).pow(4) * (4.0 * scaled + 1.0)
        weights = torch.where(distances <= radius, weights, torch.zeros_like(weights))
        empty_support = weights.sum(dim=1) <= config.area_eps
        if bool(empty_support.any()):
            nearest = torch.argmin(distances[empty_support], dim=1)
            weights[empty_support] = 0.0
            weights[empty_support, nearest] = 1.0
    else:
        raise ValueError(f"Unknown autograd transport basis: {config.basis}")
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(config.area_eps)
    weights[boundary_nodes] = 0.0
    return weights


def _select_control_indices(points: Tensor, interior: Tensor, max_control_points: int) -> Tensor:
    if interior.numel() <= max_control_points:
        return interior
    selected = torch.empty(max_control_points, dtype=interior.dtype, device=interior.device)
    center = points[interior].mean(dim=0, keepdim=True)
    dist2 = ((points[interior] - center) ** 2).sum(dim=1)
    selected[0] = interior[torch.argmax(dist2)]
    min_dist2 = ((points[interior] - points[selected[0]]) ** 2).sum(dim=1)
    for idx in range(1, max_control_points):
        selected[idx] = interior[torch.argmax(min_dist2)]
        next_dist2 = ((points[interior] - points[selected[idx]]) ** 2).sum(dim=1)
        min_dist2 = torch.minimum(min_dist2, next_dist2)
    return selected


def _candidate_is_valid(
        points: Tensor,
        cells: Tensor,
        orientation: Tensor,
        graph_edges: Tensor,
        reference_edge_lengths: Tensor,
        config: AutogradTransportConfig,
) -> bool:
    oriented = triangle_signed_areas(points, cells) * orientation
    if not bool(torch.isfinite(oriented).all()) or bool((oriented <= config.area_eps).any()):
        return False
    if config.max_step_edge_stretch is None and config.min_step_edge_compression is None:
        return True
    ratios = _edge_lengths_for_edges(points, graph_edges, config.area_eps) / reference_edge_lengths
    if not bool(torch.isfinite(ratios).all()):
        return False
    if config.max_step_edge_stretch is not None and bool((ratios >= config.max_step_edge_stretch).any()):
        return False
    if config.min_step_edge_compression is not None and bool((ratios <= config.min_step_edge_compression).any()):
        return False
    return True


def _bbox_scale(points: Tensor, eps: float) -> float:
    width = points.max(dim=0).values - points.min(dim=0).values
    return float(torch.linalg.vector_norm(width).clamp_min(eps))


def _require_triangle_mesh(mesh: MeshState) -> Tensor:
    if mesh.dim != 2:
        raise NotImplementedError("autograd transport adaptation currently supports 2D meshes only")
    if len(mesh.cell_blocks) != 1 or mesh.cell_blocks[0].ndim != 2 or mesh.cell_blocks[0].shape[1] != 3:
        raise NotImplementedError("autograd transport adaptation currently supports single-block triangle meshes only")
    return mesh.cell_blocks[0].detach().to(device=mesh.points.device)
