from __future__ import annotations

from dataclasses import dataclass

import torch

from diff_mesh_adapter.adapt import AdaptationResult
from diff_mesh_adapter.geometry import cell_signed_measures, triangle_signed_areas
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


@dataclass
class ReplicatorLaplacianTopologyCache:
    num_points: int | None = None
    tris: Tensor | None = None
    boundary_nodes: Tensor | None = None
    hits: int = 0
    misses: int = 0

    def tensors(self, mesh: MeshState) -> tuple[Tensor, Tensor]:
        if any(block.shape[1] != 3 for block in mesh.cell_blocks):
            raise NotImplementedError("replicator-laplace adaptation currently supports triangle cells only")
        tris = torch.cat(tuple(block.detach().to(device=mesh.points.device) for block in mesh.cell_blocks), dim=0)
        boundary_nodes = mesh.boundary_nodes.detach().to(device=mesh.points.device)
        if (
                self.num_points == mesh.num_points
                and self.tris is not None
                and self.tris.shape == tris.shape
                and self.tris.device == tris.device
                and torch.equal(self.tris, tris)
                and self.boundary_nodes is not None
                and self.boundary_nodes.shape == boundary_nodes.shape
                and torch.equal(self.boundary_nodes, boundary_nodes)
        ):
            self.hits += 1
            return self.tris, self.boundary_nodes

        self.num_points = mesh.num_points
        self.tris = tris
        self.boundary_nodes = boundary_nodes
        self.misses += 1
        return self.tris, self.boundary_nodes


@dataclass(frozen=True)
class ReplicatorLaplacianConfig:
    steps: int = 12
    r: float = 1.25
    mu: float = 1.0e-8
    dt: float = 0.2
    tol_var: float = 1.0e-3
    tol_slack: float = 0.30
    alpha_lift: float = 4.0
    smooth_relax: float = 1.0
    lift_iters: int = 8
    dt_min: float = 1.0e-5
    area_eps: float = 1.0e-12
    min_area_fraction: float = 0.08
    max_area_ratio: float = 100.0
    max_area_ratio_growth: float = 6.0
    max_backtracks: int = 8
    collect_step_diagnostics: bool = True


def adapt_replicator_laplacian(
        mesh: MeshState,
        cell_indicator: Tensor,
        config: ReplicatorLaplacianConfig | None = None,
        topology_cache: ReplicatorLaplacianTopologyCache | None = None,
) -> AdaptationResult:
    """Move a 2D triangular fixed-topology mesh with replicator mass flow and Laplacian lift."""
    config = config or ReplicatorLaplacianConfig()
    _validate_mesh(mesh)
    points = mesh.points.detach().clone()
    if topology_cache is None:
        cell_blocks = tuple(block.detach().to(device=points.device) for block in mesh.cell_blocks)
        boundary_nodes = mesh.boundary_nodes.detach().to(device=points.device)
        tris = torch.cat(cell_blocks, dim=0)
    else:
        tris, boundary_nodes = topology_cache.tensors(mesh)
        cell_blocks = (tris,)
    indicator = cell_indicator.detach().to(device=points.device, dtype=points.dtype).reshape(-1).clamp_min(
        config.area_eps)
    if indicator.shape != (tris.shape[0],):
        raise ValueError("cell_indicator must contain one value per cell")

    initial_signed = cell_signed_measures(points, cell_blocks).detach()
    if bool((initial_signed.abs() <= config.area_eps).any()):
        raise ValueError("Initial mesh contains degenerate cells with near-zero signed area")
    orientation = torch.sign(initial_signed).detach()

    loss_history: list[float] = []
    area_std_history: list[float] = []
    min_area_history: list[float] = []
    lr_history: list[float] = []

    measures = (cell_signed_measures(points, cell_blocks) * orientation).detach()
    if bool((measures <= config.area_eps).any()):
        raise ValueError("Initial mesh contains inverted or near-degenerate cells")
    min_allowed_area = measures.min() * config.min_area_fraction
    max_allowed_area_ratio = min(
        float(config.max_area_ratio),
        float(measures.max() / measures.min().clamp_min(config.area_eps)) * config.max_area_ratio_growth,
    )
    m = _normalized_mass(measures)
    p, p_bar, var, normalized_var = _pressure_stats(indicator, m, config)
    _append_diagnostics(loss_history, area_std_history, min_area_history, lr_history, normalized_var, measures,
                        config.dt, config)

    stopped_step = 0
    early_stopped = False
    for step in range(1, config.steps + 1):
        if normalized_var < config.tol_var:
            early_stopped = True
            break

        dt_try = config.dt
        accepted = False
        while True:
            delta = p - p_bar
            scale = float(delta.abs().max())
            if scale <= 0.0:
                accepted = True
                trial_points = points.clone()
                trial_m = m.clone()
                trial_p = p
                trial_p_bar = p_bar
                trial_var = var
                trial_normalized_var = normalized_var
                trial_measures = measures
                break

            dt_eff = dt_try / scale
            m_trial = m * (1.0 - dt_eff * delta)
            if bool((m_trial <= config.area_eps).any()) or not bool(torch.isfinite(m_trial).all()):
                dt_try *= 0.5
                if dt_try < config.dt_min:
                    break
                continue
            m_trial = m_trial / m_trial.sum().clamp_min(config.area_eps)

            trial_points = points
            omega = (m / m_trial.clamp_min(config.area_eps)).pow(config.alpha_lift)
            denominator = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
            denominator.index_add_(0, tris[:, 0], omega)
            denominator.index_add_(0, tris[:, 1], omega)
            denominator.index_add_(0, tris[:, 2], omega)
            inv_denominator = denominator.clamp_min(config.area_eps).reciprocal().unsqueeze(1)
            for _ in range(max(int(config.lift_iters), 1)):
                trial_points = _laplacian_smooth_step(
                    trial_points,
                    tris,
                    omega,
                    inv_denominator,
                    boundary_nodes,
                    orientation,
                    relax=config.smooth_relax,
                    area_eps=config.area_eps,
                    max_backtracks=config.max_backtracks,
                )

            trial_measures = (cell_signed_measures(trial_points, cell_blocks) * orientation).detach()
            if bool((trial_measures <= config.area_eps).any()) or not bool(torch.isfinite(trial_measures).all()):
                dt_try *= 0.5
                if dt_try < config.dt_min:
                    break
                continue
            trial_area_ratio = float(trial_measures.max() / trial_measures.min().clamp_min(config.area_eps))
            if (
                    bool((trial_measures < min_allowed_area).any())
                    or trial_area_ratio > max_allowed_area_ratio
            ):
                dt_try *= 0.5
                if dt_try < config.dt_min:
                    break
                continue

            trial_m = _normalized_mass(trial_measures)
            trial_p, trial_p_bar, trial_var, trial_normalized_var = _pressure_stats(indicator, trial_m, config)
            if trial_normalized_var <= (1.0 + config.tol_slack) * normalized_var or dt_try <= config.dt_min:
                accepted = True
                break
            dt_try *= 0.5
            if dt_try < config.dt_min:
                break

        if not accepted:
            break

        points = trial_points.detach()
        m = trial_m.detach()
        p = trial_p.detach()
        p_bar = trial_p_bar.detach()
        var = trial_var.detach()
        normalized_var = trial_normalized_var
        measures = trial_measures.detach()
        stopped_step = step
        _append_diagnostics(loss_history, area_std_history, min_area_history, lr_history, normalized_var, measures,
                            dt_try, config)

    result_mesh = MeshState(points=points, cell_blocks=cell_blocks, boundary_nodes=boundary_nodes)
    return AdaptationResult(
        mesh=result_mesh,
        loss_history=loss_history,
        area_std_history=area_std_history,
        min_area_history=min_area_history,
        lr_history=lr_history,
        stopped_step=stopped_step,
        early_stopped=early_stopped,
    )


def _validate_mesh(mesh: MeshState) -> None:
    if mesh.dim != 2:
        raise NotImplementedError("replicator-laplace adaptation currently supports 2D meshes only")
    if any(block.shape[1] != 3 for block in mesh.cell_blocks):
        raise NotImplementedError("replicator-laplace adaptation currently supports triangle cells only")


def _normalized_mass(measures: Tensor) -> Tensor:
    return measures / measures.sum().clamp_min(torch.finfo(measures.dtype).eps)


def _pressure_stats(
        indicator: Tensor,
        m: Tensor,
        config: ReplicatorLaplacianConfig,
) -> tuple[Tensor, Tensor, Tensor, float]:
    p = config.r * indicator * m.pow(config.r - 1.0) - config.mu / m.clamp_min(config.area_eps)
    p_bar = torch.sum(m * p) / m.sum().clamp_min(config.area_eps)
    var = torch.sum(m * (p - p_bar).square()) / m.sum().clamp_min(config.area_eps)
    p_scale = torch.sum(m * p.abs()) / m.sum().clamp_min(config.area_eps)
    normalized_var = float(var / p_scale.square().clamp_min(torch.finfo(m.dtype).tiny))
    return p, p_bar, var, normalized_var


def _laplacian_smooth_step(
        points: Tensor,
        tris: Tensor,
        omega: Tensor,
        inv_denominator: Tensor,
        boundary_nodes: Tensor,
        orientation: Tensor,
        *,
        relax: float,
        area_eps: float,
        max_backtracks: int,
) -> Tensor:
    centroids = (points[tris[:, 0]] + points[tris[:, 1]] + points[tris[:, 2]]) / 3.0
    weighted_centroids = omega.unsqueeze(1) * centroids
    numerator = torch.zeros_like(points)
    numerator.index_add_(0, tris[:, 0], weighted_centroids)
    numerator.index_add_(0, tris[:, 1], weighted_centroids)
    numerator.index_add_(0, tris[:, 2], weighted_centroids)
    target = numerator * inv_denominator
    target[boundary_nodes] = points[boundary_nodes]

    current_relax = float(relax)
    if current_relax == 1.0:
        oriented = triangle_signed_areas(target, tris) * orientation
        if bool((oriented > area_eps).all()):
            return target
        current_relax = 0.5
    for _ in range(max_backtracks + 1):
        trial = (1.0 - current_relax) * points + current_relax * target
        oriented = triangle_signed_areas(trial, tris) * orientation
        if bool((oriented > area_eps).all()):
            return trial
        current_relax *= 0.5
    return points.clone()


def _append_diagnostics(
        loss_history: list[float],
        area_std_history: list[float],
        min_area_history: list[float],
        lr_history: list[float],
        normalized_var: float,
        measures: Tensor,
        dt: float,
        config: ReplicatorLaplacianConfig,
) -> None:
    loss_history.append(float(normalized_var))
    if config.collect_step_diagnostics:
        area_std_history.append(float(measures.detach().std(unbiased=False)))
        min_area_history.append(float(measures.detach().min()))
        lr_history.append(float(dt))
