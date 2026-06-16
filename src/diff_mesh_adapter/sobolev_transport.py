from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from diff_mesh_adapter.adapt import (
    AdaptationResult,
    _edge_lengths_for_edges,
    _triangle_weighted_area_gradient,
    _unique_triangle_edges,
)
from diff_mesh_adapter.geometry import tetra_signed_volumes, triangle_signed_areas
from diff_mesh_adapter.mesh import MeshState

Tensor = torch.Tensor


@dataclass(frozen=True)
class SobolevTransportConfig:
    steps: int = 1
    beta0: float = 10.0
    beta1: float = 2.0e-2
    max_cg_iters: int = 8
    cg_rtol: float = 1.0e-6
    area_eps: float = 1.0e-12
    initial_alpha: float = 7.8125e-3
    backtrack_factor: float = 0.5
    max_backtracks: int = 12
    require_loss_decrease: bool = True
    max_step_edge_stretch: float | None = 2.2
    min_step_edge_compression: float | None = 0.3
    collect_step_diagnostics: bool = True
    freeze_metric: bool = False


@dataclass
class SobolevTransportTopologyCache:
    num_points: int | None = None
    cells: Tensor | None = None
    boundary_nodes: Tensor | None = None
    graph_edges: Tensor | None = None
    metric_lumped_mass: Tensor | None = None
    metric_stiffness: Tensor | None = None
    metric_diagonal: Tensor | None = None
    metric_beta0: float | None = None
    metric_beta1: float | None = None
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
        self.graph_edges = _unique_tetra_edges(cells) if cells.shape[1] == 4 else _unique_triangle_edges(cells)
        self.metric_lumped_mass = None
        self.metric_stiffness = None
        self.metric_diagonal = None
        self.metric_beta0 = None
        self.metric_beta1 = None
        self.misses += 1
        return self.cells, self.boundary_nodes, self.graph_edges

    def metric_tensors(
            self,
            points: Tensor,
            cells: Tensor,
            boundary_nodes: Tensor,
            measures: Tensor,
            config: SobolevTransportConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
                self.metric_lumped_mass is not None
                and self.metric_stiffness is not None
                and self.metric_diagonal is not None
                and self.metric_lumped_mass.device == points.device
                and self.metric_lumped_mass.dtype == points.dtype
                and self.metric_beta0 == float(config.beta0)
                and self.metric_beta1 == float(config.beta1)
        ):
            return self.metric_lumped_mass, self.metric_stiffness, self.metric_diagonal
        if cells.shape[1] == 4:
            lumped_mass = _tetra_lumped_mass(points, cells, measures, config.area_eps)
            stiffness = _tetra_stiffness_coefficients(points, cells, measures, config.area_eps)
        else:
            lumped_mass = _triangle_lumped_mass(points, cells, measures, config.area_eps)
            stiffness = _triangle_stiffness_coefficients(points, cells, measures, config.area_eps)
        diagonal = _sobolev_metric_diagonal(lumped_mass, stiffness, cells, boundary_nodes, config)
        self.metric_lumped_mass = lumped_mass
        self.metric_stiffness = stiffness
        self.metric_diagonal = diagonal
        self.metric_beta0 = float(config.beta0)
        self.metric_beta1 = float(config.beta1)
        return lumped_mass, stiffness, diagonal


def adapt_sobolev_transport_map(
        mesh: MeshState,
        cell_monitor: Tensor,
        config: SobolevTransportConfig | None = None,
        *,
        reference_points: Tensor | None = None,
        topology_cache: SobolevTransportTopologyCache | None = None,
) -> AdaptationResult:
    """Build one zero-shot Sobolev-gradient transport map for a 2D triangle mesh."""
    config = config or SobolevTransportConfig()
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
        raise ValueError("cell_monitor must contain one value per cell")

    is_3d = points.shape[1] == 3
    if is_3d:
        initial_signed = tetra_signed_volumes(points, cells).detach()
    else:
        initial_signed = triangle_signed_areas(points, cells).detach()
    if bool((initial_signed.abs() <= config.area_eps).any()):
        raise ValueError("Initial mesh contains degenerate cells with near-zero signed area")
    orientation = torch.sign(initial_signed).detach()

    timings = {
        "adapter_forward_loss": 0.0,
        "adapter_backward_or_grad": 0.0,
        "adapter_sobolev_solve": 0.0,
        "adapter_validation": 0.0,
        "adapter_step": 0.0,
    }

    forward_start = time.perf_counter()
    if is_3d:
        signed_measures = tetra_signed_volumes(points, cells)
    else:
        signed_measures = triangle_signed_areas(points, cells)
    oriented_measures = signed_measures * orientation
    weighted_measures = monitor * oriented_measures
    weighted_sum = weighted_measures.sum().clamp_min(config.area_eps)
    weighted_square_sum = weighted_measures.square().sum()
    n_cells_loss = max(int(weighted_measures.numel()), 1)
    initial_loss = (weighted_square_sum * float(n_cells_loss) / weighted_sum.square() - 1.0).detach()
    timings["adapter_forward_loss"] += time.perf_counter() - forward_start

    loss_history = [float(initial_loss.detach())]
    area_std_history = []
    min_area_history = []
    lr_history = []
    if config.collect_step_diagnostics:
        measures = oriented_measures.detach().abs()
        area_std_history.append(float(measures.std(unbiased=False)))
        min_area_history.append(float(oriented_measures.detach().min()))
        lr_history.append(0.0)

    if config.steps <= 0:
        return AdaptationResult(
            mesh=MeshState(points=points, cell_blocks=cell_blocks, boundary_nodes=boundary_nodes),
            loss_history=loss_history,
            area_std_history=area_std_history,
            min_area_history=min_area_history,
            lr_history=lr_history,
            stopped_step=0,
            early_stopped=False,
            timings=timings,
        )

    grad_start = time.perf_counter()
    if is_3d:
        grad = _tetra_weighted_volume_gradient(
            points, cells, orientation, monitor,
            weighted_measures, weighted_sum, weighted_square_sum, config.area_eps,
        )
    else:
        grad = _triangle_weighted_area_gradient(
            points, cells, orientation, monitor,
            weighted_measures, weighted_sum, weighted_square_sum, config.area_eps,
        )
    grad[boundary_nodes] = 0.0
    timings["adapter_backward_or_grad"] += time.perf_counter() - grad_start

    solve_start = time.perf_counter()
    if config.freeze_metric and topology_cache is not None:
        lumped_mass, stiffness, metric_diagonal = topology_cache.metric_tensors(
            points,
            cells,
            boundary_nodes,
            oriented_measures.abs(),
            config,
        )
    else:
        if is_3d:
            lumped_mass = _tetra_lumped_mass(points, cells, oriented_measures.abs(), config.area_eps)
            stiffness = _tetra_stiffness_coefficients(points, cells, oriented_measures.abs(), config.area_eps)
        else:
            lumped_mass = _triangle_lumped_mass(points, cells, oriented_measures.abs(), config.area_eps)
            stiffness = _triangle_stiffness_coefficients(points, cells, oriented_measures.abs(), config.area_eps)
        metric_diagonal = _sobolev_metric_diagonal(lumped_mass, stiffness, cells, boundary_nodes, config)
    rhs = -grad
    direction, cg_iters, cg_residual = _conjugate_gradient(
        lambda value: _apply_sobolev_metric(value, cells, boundary_nodes, lumped_mass, stiffness, config),
        rhs,
        boundary_nodes,
        metric_diagonal,
        max_iters=max(int(config.max_cg_iters), 1),
        rtol=max(float(config.cg_rtol), 0.0),
    )
    direction[boundary_nodes] = 0.0
    timings["adapter_sobolev_solve"] += time.perf_counter() - solve_start

    validation_start = time.perf_counter()
    reference_edge_lengths = _edge_lengths_for_edges(reference_points, graph_edges, config.area_eps)
    final_points = points
    final_loss = initial_loss
    accepted_alpha = 0.0
    alpha = float(config.initial_alpha)
    for _ in range(max(int(config.max_backtracks), 0) + 1):
        trial_points = points + alpha * direction
        trial_points[boundary_nodes] = points[boundary_nodes]
        if is_3d:
            trial_signed = tetra_signed_volumes(trial_points, cells) * orientation
        else:
            trial_signed = triangle_signed_areas(trial_points, cells) * orientation
        valid = bool(torch.isfinite(trial_signed).all()) and not bool((trial_signed <= config.area_eps).any())
        if valid:
            valid = _edge_step_is_valid(trial_points, graph_edges, reference_edge_lengths, config)
        if valid:
            if is_3d:
                trial_signed_m = tetra_signed_volumes(trial_points, cells)
            else:
                trial_signed_m = triangle_signed_areas(trial_points, cells)
            trial_oriented = trial_signed_m * orientation
            trial_weighted = monitor * trial_oriented
            trial_w_sum = trial_weighted.sum().clamp_min(config.area_eps)
            trial_w_sq_sum = trial_weighted.square().sum()
            trial_loss = (trial_w_sq_sum * float(n_cells_loss) / trial_w_sum.square() - 1.0)
            valid = bool(torch.isfinite(trial_loss))
            if valid and config.require_loss_decrease:
                valid = bool(trial_loss <= initial_loss)
            if valid:
                final_points = trial_points.detach()
                final_loss = trial_loss.detach()
                accepted_alpha = alpha
                break
        alpha *= float(config.backtrack_factor)
    timings["adapter_validation"] += time.perf_counter() - validation_start

    step_start = time.perf_counter()
    result_mesh = MeshState(points=final_points.detach(), cell_blocks=cell_blocks, boundary_nodes=boundary_nodes)
    timings["adapter_step"] += time.perf_counter() - step_start

    if is_3d:
        final_oriented = tetra_signed_volumes(result_mesh.points, cells) * orientation
    else:
        final_oriented = triangle_signed_areas(result_mesh.points, cells) * orientation
    loss_history.append(float(final_loss.detach()))
    if config.collect_step_diagnostics:
        final_measures = final_oriented.detach().abs()
        area_std_history.append(float(final_measures.std(unbiased=False)))
        min_area_history.append(float(final_oriented.detach().min()))
        lr_history.append(float(accepted_alpha))
    timings["adapter_cg_iters"] = float(cg_iters)
    timings["adapter_cg_residual"] = float(cg_residual)
    timings["adapter_alpha"] = float(accepted_alpha)

    return AdaptationResult(
        mesh=result_mesh,
        loss_history=loss_history,
        area_std_history=area_std_history,
        min_area_history=min_area_history,
        lr_history=lr_history,
        stopped_step=1 if accepted_alpha > 0.0 else 0,
        early_stopped=accepted_alpha == 0.0,
        timings=timings,
    )


def _require_triangle_mesh(mesh: MeshState) -> Tensor:
    if mesh.dim == 3:
        if len(mesh.cell_blocks) != 1 or mesh.cell_blocks[0].ndim != 2 or mesh.cell_blocks[0].shape[1] != 4:
            raise NotImplementedError("sobolev-transport adaptation currently supports single-block tetrahedral meshes only")
        return mesh.cell_blocks[0].detach().to(device=mesh.points.device)
    if mesh.dim != 2:
        raise NotImplementedError("sobolev-transport adaptation currently supports 2D and 3D meshes only")
    if len(mesh.cell_blocks) != 1 or mesh.cell_blocks[0].ndim != 2 or mesh.cell_blocks[0].shape[1] != 3:
        raise NotImplementedError("sobolev-transport adaptation currently supports single-block triangle meshes only")
    return mesh.cell_blocks[0].detach().to(device=mesh.points.device)


def _triangle_lumped_mass(points: Tensor, cells: Tensor, measures: Tensor, eps: float) -> Tensor:
    mass = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    cell_mass = measures.clamp_min(eps) / 3.0
    mass.index_add_(0, cells[:, 0], cell_mass)
    mass.index_add_(0, cells[:, 1], cell_mass)
    mass.index_add_(0, cells[:, 2], cell_mass)
    return mass.clamp_min(eps)


def _tetra_lumped_mass(points: Tensor, cells: Tensor, measures: Tensor, eps: float) -> Tensor:
    """Lumped P1 mass for tetrahedral mesh: m_i = sum_{K∋i} |K| / 4."""
    mass = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    cell_mass = measures.clamp_min(eps) / 4.0
    mass.index_add_(0, cells[:, 0], cell_mass)
    mass.index_add_(0, cells[:, 1], cell_mass)
    mass.index_add_(0, cells[:, 2], cell_mass)
    mass.index_add_(0, cells[:, 3], cell_mass)
    return mass.clamp_min(eps)


def _tetra_weighted_volume_gradient(
        points: Tensor,
        cells: Tensor,
        orientation: Tensor,
        monitor: Tensor,
        weighted_measures: Tensor,
        weighted_sum: Tensor,
        weighted_square_sum: Tensor,
        eps: float,
) -> Tensor:
    """Gradient of the equidistribution loss w.r.t. tetrahedral node coordinates.

    For signed volume V = (1/6) det[p1-p0, p2-p0, p3-p0]:
        ∂V/∂p1 = (1/6) (p2-p0) × (p3-p0)
        ∂V/∂p2 = (1/6) (p3-p0) × (p1-p0)
        ∂V/∂p3 = (1/6) (p1-p0) × (p2-p0)
        ∂V/∂p0 = -(∂V/∂p1 + ∂V/∂p2 + ∂V/∂p3)  (translation invariance)
    """
    vertices = points[cells]  # (n_cells, 4, 3)
    p0, p1, p2, p3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]

    n_cells = max(int(weighted_measures.numel()), 1)
    dloss_dweighted = (
            2.0
            * float(n_cells)
            * (weighted_measures * weighted_sum - weighted_square_sum)
            / weighted_sum.clamp_min(eps).pow(3)
    )
    dloss_dvolume = monitor * dloss_dweighted  # (n_cells,)
    oriented_factor = orientation * dloss_dvolume / 6.0  # (n_cells,)

    grad0 = torch.cross(p3 - p1, p2 - p1, dim=1) * oriented_factor[:, None]
    grad1 = torch.cross(p2 - p0, p3 - p0, dim=1) * oriented_factor[:, None]
    grad2 = torch.cross(p3 - p0, p1 - p0, dim=1) * oriented_factor[:, None]
    grad3 = torch.cross(p1 - p0, p2 - p0, dim=1) * oriented_factor[:, None]

    grad = torch.zeros_like(points)
    grad.index_add_(0, cells[:, 0], grad0)
    grad.index_add_(0, cells[:, 1], grad1)
    grad.index_add_(0, cells[:, 2], grad2)
    grad.index_add_(0, cells[:, 3], grad3)
    return grad


def _triangle_stiffness_coefficients(points: Tensor, cells: Tensor, measures: Tensor, eps: float) -> Tensor:
    vertices = points[cells]
    x = vertices[:, :, 0]
    y = vertices[:, :, 1]
    b = torch.stack((y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]), dim=1)
    c = torch.stack((x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]), dim=1)
    denominator = 4.0 * measures.clamp_min(eps)
    return (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / denominator[:, None, None]


def _tetra_stiffness_coefficients(points: Tensor, cells: Tensor, measures: Tensor, eps: float) -> Tensor:
    """P1 scalar stiffness coefficients for tetrahedral elements.

    Returns (n_cells, 4, 4) tensor where K[i,j] is the scalar stiffness entry
    K_ij^K = V * ∇φ_i · ∇φ_j for element K with signed volume V.

    The P1 gradient formula: g_i = 6V ∇φ_i (un-normalized gradient).
    K_ij = g_i · g_j / (36|V|).
    """
    vertices = points[cells]  # (n_cells, 4, 3)
    p0, p1, p2, p3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]

    g0 = -torch.cross(p2 - p1, p3 - p1, dim=1)
    g1 = torch.cross(p2 - p0, p3 - p0, dim=1)
    g2 = -torch.cross(p1 - p0, p3 - p0, dim=1)
    g3 = torch.cross(p1 - p0, p2 - p0, dim=1)

    g = torch.stack([g0, g1, g2, g3], dim=1)  # (n_cells, 4, 3)
    vol_abs = measures.abs().clamp_min(eps)
    return torch.bmm(g, g.transpose(1, 2)) / (36.0 * vol_abs[:, None, None])


def _unique_tetra_edges(cells: Tensor) -> Tensor:
    """Extract unique edges from tetrahedral connectivity."""
    edges = torch.cat(
        (
            cells[:, [0, 1]],
            cells[:, [0, 2]],
            cells[:, [0, 3]],
            cells[:, [1, 2]],
            cells[:, [1, 3]],
            cells[:, [2, 3]],
        ),
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def _apply_sobolev_metric(
        value: Tensor,
        cells: Tensor,
        boundary_nodes: Tensor,
        lumped_mass: Tensor,
        stiffness: Tensor,
        config: SobolevTransportConfig,
) -> Tensor:
    n_vertices_per_cell = cells.shape[1]
    result = float(config.beta0) * lumped_mass[:, None] * value
    if config.beta1 != 0.0:
        stiff_result = torch.zeros_like(value)
        local_values = value[cells]
        local_result = torch.bmm(stiffness, local_values)
        for local_i in range(n_vertices_per_cell):
            stiff_result.index_add_(0, cells[:, local_i], local_result[:, local_i])
        result = result + float(config.beta1) * stiff_result
    result[boundary_nodes] = value[boundary_nodes]
    return result


def _sobolev_metric_diagonal(
        lumped_mass: Tensor,
        stiffness: Tensor,
        cells: Tensor,
        boundary_nodes: Tensor,
        config: SobolevTransportConfig,
) -> Tensor:
    n_vertices_per_cell = cells.shape[1]
    diagonal = float(config.beta0) * lumped_mass
    if config.beta1 != 0.0:
        stiff_diagonal = torch.zeros_like(lumped_mass)
        for local_i in range(n_vertices_per_cell):
            stiff_diagonal.index_add_(0, cells[:, local_i], stiffness[:, local_i, local_i])
        diagonal = diagonal + float(config.beta1) * stiff_diagonal
    diagonal[boundary_nodes] = 1.0
    return diagonal.clamp_min(torch.finfo(lumped_mass.dtype).eps)


def _conjugate_gradient(
        apply_matrix,
        rhs: Tensor,
        boundary_nodes: Tensor,
        diagonal: Tensor,
        *,
        max_iters: int,
        rtol: float,
) -> tuple[Tensor, int, float]:
    x = torch.zeros_like(rhs)
    r = rhs - apply_matrix(x)
    r[boundary_nodes] = 0.0
    z = r / diagonal[:, None]
    z[boundary_nodes] = 0.0
    p = z.clone()
    rz_old = torch.sum(r * z)
    rs_old = torch.sum(r * r)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    residual = float(torch.sqrt(rs_old).detach() / rhs_norm)
    if residual <= rtol:
        return x, 0, residual

    iterations = 0
    for iterations in range(1, max_iters + 1):
        ap = apply_matrix(p)
        denom = torch.sum(p * ap)
        if bool(denom.abs() <= torch.finfo(rhs.dtype).tiny):
            break
        alpha = rz_old / denom
        x = x + alpha * p
        x[boundary_nodes] = 0.0
        r = r - alpha * ap
        r[boundary_nodes] = 0.0
        rs_new = torch.sum(r * r)
        residual = float(torch.sqrt(rs_new).detach() / rhs_norm)
        if residual <= rtol:
            break
        z = r / diagonal[:, None]
        z[boundary_nodes] = 0.0
        rz_new = torch.sum(r * z)
        beta = rz_new / rz_old.clamp_min(torch.finfo(rhs.dtype).tiny)
        p = z + beta * p
        p[boundary_nodes] = 0.0
        rz_old = rz_new
    return x, iterations, residual


def _edge_step_is_valid(
        points: Tensor,
        graph_edges: Tensor,
        reference_edge_lengths: Tensor,
        config: SobolevTransportConfig,
) -> bool:
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
