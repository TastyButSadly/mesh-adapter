"""Vectorized v4 features.

Same semantics as v3's 7 features (6 v2 + boundary-incident flag),
but computed in one pass over a flattened (C, K) cell tensor — no Python
loop over `cell_blocks`. Identical math, just less host overhead.

Features (per cell, dim-invariant):
  f1 = u_centroid
  f2 = h * |grad u_h|
  f3 = h^2 * |H_recon|_F
  f4 = std(u@verts) / range(u@all verts)
  f5 = max_edge / min_edge       (capped at 1e3)
  f6 = h * |grad u_h| / (|u_centroid|+eps)
  f7 = boundary-incident flag (0/1)   -- only if topology.boundary_cell is used.
"""
from __future__ import annotations

import torch

from .topology import TopologyV4

Tensor = torch.Tensor


def per_cell_p1_gradient_v4(
        points: Tensor,
        u_nodes: Tensor,
        cells: Tensor,
        *,
        ridge: float = 1e-9,
) -> Tensor:
    """Constant per-cell gradient of the P1 interpolant of u_nodes.

    Args:
      points: (N, d)
      u_nodes: (N,) scalar nodal field.
      cells: (C, d+1) long tensor of vertex indices.
    Returns:
      grad: (C, d)
    """
    verts = points[cells]               # (C, K, d)
    v0 = verts[:, 0]                    # (C, d)
    edges = verts[:, 1:] - v0.unsqueeze(1)   # (C, d, d)
    u_at = u_nodes[cells]               # (C, K)
    du = u_at[:, 1:] - u_at[:, 0:1]     # (C, d)

    d = edges.shape[-1]
    I = torch.eye(d, dtype=edges.dtype, device=edges.device).expand_as(edges)
    grad = torch.linalg.solve(edges + ridge * I, du.unsqueeze(-1)).squeeze(-1)
    return grad


def _nodal_average_from_cells(
        per_cell_values: Tensor,
        cells: Tensor,
        n_nodes: int,
) -> Tensor:
    """Scatter-mean (C, K_out) -> (N, K_out)."""
    n_cells, kdim = per_cell_values.shape
    nodes_per_cell = cells.shape[1]
    flat_idx = cells.reshape(-1)
    flat_val = per_cell_values.unsqueeze(1).expand(-1, nodes_per_cell, -1).reshape(-1, kdim)
    nodal_sum = torch.zeros((n_nodes, kdim), dtype=per_cell_values.dtype,
                            device=per_cell_values.device)
    nodal_sum.index_add_(0, flat_idx, flat_val)
    ones = torch.ones((flat_idx.shape[0], 1), dtype=per_cell_values.dtype,
                      device=per_cell_values.device)
    nodal_count = torch.zeros((n_nodes, 1), dtype=per_cell_values.dtype,
                              device=per_cell_values.device)
    nodal_count.index_add_(0, flat_idx, ones)
    return nodal_sum / nodal_count.clamp_min(1.0)


def recover_cell_hessian_frobenius_v4(
        points: Tensor,
        u_nodes: Tensor,
        cells: Tensor,
        n_nodes: int,
) -> Tensor:
    """Zienkiewicz-Zhu Hessian recovery + Frobenius norm per cell.

    Same algorithm as v2/v3 but vectorized over all cells in one pass.
    For 2D triangles d=2, so the inner loop is over k=0,1 — trivial.
    """
    d = points.shape[1]
    g_cell = per_cell_p1_gradient_v4(points, u_nodes, cells)  # (C, d)
    g_node = _nodal_average_from_cells(g_cell, cells, n_nodes)  # (N, d)

    # For each component k, take its P1 grad per cell -> a row of H_recon.
    # Stack into (C, d, d).
    H_rows = []
    for k in range(d):
        comp_k = g_node[:, k]
        H_row_k = per_cell_p1_gradient_v4(points, comp_k, cells)  # (C, d)
        H_rows.append(H_row_k)
    H = torch.stack(H_rows, dim=1)  # (C, d, d)
    H = 0.5 * (H + H.transpose(1, 2))
    return torch.linalg.norm(H.reshape(H.shape[0], -1), dim=-1)


def _cell_edge_lengths_pairwise(points: Tensor, cells: Tensor) -> Tensor:
    """All pairwise edge lengths per cell, fully vectorized.

    For K = d+1 vertices per cell there are K*(K-1)/2 unordered pairs.
    Returns: (C, n_pairs)
    """
    verts = points[cells]  # (C, K, d)
    K = cells.shape[1]
    # Build pair indices once.
    i_idx, j_idx = torch.triu_indices(K, K, offset=1)
    diff = verts[:, i_idx] - verts[:, j_idx]   # (C, n_pairs, d)
    return torch.linalg.norm(diff, dim=-1)


def _cell_signed_measure(points: Tensor, cells: Tensor) -> Tensor:
    """Signed measure (area in 2D, volume in 3D) per cell.

    For a simplex with vertices v_0, ..., v_d, measure = |det(E)| / d!
    where E = [v_1-v_0; ...; v_d-v_0]^T.
    """
    verts = points[cells]
    v0 = verts[:, 0]
    edges = verts[:, 1:] - v0.unsqueeze(1)   # (C, d, d)
    d = edges.shape[-1]
    # det of (d, d) matrix per cell.
    sign_meas = torch.linalg.det(edges)
    # factorial(d): for 2D d=2 -> 2, for 3D d=3 -> 6
    fact = 1.0
    for k in range(1, d + 1):
        fact *= k
    return sign_meas / fact


def feature_dim_for_mode(feature_mode: str, *, with_boundary: bool = True) -> int:
    if feature_mode == "full7":
        return 7 if with_boundary else 6
    if feature_mode == "fast5":
        return 5 if with_boundary else 4
    raise ValueError(f"Unknown feature_mode {feature_mode!r}; expected 'full7' or 'fast5'")


def compute_v4_features(
        points: Tensor,
        topology: TopologyV4,
        u_nodes: Tensor,
        *,
        with_boundary: bool = True,
        feature_mode: str = "full7",
) -> Tensor:
    """Compute v3-style features in one vectorized pass.

    Args:
      points: (N, d) — current node positions (autograd-tracked during unroll).
      topology: TopologyV4 — pre-built; provides `cell_blocks_flat`, etc.
      u_nodes: (N,) scalar field. For vector u (NS velocity), compose the
               scalar invariant outside this function (e.g. |u| or its components).
      with_boundary: if True, append the boundary-incident flag.
      feature_mode: "full7" preserves v3 semantics. "fast5" skips Hessian
                    recovery and relative-gradient features for faster inference.

    Returns:
      features: (C, F) with F = 6 (no boundary) or 7 (with boundary).
    """
    cells = topology.cell_blocks_flat
    n_nodes = topology.n_points
    d = points.shape[1]

    # h = |K|^(1/d)
    sign_meas = _cell_signed_measure(points, cells)
    measures = sign_meas.abs().clamp_min(1e-12)
    h = measures.pow(1.0 / d)

    # u_centroid (P1 = mean of nodal values)
    u_at = u_nodes[cells]                                # (C, K)
    u_centroid = u_at.mean(dim=-1)

    # |grad u_h|
    grad = per_cell_p1_gradient_v4(points, u_nodes, cells)   # (C, d)
    grad_norm = torch.linalg.norm(grad, dim=-1)

    # Cell saliency: std(verts) / global range
    u_range = (u_nodes.max() - u_nodes.min()).clamp_min(1e-12).detach()
    u_std = u_at.std(dim=-1, unbiased=False)
    saliency = u_std / u_range

    # Anisotropy: max_edge / min_edge
    edge_lens = _cell_edge_lengths_pairwise(points, cells)   # (C, n_pairs)
    edge_max = edge_lens.max(dim=-1).values
    edge_min = edge_lens.min(dim=-1).values.clamp_min(1e-12)
    aniso = (edge_max / edge_min).clamp(max=1e3)

    if feature_mode == "full7":
        # |H|_F via ZZ recovery
        h_frob = recover_cell_hessian_frobenius_v4(points, u_nodes, cells, n_nodes)
        # Relative gradient
        rel_grad = (h * grad_norm) / (u_centroid.abs() + 1e-6)
        feats = [u_centroid, h * grad_norm, h * h * h_frob,
                 saliency, aniso, rel_grad]
    elif feature_mode == "fast5":
        feats = [u_centroid, h * grad_norm, saliency, aniso]
    else:
        raise ValueError(f"Unknown feature_mode {feature_mode!r}; expected 'full7' or 'fast5'")

    if with_boundary:
        feats.append(topology.boundary_cell.to(points.dtype))

    f = torch.stack(feats, dim=-1)
    # Guard against pathological values from collapsed/degenerate cells.
    f = torch.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)
    return f
