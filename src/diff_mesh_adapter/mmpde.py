"""Small fixed-boundary MMPDE mesh movement kernels."""

from __future__ import annotations

import numpy as np


def mmpde_winslow_coordinates(
    coordinates: np.ndarray,
    triangles: np.ndarray,
    monitor: np.ndarray,
    *,
    fixed_nodes: np.ndarray | None = None,
    iterations: int = 20,
    relaxation: float = 0.2,
    monitor_power: float = 1.0,
) -> np.ndarray:
    """Move a 2D triangular mesh with a scalar Winslow/MMPDE relaxation.

    This is a compact fixed-boundary variant of the variable-diffusion
    MMPDE idea used in Huang's MMPDElab: for a scalar monitor, solve the
    pseudo-time mesh equation associated with the weighted Dirichlet energy
    ``sum_edges rho_edge |x_i - x_j|^2``.
    """

    x = np.asarray(coordinates, dtype=float).copy()
    cells = np.asarray(triangles, dtype=np.int64)
    rho = np.asarray(monitor, dtype=float).reshape(-1)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("coordinates must have shape (num_vertices, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("triangles must have shape (num_cells, 3)")
    if rho.shape[0] != x.shape[0]:
        raise ValueError("monitor must contain one value per vertex")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not (0.0 <= relaxation <= 1.0):
        raise ValueError("relaxation must be in [0, 1]")

    rho = np.maximum(rho, 1.0e-12)
    rho = rho / max(float(np.mean(rho)), 1.0e-12)
    rho = rho**monitor_power

    edges = _triangle_edges(cells)
    edge_weight = 0.5 * (rho[edges[:, 0]] + rho[edges[:, 1]])
    fixed = np.zeros(x.shape[0], dtype=bool)
    if fixed_nodes is not None:
        fixed[np.asarray(fixed_nodes, dtype=np.int64)] = True

    for _ in range(iterations):
        weighted_sum = np.zeros_like(x)
        weight_total = np.zeros(x.shape[0], dtype=float)
        i = edges[:, 0]
        j = edges[:, 1]
        w = edge_weight[:, None]
        np.add.at(weighted_sum, i, w * x[j])
        np.add.at(weighted_sum, j, w * x[i])
        np.add.at(weight_total, i, edge_weight)
        np.add.at(weight_total, j, edge_weight)

        movable = (~fixed) & (weight_total > 0.0)
        target = x.copy()
        target[movable] = weighted_sum[movable] / weight_total[movable, None]
        x[movable] = (1.0 - relaxation) * x[movable] + relaxation * target[movable]

    return x


def _triangle_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)
