from __future__ import annotations

import numpy as np


def recover_velocity_hessian_frobenius(
        points: np.ndarray,
        cells: np.ndarray,
        velocity_nodes: np.ndarray,
) -> np.ndarray:
    """Recover a cellwise velocity Hessian norm with ZZ-style nodal averaging."""
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    velocity_nodes = np.asarray(velocity_nodes, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    if velocity_nodes.ndim != 2 or velocity_nodes.shape[0] != points.shape[0]:
        raise ValueError("velocity_nodes must have shape (n_points, n_components)")

    cell_gradient = _p1_cell_gradient(points, cells, velocity_nodes)
    nodal_gradient = _nodal_average(cell_gradient, cells, points.shape[0])
    hessian_sq = np.zeros(cells.shape[0], dtype=np.float64)
    for derivative_component in range(points.shape[1]):
        recovered_gradient = _p1_cell_gradient(
            points,
            cells,
            nodal_gradient[:, derivative_component, :],
        )
        hessian_sq += np.sum(recovered_gradient * recovered_gradient, axis=(1, 2))
    return np.sqrt(hessian_sq)


def recover_scalar_hessian_frobenius(
        points: np.ndarray,
        cells: np.ndarray,
        scalar_nodes: np.ndarray,
) -> np.ndarray:
    """Recover a cellwise scalar Hessian norm with ZZ-style nodal averaging."""
    scalar_nodes = np.asarray(scalar_nodes, dtype=np.float64)
    if scalar_nodes.ndim != 1:
        raise ValueError("scalar_nodes must have shape (n_points,)")
    return recover_velocity_hessian_frobenius(points, cells, scalar_nodes[:, None])


def _p1_cell_gradient(points: np.ndarray, cells: np.ndarray, nodal_values: np.ndarray) -> np.ndarray:
    vertices = points[cells]
    edges = vertices[:, 1:, :] - vertices[:, :1, :]
    values = nodal_values[cells]
    differences = values[:, 1:, :] - values[:, :1, :]
    return np.linalg.solve(edges, differences)


def _nodal_average(cell_values: np.ndarray, cells: np.ndarray, n_points: int) -> np.ndarray:
    nodal_sum = np.zeros((n_points,) + cell_values.shape[1:], dtype=cell_values.dtype)
    nodal_count = np.zeros(n_points, dtype=np.float64)
    for column in range(cells.shape[1]):
        np.add.at(nodal_sum, cells[:, column], cell_values)
        np.add.at(nodal_count, cells[:, column], 1.0)
    return nodal_sum / np.maximum(nodal_count, 1.0).reshape((-1,) + (1,) * (cell_values.ndim - 1))
