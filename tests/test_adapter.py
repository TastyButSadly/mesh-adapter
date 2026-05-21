import socket

import numpy as np
import torch

from diff_mesh_adapter import AdaptationConfig, MeshState, adapt_cell_area_equalization, cell_abs_areas, cell_signed_areas
from examples.firedrake._adapter_tcp_protocol import recv_message, send_message
from examples.firedrake._diff_adapter_subprocess import AdapterTopologyCache, adapt_coordinates


def _four_triangle_mesh() -> MeshState:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.22, 0.37],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=torch.long,
    )
    boundary_nodes = torch.tensor([True, True, True, True, False])
    return MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)


def test_adapter_tcp_protocol_round_trips_arrays_and_scalars():
    left, right = socket.socketpair()
    try:
        points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        cells = np.array([[0, 1, 0]], dtype=np.int64)
        send_message(left, scalars={"steps": 2, "profile": "regularized"}, arrays={"points": points, "cells": cells})

        scalars, arrays = recv_message(right)
    finally:
        left.close()
        right.close()

    assert scalars == {"steps": 2, "profile": "regularized"}
    assert np.array_equal(arrays["points"], points)
    assert np.array_equal(arrays["cells"], cells)


def test_adapter_equalizes_synthetic_triangle_areas_and_keeps_boundary_fixed():
    mesh = _four_triangle_mesh()
    initial_areas = cell_abs_areas(mesh.points, mesh.cell_blocks)
    initial_cells = mesh.cell_blocks[0].clone()

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=200, lr=3e-2, movement_weight=0.0, shape_weight=0.0),
    )

    final_areas = cell_abs_areas(result.mesh.points, result.mesh.cell_blocks)
    final_signed = cell_signed_areas(result.mesh.points, result.mesh.cell_blocks)

    assert torch.equal(result.mesh.cell_blocks[0], initial_cells)
    assert torch.allclose(result.mesh.points[mesh.boundary_nodes], mesh.points[mesh.boundary_nodes])
    assert torch.all(final_signed > 0.0)
    assert final_areas.std(unbiased=False) < initial_areas.std(unbiased=False)
    assert result.final_loss < result.initial_loss


def test_adapter_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        return

    mesh = _four_triangle_mesh().to(device="cuda")
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=10, lr=1e-2, movement_weight=0.0, shape_weight=0.0),
    )

    assert result.mesh.points.is_cuda
    assert result.min_area_history[-1] > 0.0


def test_adapter_preserves_clockwise_connectivity_and_uses_relative_orientation():
    mesh = _four_triangle_mesh()
    clockwise_cells = torch.flip(mesh.cell_blocks[0], dims=[1])
    clockwise_mesh = MeshState(points=mesh.points, cell_blocks=(clockwise_cells,), boundary_nodes=mesh.boundary_nodes)

    result = adapt_cell_area_equalization(
        clockwise_mesh,
        AdaptationConfig(steps=20, lr=1e-2, movement_weight=0.0, shape_weight=0.0),
    )

    assert torch.equal(result.mesh.cell_blocks[0], clockwise_cells)
    assert result.min_area_history[-1] > 0.0


def test_adapter_stops_after_patience_without_significant_improvement():
    mesh = _four_triangle_mesh()

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=100,
            lr=1e-2,
            movement_weight=0.0,
            shape_weight=0.0,
            early_stopping_patience=3,
            early_stopping_min_delta=0.99,
        ),
    )

    assert result.early_stopped
    assert result.steps_completed < 100
    assert len(result.loss_history) == result.steps_completed + 1


def test_adapter_can_skip_step_diagnostics():
    mesh = _four_triangle_mesh()

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=2,
            lr=1e-2,
            movement_weight=0.0,
            shape_weight=0.0,
            collect_step_diagnostics=False,
        ),
    )

    assert len(result.loss_history) == result.steps_completed + 1
    assert result.area_std_history == []
    assert result.min_area_history == []
    assert result.lr_history == []


def test_adapter_boundary_edge_barrier_supports_2d_quads():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.45, 0.40],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        dtype=torch.long,
    )
    boundary_nodes = torch.tensor([True, True, True, True, False, True, True, True, True])
    mesh = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)

    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(
            steps=2,
            lr=1e-3,
            edge_length_barrier_weight=0.5,
            boundary_edge_length_barrier_weight=1.0,
            early_stopping_patience=None,
        ),
    )

    assert result.mesh.points.shape == mesh.points.shape


def test_firedrake_adapter_exchange_accepts_tetra_cells():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            [0, 1, 2, 8],
            [0, 2, 3, 8],
            [4, 6, 5, 8],
            [4, 7, 6, 8],
            [0, 4, 5, 8],
            [0, 5, 1, 8],
            [1, 5, 6, 8],
            [1, 6, 2, 8],
            [2, 6, 7, 8],
            [2, 7, 3, 8],
            [3, 7, 4, 8],
            [3, 4, 0, 8],
        ],
        dtype=np.int64,
    )
    monitor = np.ones(points.shape[0], dtype=np.float64)

    adapted, info = adapt_coordinates(
        points_np=points,
        cells_np=cells,
        monitor_np=monitor,
        steps=2,
        lr=1.0e-4,
    )

    assert adapted.shape == points.shape
    assert info["steps_completed"] == 2


def test_firedrake_adapter_exchange_caches_topology_and_supports_float32():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.22, 0.37],
        ],
        dtype=np.float32,
    )
    cells = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    monitor = np.ones(points.shape[0], dtype=np.float32)
    cache = AdapterTopologyCache()

    adapted, info = adapt_coordinates(
        points_np=points,
        cells_np=cells,
        monitor_np=monitor,
        steps=1,
        lr=1.0e-4,
        dtype="float32",
        topology_cache=cache,
    )
    adapted_again, _ = adapt_coordinates(
        points_np=points,
        cells_np=cells,
        monitor_np=monitor,
        steps=1,
        lr=1.0e-4,
        dtype="float32",
        topology_cache=cache,
    )
    adapted_from_cell_monitor, _ = adapt_coordinates(
        points_np=points,
        cells_np=cells,
        cell_monitor_np=np.ones(cells.shape[0], dtype=np.float32),
        steps=1,
        lr=1.0e-4,
        dtype="float32",
        topology_cache=cache,
    )

    assert adapted.shape == points.shape
    assert adapted.dtype == np.float32
    assert adapted_again.shape == points.shape
    assert adapted_from_cell_monitor.shape == points.shape
    assert info["steps_completed"] == 1
    assert cache.misses == 1
    assert cache.hits == 2
