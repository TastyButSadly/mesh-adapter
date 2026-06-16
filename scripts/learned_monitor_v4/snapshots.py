"""Snapshot dataset and differentiable P1 field helpers for v4 training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import torch

from diff_mesh_adapter.mesh import MeshState
from diff_mesh_adapter.io import read_gmsh_mesh

from .topology import TopologyV4

Tensor = torch.Tensor


@dataclass(frozen=True)
class SnapshotSample:
    index: int
    time: float
    u_nodes: Tensor


class SnapshotDataset:
    """Loads NS snapshots saved by scripts/generate_ns_snapshots.py."""

    def __init__(
            self,
            npz_path: str | Path,
            *,
            dtype: torch.dtype = torch.float64,
            device: torch.device | str | None = None,
            field: str = "vorticity_abs",
    ) -> None:
        data = np.load(Path(npz_path))
        points = torch.tensor(data["points"], dtype=dtype, device=device)
        cells = torch.tensor(data["cells"], dtype=torch.long, device=points.device)
        boundary_mask = torch.tensor(data["boundary_mask"], dtype=torch.bool, device=points.device)

        if field not in data:
            raise KeyError(f"Snapshot field {field!r} not found in {npz_path}. Available keys: {list(data.files)}")
        values_np = data[field]
        if values_np.ndim != 2:
            raise ValueError(f"Expected scalar snapshot field with shape (T, N), got {values_np.shape}")
        values = torch.tensor(values_np, dtype=dtype, device=points.device)

        self.path = Path(npz_path)
        self.field = field
        self.mesh = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_mask)
        self.topology = TopologyV4.build(points, self.mesh.cell_blocks, boundary_mask, dtype=dtype)
        self.times = torch.tensor(data["times"], dtype=dtype, device=points.device)
        self.values = values

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> SnapshotSample:
        return SnapshotSample(
            index=int(index),
            time=float(self.times[index].detach().cpu()),
            u_nodes=self.values[index],
        )


class ReferenceSnapshotDataset:
    """Fine-reference snapshots paired with a separate coarse train/eval mesh."""

    def __init__(
            self,
            reference_npz_path: str | Path,
            coarse_mesh_path: str | Path,
            *,
            dtype: torch.dtype = torch.float64,
            device: torch.device | str | None = None,
            field: str = "vorticity_abs",
    ) -> None:
        data = np.load(Path(reference_npz_path))
        ref_points = torch.tensor(data["points"], dtype=dtype, device=device)
        ref_cells = torch.tensor(data["cells"], dtype=torch.long, device=ref_points.device)
        if field not in data:
            raise KeyError(f"Snapshot field {field!r} not found in {reference_npz_path}. Available keys: {list(data.files)}")
        values_np = data[field]
        if values_np.ndim != 2:
            raise ValueError(f"Expected scalar snapshot field with shape (T, N), got {values_np.shape}")

        self.path = Path(reference_npz_path)
        self.coarse_mesh_path = Path(coarse_mesh_path)
        self.field = field
        self.reference_points = ref_points
        self.reference_cells = ref_cells
        self.reference_values = torch.tensor(values_np, dtype=dtype, device=ref_points.device)
        self.times = torch.tensor(data["times"], dtype=dtype, device=ref_points.device)

        self.mesh = read_gmsh_mesh(coarse_mesh_path, dtype=dtype, device=device)
        self.topology = TopologyV4.build(
            self.mesh.points,
            self.mesh.cell_blocks,
            self.mesh.boundary_nodes,
            dtype=dtype,
        )

    def __len__(self) -> int:
        return int(self.reference_values.shape[0])

    def __getitem__(self, index: int) -> SnapshotSample:
        field = self.field_for(index)
        u_nodes = field.evaluate(self.mesh.points).detach()
        return SnapshotSample(
            index=int(index),
            time=float(self.times[index].detach().cpu()),
            u_nodes=u_nodes,
        )

    def field_for(self, index: int) -> "SnapshotP1Field":
        return SnapshotP1Field(
            self.reference_points,
            self.reference_cells,
            self.reference_values[index],
        )


class SnapshotP1Field:
    """P1 interpolation of a scalar snapshot on the original triangulation.

    Triangle lookup is non-differentiable and is done from detached query
    coordinates. Once a source triangle is selected, barycentric interpolation
    is computed with torch so gradients flow to query point coordinates.
    """

    def __init__(self, points: Tensor, cells: Tensor, values: Tensor) -> None:
        if cells.shape[1] != 3:
            raise ValueError("SnapshotP1Field currently supports triangle meshes only")
        self.points = points.detach()
        self.cells = cells.detach().to(dtype=torch.long)
        self.values = values.detach()

        points_np = self.points.detach().cpu().numpy()
        cells_np = self.cells.detach().cpu().numpy()
        self._triangulation = mtri.Triangulation(points_np[:, 0], points_np[:, 1], cells_np)
        self._finder = self._triangulation.get_trifinder()
        centroids = points_np[cells_np].mean(axis=1)
        self._centroids = centroids

    def evaluate(self, query_points: Tensor) -> Tensor:
        tri_idx_np = self._find_triangles(query_points)
        tri_idx = torch.tensor(tri_idx_np, dtype=torch.long, device=query_points.device)
        cells_q = self.cells.to(device=query_points.device)[tri_idx]
        verts = self.points.to(device=query_points.device, dtype=query_points.dtype)[cells_q]
        values = self.values.to(device=query_points.device, dtype=query_points.dtype)[cells_q]
        bary = _triangle_barycentric(query_points, verts)
        return (bary * values).sum(dim=-1)

    def _find_triangles(self, query_points: Tensor) -> np.ndarray:
        q = query_points.detach().cpu().numpy()
        tri_idx = np.asarray(self._finder(q[:, 0], q[:, 1]), dtype=np.int64)
        outside = tri_idx < 0
        if outside.any():
            # Fallback for tiny excursions outside the initial triangulation.
            # This keeps training finite; the valid mesh barrier should make it rare.
            diff = q[outside, None, :] - self._centroids[None, :, :]
            tri_idx[outside] = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
        return tri_idx


def _triangle_barycentric(query_points: Tensor, verts: Tensor, eps: float = 1e-30) -> Tensor:
    a = verts[:, 0]
    b = verts[:, 1]
    c = verts[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = query_points - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = torch.where(den.abs() < eps, den.sign().clamp(min=0.0) * eps + eps, den)
    beta = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    gamma = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    alpha = 1.0 - beta - gamma
    return torch.stack([alpha, beta, gamma], dim=-1)
