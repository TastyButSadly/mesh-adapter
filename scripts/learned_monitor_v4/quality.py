"""Mesh quality metrics for learned-monitor evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from diff_mesh_adapter.geometry import cell_signed_measures

from .topology import TopologyV4

Tensor = torch.Tensor


@dataclass(frozen=True)
class QualityGate:
    max_area_ratio: float = 50.0
    min_area_fraction: float = 0.10
    max_displacement_over_mean_edge: float | None = None


def triangle_mesh_quality(points: Tensor, topology: TopologyV4, *,
                          initial_points: Tensor | None = None) -> dict[str, float | int]:
    signed = cell_signed_measures(points, topology.cell_blocks_tuple).detach()
    area = signed.abs().clamp_min(1e-300)
    metrics: dict[str, float | int] = {
        "min_area": float(area.min()),
        "max_area": float(area.max()),
        "area_ratio": float(area.max() / area.min()),
    }
    if initial_points is not None:
        initial_signed = cell_signed_measures(initial_points, topology.cell_blocks_tuple).detach()
        orientation = torch.sign(initial_signed)
        metrics["orientation_flips"] = int(((signed * orientation) <= 0.0).sum())
        metrics["initial_min_area"] = float(initial_signed.abs().min())
        displacement = torch.linalg.norm(points.detach() - initial_points.detach().to(points), dim=1)
        metrics["max_displacement"] = float(displacement.max())
        metrics["max_displacement_over_mean_edge"] = float(displacement.max() / _mean_edge_length(initial_points, topology))
    else:
        metrics["orientation_flips"] = int((signed == 0.0).sum())
    return metrics


def passes_quality_gate(metrics: dict[str, float | int], gate: QualityGate) -> bool:
    if int(metrics.get("orientation_flips", 0)) > 0:
        return False
    if float(metrics["area_ratio"]) > gate.max_area_ratio:
        return False
    initial_min_area = metrics.get("initial_min_area")
    if initial_min_area is not None:
        if float(metrics["min_area"]) < gate.min_area_fraction * float(initial_min_area):
            return False
    if gate.max_displacement_over_mean_edge is not None:
        value = metrics.get("max_displacement_over_mean_edge")
        if value is not None and float(value) > gate.max_displacement_over_mean_edge:
            return False
    return True


def _mean_edge_length(points: Tensor, topology: TopologyV4) -> Tensor:
    cells = topology.cell_blocks_flat
    verts = points[cells]
    i_idx, j_idx = torch.triu_indices(cells.shape[1], cells.shape[1], offset=1, device=points.device)
    lengths = torch.linalg.norm(verts[:, i_idx] - verts[:, j_idx], dim=-1)
    return lengths.mean().clamp_min(1e-300)
