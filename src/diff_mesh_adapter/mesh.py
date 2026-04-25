from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class MeshState:
    """Fixed-topology mesh represented by points and same-arity connectivity blocks."""

    points: Tensor
    cell_blocks: tuple[Tensor, ...]
    boundary_nodes: Tensor | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError("points must have shape (n_points, dim)")
        if not torch.is_floating_point(self.points):
            raise TypeError("points must be a floating point tensor")

        blocks = tuple(self._normalize_block(block) for block in self.cell_blocks)
        if not blocks:
            raise ValueError("cell_blocks must contain at least one connectivity block")
        self.cell_blocks = blocks
        self.boundary_nodes = self._normalize_boundary_nodes(self.boundary_nodes)

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        return sum(int(block.shape[0]) for block in self.cell_blocks)

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "MeshState":
        points = self.points.to(device=device, dtype=dtype if dtype is not None else self.points.dtype)
        blocks = tuple(block.to(device=points.device) for block in self.cell_blocks)
        boundary_nodes = self.boundary_nodes.to(device=points.device)
        return MeshState(points=points, cell_blocks=blocks, boundary_nodes=boundary_nodes)

    def with_points(self, points: Tensor) -> "MeshState":
        return MeshState(points=points, cell_blocks=self.cell_blocks, boundary_nodes=self.boundary_nodes)

    def detached(self) -> "MeshState":
        return MeshState(
            points=self.points.detach(),
            cell_blocks=tuple(block.detach() for block in self.cell_blocks),
            boundary_nodes=self.boundary_nodes.detach(),
        )

    def _normalize_block(self, block: Tensor) -> Tensor:
        if block.ndim != 2:
            raise ValueError("each cell block must have shape (n_cells, nodes_per_cell)")
        if block.shape[1] < 3:
            raise ValueError("each cell must contain at least three nodes")

        normalized = block.to(device=self.points.device, dtype=torch.long)
        if normalized.numel() > 0:
            if int(normalized.min()) < 0 or int(normalized.max()) >= self.num_points:
                raise ValueError("cell connectivity contains point indices outside points")
        return normalized

    def _normalize_boundary_nodes(self, boundary_nodes: Tensor | None) -> Tensor:
        if boundary_nodes is None:
            return torch.zeros(self.num_points, dtype=torch.bool, device=self.points.device)

        boundary_nodes = boundary_nodes.to(device=self.points.device)
        if boundary_nodes.dtype == torch.bool:
            if boundary_nodes.shape != (self.num_points,):
                raise ValueError("boolean boundary_nodes must have shape (n_points,)")
            return boundary_nodes

        indices = boundary_nodes.to(dtype=torch.long).reshape(-1)
        mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.points.device)
        if indices.numel() > 0:
            if int(indices.min()) < 0 or int(indices.max()) >= self.num_points:
                raise ValueError("boundary_nodes contains point indices outside points")
            mask[indices] = True
        return mask
