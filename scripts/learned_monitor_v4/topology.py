"""Topology cache and vectorized adjacency builder.

The v2 builder used a Python dict-of-tuples and a triple nested loop. For
N ~ 1e5 cells this was O(N * (d+1)) hash operations on Python strings and
became a noticeable cold-start cost (~5-30 s).

The v4 builder uses pure numpy:
    faces  = sort(triples of cell vertices)  -> (N_faces, face_size)
    sorted = lexsort(faces)
    pairs where adjacent rows in sorted-faces are equal share a face.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

Tensor = torch.Tensor


def build_cell_adjacency_fast(
        cell_blocks: tuple[Tensor, ...], dim: int
) -> Tensor:
    """Vectorized cell-cell adjacency. Two cells share a face iff their
    (sorted) face-vertex tuple appears twice in the global face list.

    Returns: (E, 2) tensor of cell-index pairs (each pair appears once).
    """
    face_size = dim  # 2 for triangles in 2D, 3 for tetra in 3D

    # Concatenate all blocks; remember the per-block offset.
    n_per_block = [int(b.shape[0]) for b in cell_blocks]
    offsets = np.concatenate([[0], np.cumsum(n_per_block)])
    total_cells = int(offsets[-1])

    # Build face-vertex arrays for every cell, every (K choose face_size) face.
    face_lists = []   # rows: (cell_idx, *face_vertices_sorted)
    for bi, block in enumerate(cell_blocks):
        b_np = block.detach().cpu().numpy()
        K = b_np.shape[1]
        face_combos = list(combinations(range(K), face_size))  # tiny: <=4 in 2D, <=4 in 3D
        cell_ids = np.arange(b_np.shape[0]) + offsets[bi]
        for combo in face_combos:
            faces = b_np[:, list(combo)]
            faces_sorted = np.sort(faces, axis=1)
            cols = np.concatenate(
                [cell_ids.reshape(-1, 1), faces_sorted], axis=1
            )
            face_lists.append(cols)
    if not face_lists:
        return torch.empty((0, 2), dtype=torch.long)

    all_faces = np.concatenate(face_lists, axis=0)
    # Lex-sort by face columns (everything after column 0)
    order = np.lexsort(all_faces[:, 1:][:, ::-1].T)
    s = all_faces[order]
    # Adjacent rows share a face iff their face columns are identical
    face_cols = s[:, 1:]
    same = np.all(face_cols[1:] == face_cols[:-1], axis=1)
    # Where same[i] is True, cells s[i,0] and s[i+1,0] are adjacent.
    pairs_a = s[:-1, 0][same]
    pairs_b = s[1:, 0][same]
    pairs = np.stack([pairs_a, pairs_b], axis=1).astype(np.int64)
    # Drop self-pairs that can arise if a degenerate face hashes to itself.
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return torch.from_numpy(pairs)


@dataclass
class TopologyV4:
    """Pre-computed, immutable, per-mesh topology.

    Built once per mesh; reused for every feature / monitor / inner-step call.
    All index tensors live on the same device as `points_template`.

    Attributes
    ----------
    cell_blocks_flat : (C, K) long tensor
        Concatenated cell-vertex indices across all blocks. K = d+1.
    block_sizes : list[int]
        Number of cells per block (for compatibility with cell_signed_measures
        which takes a tuple of blocks).
    cell_blocks_tuple : tuple of (C_b, K) long tensors
        The original blocks (kept as device tensors), needed for routines in
        diff_mesh_adapter.geometry that still consume the tuple form.
    n_cells : int
    n_points : int
    dim : int
    adjacency : (E, 2) long tensor
    adj_src_undirected : (2E,) long tensor
        Source cell indices for both directions of every undirected edge.
    adj_dst_undirected : (2E,) long tensor
        Destination cell indices for both directions.
    adj_degree : (C, 1) tensor of float dtype
        Per-cell neighbor count (>= 0). Used to normalize scatter-mean.
    boundary_mask : (N,) bool tensor
    boundary_cell : (C,) bool tensor
        True iff at least one vertex of the cell is on the boundary.
    """
    cell_blocks_flat: Tensor
    block_sizes: list[int]
    cell_blocks_tuple: tuple[Tensor, ...]
    n_cells: int
    n_points: int
    dim: int
    adjacency: Tensor
    adj_src_undirected: Tensor
    adj_dst_undirected: Tensor
    adj_degree: Tensor
    boundary_mask: Tensor
    boundary_cell: Tensor

    @staticmethod
    def build(
            points: Tensor,
            cell_blocks: tuple[Tensor, ...],
            boundary_mask: Tensor,
            *,
            dtype: torch.dtype | None = None,
    ) -> "TopologyV4":
        device = points.device
        dim = int(points.shape[1])
        if dtype is None:
            dtype = points.dtype

        # 1) flatten cell_blocks into a single (C, K) long tensor
        n_per = [int(b.shape[0]) for b in cell_blocks]
        flat = torch.cat([b.to(device=device, dtype=torch.long) for b in cell_blocks], dim=0)
        n_cells = int(flat.shape[0])
        n_points = int(points.shape[0])

        # 2) adjacency
        adj = build_cell_adjacency_fast(cell_blocks, dim=dim).to(device=device)

        # 3) undirected scatter indices (cat both directions)
        if adj.numel() > 0:
            src = adj[:, 0]
            dst = adj[:, 1]
            src_u = torch.cat([src, dst])
            dst_u = torch.cat([dst, src])
        else:
            src_u = torch.empty((0,), dtype=torch.long, device=device)
            dst_u = torch.empty((0,), dtype=torch.long, device=device)

        # 4) degree
        degree = torch.zeros((n_cells, 1), dtype=dtype, device=device)
        if dst_u.numel() > 0:
            degree.index_add_(
                0, dst_u,
                torch.ones((dst_u.shape[0], 1), dtype=dtype, device=device),
            )

        # 5) boundary cells: any vertex on the boundary
        bmask = boundary_mask.to(device=device, dtype=torch.bool)
        boundary_cell = bmask[flat].any(dim=-1)

        # The original cell_blocks_tuple should be device-tensors with long dtype.
        cb_tuple = tuple(b.to(device=device, dtype=torch.long) for b in cell_blocks)

        return TopologyV4(
            cell_blocks_flat=flat,
            block_sizes=n_per,
            cell_blocks_tuple=cb_tuple,
            n_cells=n_cells,
            n_points=n_points,
            dim=dim,
            adjacency=adj,
            adj_src_undirected=src_u,
            adj_dst_undirected=dst_u,
            adj_degree=degree,
            boundary_mask=bmask,
            boundary_cell=boundary_cell,
        )
