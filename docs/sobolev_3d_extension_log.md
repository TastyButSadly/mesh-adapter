# Sobolev Transport 3D Extension — Development Log

## Goal

Extend `adapt_sobolev_transport_map` to support 3D tetrahedral meshes while
preserving the existing 2D triangle path unchanged.

## Mathematical Background

The Sobolev inner product on a tetrahedral mesh is:

```
a_h(v_h, ψ_h) = β₀ ∫ v_h · ψ_h dx + β₁ ∫ ∇v_h : ∇ψ_h dx
```

In P1 finite elements this gives the same matrix structure:

```
A_h = β₀ M_lumped + β₁ K_fem
```

The difference from 2D is only in the element-level computation:
- Area |K| → Volume |K|
- P1 gradients on triangle → P1 gradients on tetrahedron
- 3 vertex basis functions → 4 vertex basis functions

### P1 gradients on a tetrahedron

For tetrahedron with vertices p₀, p₁, p₂, p₃, the basis function gradients
are constant per element:

```
∇φ_i = (1 / (6|K|)) * (p_j - p_k) × (p_l - p_k)    (cyclic permutation)
```

Or equivalently, using the inverse Jacobian:

```
J = [p₁-p₀, p₂-p₀, p₃-p₀]    (3×3 matrix of edge vectors)
∇φ₀ = -J⁻ᵀ [1,1,1]ᵀ
∇φ₁ =  J⁻ᵀ [1,0,0]ᵀ
∇φ₂ =  J⁻ᵀ [0,1,0]ᵀ
∇φ₃ =  J⁻ᵀ [0,0,1]ᵀ
```

### P1 stiffness on a tetrahedron

```
K_ij^K = |K| ∇φ_i · ∇φ_j
```

where |K| is the signed volume and the dot product is in R³.

### Lumped mass on a tetrahedron

```
m_i = Σ_{K∋i} |K| / 4
```

(each vertex gets 1/4 of the tetrahedron volume)

## Implementation Plan

1. Add `tetra_lumped_mass(points, cells, measures, eps)` — analogous to
   `_triangle_lumped_mass`
2. Add `tetra_stiffness_coefficients(points, cells, measures, eps)` —
   analogous to `_triangle_stiffness_coefficients`
3. Update `_apply_sobolev_metric` to handle both 2D and 3D
4. Update `_sobolev_metric_diagonal` similarly
5. Update `adapt_sobolev_transport_map` to branch on `mesh.dim`
6. Add 3D edge extraction (reuse `_tetra_edge_pairs` from adapt.py)
7. Keep all existing 2D code paths unchanged

## Files modified

- `src/diff_mesh_adapter/sobolev_transport.py` — main changes (3D functions, dimension-agnostic code)

## 3D P1 Stiffness Derivation

For tetrahedron with vertices p₀, p₁, p₂, p₃, signed volume:

```
V = (1/6) det[p₁-p₀, p₂-p₀, p₃-p₀]
```

P1 basis function gradients (standard FEM, e.g. Brenner & Scott):

```
∇φ₀ = (1/(6V)) * (p₂-p₁) × (p₃-p₁)
∇φ₁ = (1/(6V)) * (p₃-p₀) × (p₂-p₀)
∇φ₂ = (1/(6V)) * (p₁-p₀) × (p₃-p₀)
∇φ₃ = (1/(6V)) * (p₂-p₀) × (p₁-p₀)
```

Equivalently, using the Jacobian J = [e₁, e₂, e₃] where eᵢ = p_{i+1}-p₀:

```
∇φ₀ = -J⁻ᵀ [1,1,1]ᵀ
∇φ₁ =  J⁻ᵀ [1,0,0]ᵀ
∇φ₂ =  J⁻ᵀ [0,1,0]ᵀ
∇φ₃ =  J⁻ᵀ [0,0,1]ᵀ
```

Stiffness entry K_ij^K = |V| ∇φ_i · ∇φ_j, or equivalently via cross products:

```
K_ij^K = (1/(36|V|)) * (b_i · b_j)
```

where b_i are the cross-product vectors defined above.

## 3D Sphere Flow Test Results

Test: 3D channel with sphere (783 points, 3269 tetrahedra), synthetic monitor
concentrated near sphere and wake.

| Step | Loss | Volume Ratio | Flips | Time |
|------|------|-------------|-------|------|
| 0 (initial) | 0.6083 | 367.1 | 0 | -- |
| 1 | 0.5609 | 273.2 | 0 | 0.032s |
| 2 | 0.5254 | 226.4 | 0 | 0.019s |
| 3 | 0.4989 | 200.7 | 0 | 0.019s |
| 4 | 0.4780 | 185.4 | 0 | 0.019s |
| 5 | 0.4607 | 175.8 | 0 | 0.019s |

- Loss decreased 24% over 5 steps
- Volume ratio improved 52% (367 → 176)
- Zero orientation flips at every step
- ~20ms per step

- [x] Unit tests for tetra lumped mass (via integration in main function)
- [x] Unit tests for tetra stiffness (via integration in main function)
- [x] 3D test: uniform monitor → mesh should not move (fixed point) ✓
- [x] 3D test: non-uniform monitor → mesh should concentrate ✓
- [x] 3D test: orientation preserved after adaptation ✓
- [ ] Compare with regularized adapter on same 3D problem

## Bugs Found and Fixed

### Bug 1: Gradient cross product order (CRITICAL)
The P1 gradient formula for tetrahedral volume `V = (1/6) det[p1-p0, p2-p0, p3-p0]`
requires:

```
∂V/∂p1 = (1/6) (p2-p0) × (p3-p0)
∂V/∂p2 = (1/6) (p3-p0) × (p1-p0)
∂V/∂p3 = (1/6) (p1-p0) × (p2-p0)
```

The initial code had the cross products in reversed order for ALL FOUR gradients,
producing gradients with the wrong sign. This was verified against PyTorch autograd
and finite differences.

Root cause: The code was derived from the 2D triangle gradient formula where the
cross product structure is different (scalar cross product in 2D vs vector cross
product in 3D). In 2D, `∂|K|/∂p_i` involves rotated edge vectors `(y_j-y_k, x_k-x_j)`,
which naturally have the correct orientation. In 3D, the vector cross product is
anti-commutative (`a×b = -b×a`), so the argument order matters.

### Bug 2: `_apply_sobolev_metric` hardcoded range(3)
The loop `for local_i in range(3)` assumed triangle meshes. Fixed to use
`cells.shape[1]` for dimension-agnostic behavior.
