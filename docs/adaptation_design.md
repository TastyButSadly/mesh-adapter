# Differentiable Fixed-Topology Adaptation Notes

This document is the running design record for the adapter. Update it when the loss, monitor functions, constraints, or demo assumptions change.

## Core Assumption

We do not differentiate the external mesh generator. Gmsh or another tool creates a valid initial mesh, then the adapter keeps connectivity fixed and optimizes only nodal coordinates.

The public mesh state stays simple:

```text
points: (num_points, dim)
cell_blocks: tuple[(num_cells_in_block, nodes_per_cell)]
boundary_nodes: (num_points,) bool
```

The same optimization loop is used for 2D and 3D. Geometry kernels choose the appropriate measure and quality terms from `points.shape[1]` and block arity.

## Current Optimization Loss

For monitor-based r-adaptation we currently optimize

```text
L = L_monitor
  + lambda_move * L_move
  + lambda_smooth * L_smooth
  + lambda_shape * L_shape
  + lambda_quality * L_quality_barrier
  + lambda_edge * L_edge
  + lambda_barrier * L_barrier
```

where

```text
L_monitor = mean((M_K |K| / mean(M_K |K|) - 1)^2)
```

This equidistributes monitor-weighted cell measure. If `M_K` is high near a feature, then `|K|` is driven lower there.

Regularization terms:

- `L_move = mean(||x - x0||^2)` keeps the new mesh close to the previous mesh. In transient mode, `x0` is the mesh from the previous physical time step.
- `L_smooth` penalizes relative displacement variation inside each cell. This discourages local fold-like motion and noisy node displacement.
- `L_shape` penalizes poor elements. In 2D triangles it uses a scale-invariant energy against an equilateral reference triangle. In 3D tetrahedra it uses the analogous regular-tetrahedron energy.
- `L_quality_barrier` penalizes cells whose quality falls below a configured threshold. Boundary-touching cells can receive a larger weight because fixed boundaries make them the first place where monitor forcing creates needles.
- `L_edge` penalizes edge-length drift from a reference mesh. This is the size-control term that prevents a small set of cells from stretching into long strips while still passing an angle-only quality check.
- `L_barrier = mean(relu(eps - oriented_measure)^2 / target_measure^2)` keeps signed areas/volumes positive.

Boundary nodes are fixed in V1. After every optimizer step, boundary coordinates are reset to their initial coordinates for that adaptation call.

There is also a hard quality guard:

```text
if min(cell_quality) <= min_step_cell_quality:
    rollback step
    shrink learning rate
```

This is intentionally stronger than the monitor term. It prevents the optimizer from buying monitor equidistribution by creating sharp boundary triangles.

There is an analogous edge-length guard:

```text
if max(edge_length / reference_edge_length) >= max_step_edge_stretch:
    rollback step
    shrink learning rate
```

The transient demo uses the original mesh as the reference. This prevents cumulative drift across physical time steps.

The current scalar quality is

```text
q_K = 1 / (1 + shape_energy_K)
```

so `q_K = 1` for a reference-quality simplex and `q_K -> 0` for degenerate/needle-like elements.

## Early Stopping

The default early stopping mode is relative improvement against the best loss. Transient demos use absolute improvement:

```text
early_stopping_relative = False
early_stopping_min_delta = 1e-3
early_stopping_patience = 10
```

This means we stop when the best loss has not improved by at least `1e-3` for 10 optimizer iterations.

## Monitor Function Choices

The monitor should represent where the PDE discretization needs resolution. The current demos use analytic proxy monitors because we are not yet coupled to a solver.

Implemented monitor families:

- Gaussian value monitor: `M = 1 + alpha u`, useful for showing concentration around a compact feature.
- Gaussian gradient monitor: `M = sqrt(1 + alpha |grad u|^2)`, closer to interpolation-error logic.
- Smooth-front gradient monitor: concentrates cells along steep fronts.
- Time-integrated Gaussian-gradient monitor: approximates a stationary mesh that covers a feature trajectory over a time window.
- Moving Gaussian ridge monitor: a transient line feature

```text
d(x, t) = n dot x - (offset0 + speed t)
u(x, t) = exp(-d(x, t)^2 / (2 width^2))
M(x, t) = 1 + alpha u(x, t)
```

The moving ridge is useful as a solver-free transient test: at each physical step the feature location changes, and we adapt the previous mesh to the current monitor with a small optimization budget.

## Why These Terms

Pure area equalization is not a useful scientific demo because it can make a good external mesh worse. The useful claim is different:

```text
Given a fixed cell budget, move cells toward regions that matter for the solution while preserving topology and acceptable element quality.
```

The monitor term provides feature tracking. The movement and smoothness terms provide temporal coherence and reduce ALE-like mesh motion artifacts. The shape and barrier terms keep the mesh usable for a downstream solver.

The relevant literature points in the same direction:

- Huang and Russell formulate moving mesh adaptation around equidistribution and alignment, not equidistribution alone: https://link.springer.com/book/10.1007/978-1-4419-7916-2
- Huang and Kamenski prove nonsingularity for MMPDE discretizations under coercive meshing functionals, including lower bounds on element altitudes and volumes: https://arxiv.org/abs/1512.04971
- TMOP treats mesh optimization as a target-Jacobian quality problem with local metrics, target matrices, and practical constraints such as tangential relaxation and limiting deviation from the original mesh: https://epubs.siam.org/doi/10.1137/18M1167206
- MFEM's TMOP implementation exposes local mesh quality metrics and barrier-style optimization machinery: https://docs.mfem.org/4.1/classmfem_1_1TMOP__QualityMetric.html

## Current Demo Interpretation

The transient 2D demo adapts a denser Gmsh unit-square mesh to a moving Gaussian ridge. It does not solve the PDE yet. It demonstrates the mesh-motion part that would be inserted between physical solver steps:

```text
mesh_n -> build monitor M(t_n) -> 25-step adaptation -> mesh_{n+1}
```

Current demo command:

```bash
uv run python -m examples.applications.transient_moving_ridge_2d
```

Current parameters:

- mesh size: `h = 0.06`, producing about twice as many cells as the earlier coarse demo;
- physical samples: 13 times from `t = 0.0` to `t = 0.82`;
- optimizer budget per physical step: 25 iterations;
- early stopping: absolute `1e-3` best-loss improvement patience over 10 iterations;
- monitor: moving Gaussian ridge with `width = 0.065`, `alpha = 8.0`;
- quality control: shape loss, displacement-smoothness loss, soft quality barrier, stronger boundary-cell quality barrier, edge-length drift penalty, hard `min_step_cell_quality = 0.58` guard, and hard `max_step_edge_stretch = 2.3` guard against the original mesh.

Latest run:

```text
same mesh: 380 points, 690 cells
mean high-monitor area ratio over top 10% monitor cells: 0.794262
mean relative loss reduction: 0.277584
minimum triangle angle over all physical steps: 22.5151 degrees
max edge ratio over all physical steps: 2.29883
final min signed area: 0.000912651
final min triangle angle: 24.7287 degrees
max displacement: 0.11647
boundary max displacement: 0

fine mesh: 1408 points, 2678 cells
mean high-monitor area ratio over top 10% monitor cells: 0.837265
mean relative loss reduction: 0.224016
minimum triangle angle over all physical steps: 21.0544 degrees
max edge ratio over all physical steps: 2.00358
final min signed area: 0.000251028
final min triangle angle: 23.9073 degrees
max displacement: 0.0482165
boundary max displacement: 0
```

Artifacts:

```text
outputs/transient_moving_ridge_2d/moving_ridge_transient_adaptation.gif
outputs/transient_moving_ridge_2d/moving_ridge_initial_final_monitor.png
outputs/transient_moving_ridge_2d/moving_ridge_same_transient_adaptation.gif
outputs/transient_moving_ridge_2d/moving_ridge_fine_transient_adaptation.gif
```

The next solver-coupled step should replace the analytic monitor with one computed from a numerical solution, e.g. `|grad u|`, Hessian recovery, residual indicators, or an adjoint-weighted indicator.

The current solution-driven visualization example is:

```bash
uv run python -m examples.applications.solution_driven_transient_adaptation_2d
```

It uses the exact transient solution of a linear advection problem as the
solution provider, builds a monitor from the solution-gradient magnitude,
adapts the mesh at each physical time, and writes a ParaView series:

```text
outputs/solution_driven_transient_adaptation_2d/solution_driven_transient_adaptation.pvd
```

Useful ParaView arrays are `solution`, `solution_gradient_monitor`,
`cell_area_ratio_to_initial`, and `mesh_displacement`.

## OpenFOAM ALE Coupling Check

The first OpenFOAM coupling prototype is intentionally small:

```bash
uv run python -m examples.openfoam.ale_mesh_flux_correction
```

It generates a tiny OpenFOAM scalar-transport solver inside Docker, moves a
four-cell mesh to target points produced by this adapter, and solves scalar
transport steps with

```text
phi = -mesh.phi()
```

This is the OpenFOAM finite-volume form of the zero-physical-velocity ALE flux:
the only face flux is the correction induced by mesh motion. The diagnostic
includes a constant scalar and a nonconstant linear scalar. The constant case
catches geometric conservation failures; the linear case reports whether the
current OpenFOAM coupling is good enough as a solution remap.

## OpenFOAM Cylinder Flow Adaptation

The real solver demonstrator is:

```bash
uv run python -m examples.openfoam.cylinder_flow_adaptive
```

It copies the standard OpenFOAM
`incompressible/pimpleFoam/laminar/cylinder2D` tutorial, builds a small
`adaptivePimpleFoam` wrapper, and runs the case in three phases:

1. Solve transient incompressible cylinder flow on the original mesh.
2. Read the cell-centered velocity field, build a monitor from velocity-gradient
   jumps plus the no-slip cylinder contribution, and adapt the 2D
   fixed-topology mesh.
3. Ramp the mesh motion through several absolute `constant/targetPoints`
   states, then continue the Navier-Stokes solve on the moved mesh.

The wrapper keeps the standard PIMPLE equations and OpenFOAM flux correction
after the external point motion. This is a solver-coupling prototype, not yet a
full UM2N-equivalent ALE/remap implementation: the field transfer under mesh
motion still needs to pass the stricter nonconstant diagnostic before the
cylinder result should be treated as physically validated. The adapter fixes all
physical boundary nodes, including the cylinder surface; only interior points
move. The wrapper also writes the moved `polyMesh/points` at each ramp time so
that ParaView sees the changed geometry, not only changed fields.

Useful outputs:

```text
outputs/openfoam_cylinder_adaptive/case/openfoam_cylinder_adaptive.foam
outputs/openfoam_cylinder_adaptive/case/VTK/
outputs/openfoam_cylinder_adaptive/adapter/cylinder_adapter_motion.pvd
outputs/openfoam_cylinder_adaptive/metrics.json
```

For a more meaningful unsteady view, run or inspect the longer case:

```bash
uv run python -m examples.openfoam.cylinder_flow_adaptive \
  --output-dir outputs/openfoam_cylinder_unsteady_adaptive
```

Open the `.foam` marker or the `VTK/` directory in ParaView for the OpenFOAM
`U/p` time series. Open `cylinder_adapter_motion.pvd` to inspect the extracted
2D adapter view with `solution_gradient_monitor`, `cell_area_ratio_to_initial`,
and `mesh_displacement`.

See also [UM2N Coupling Protocol](um2n_coupling_protocol.md) for the baseline
comparison plan against Monge-Ampere mesh movement.
