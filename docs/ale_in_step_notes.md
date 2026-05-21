# ALE In-Step Coupling Notes

This note records the working hypothesis, implementation changes, and remaining
checks for moving the Firedrake differentiable-adapter coupling from
post-solve adaptation toward direct ALE-in-step mesh motion.

## Problem Statement

The previous Firedrake differentiable-adapter loop was:

```text
solve PDE step on current mesh
build monitor from u_next
adapt mesh to target coordinates
set mesh.coordinates = target
set u_grid = (target - current) / dt
use u_grid in the next PDE step
```

This is neither a clean indirect ALE method nor a clean direct ALE method.
Indirect ALE needs an explicit conservative remap or projection after the
rezoning step. Direct ALE needs the mesh velocity to be known before solving the
time step whose geometry is moving.

## Literature-Guided Hypothesis

For fixed-topology Firedrake experiments, the cleaner first target is direct
ALE-in-step coupling:

```text
given x_n, u_n
build monitor from u_n
adapt to target x_{n+1}
limit target for validity and max grid speed
set u_grid = (x_{n+1} - x_n) / dt
solve the PDE step with ALE-relative convection
accept u_{n+1} and x_{n+1}
```

This matches the ALE conservation-law form

```text
F_ALE = F(U) - U tensor u_grid
```

or, for the incompressible Navier-Stokes form used here, the existing relative
advection velocity:

```text
u_adv = u_now - u_grid
```

The direct ALE path is only a reasonable claim if we also check geometric
conservation / free-stream preservation. A constant solution must not change
only because the mesh moved.

## Current Implementation Change

`examples/firedrake/_um2n_diff_adapter_cylinder_firedrake.py` now computes the
target mesh before `_solve_step(state)` on adaptation steps:

```text
current_coord = x_n
build monitor from u_now
raw_target_coord = adapter(current mesh, monitor)
target_coord = current_coord + relaxation * (raw_target_coord - current_coord)
mesh.coordinates = target_coord
u_grid = (target_coord - current_coord) / dt
_solve_step(state)
u_now = u_next
p_now = p_next
```

For `monitor_frame == "reference"`, the solution used to build the monitor is
projected from the current adapted mesh back to the initial reference mesh
before the adapter request, preserving the UM2N-style monitor frame.

The step now records:

- `coupling_scheme = "direct_ale_in_step"`;
- `time_start` and `time_end` for the adaptation;
- `accepted_adaptation`;
- `relaxation_backtracks`;
- raw and applied grid speeds;
- raw and relaxed mesh quality.

The implementation also backtracks relaxation if the proposed target creates
orientation flips. If the target remains invalid after the configured backtrack
budget, the adaptation is rejected and the step is solved with zero mesh
velocity.

## What This Does Not Yet Prove

This change fixes the ordering of the coupling, but it does not by itself prove
that the whole Firedrake Navier-Stokes discretization satisfies the discrete
geometric conservation law.

Required follow-up checks after the model-problem and Firedrake validations
below:

1. Sensitivity to `max_grid_speed_limit` and adaptation relaxation.
2. Comparison of `reference` and `current` monitor frames after the ordering
   change.
3. Decision on whether large moves should use ALE substeps instead of a single
   relaxed step.

## Test Log

Initial checks after introducing direct ALE-in-step ordering:

```bash
uv run python -m py_compile examples/firedrake/_um2n_diff_adapter_cylinder_firedrake.py
uv run pytest tests/test_ale.py tests/test_ale_in_step_shock_validation.py tests/test_adapter.py tests/test_geometry.py
```

Result:

```text
14 passed
```

Firedrake smoke runs through the `firedrake-run` wrapper:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder \
  --output-dir outputs/ale_in_step_smoke \
  --steps 2 --adapt-every 1 --save-every 1 \
  --adapter-steps 2 --adapter-lr 1e-4 \
  --monitor-frame current --adapter-profile regularized
```

Observed:

```text
coupling_scheme = direct_ale_in_step
adaptations = 2
max_grid_speed = 0.2001351717355604
final orientation_flips = 0
```

Reference-frame monitor smoke:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder \
  --output-dir outputs/ale_in_step_smoke_reference \
  --steps 2 --adapt-every 1 --save-every 1 \
  --adapter-steps 2 --adapter-lr 1e-4 \
  --monitor-frame reference --adapter-profile regularized
```

Observed:

```text
coupling_scheme = direct_ale_in_step
monitor_frame = reference
adaptations = 2
max_grid_speed = 0.39448652357981784
final orientation_flips = 0
```

Grid-speed limiter smoke:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder \
  --output-dir outputs/ale_in_step_limiter_smoke \
  --steps 1 --adapt-every 1 --save-every 1 \
  --adapter-steps 2 --adapter-lr 1e-4 \
  --monitor-frame current --adapter-profile regularized \
  --max-grid-speed 0.05
```

Observed:

```text
raw_grid_speed = 0.2001351717355604
applied_grid_speed = 0.04999999999999449
applied_relaxation = 0.24983114944965923
orientation_flips = 0
```

Mixed adaptation/non-adaptation smoke:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder \
  --output-dir outputs/ale_in_step_mixed_smoke \
  --steps 4 --adapt-every 2 --save-every 2 \
  --adapter-steps 2 --adapter-lr 1e-4 \
  --monitor-frame current --adapter-profile regularized
```

Observed:

```text
adaptations = 2
adaptation steps = 2, 4
max_grid_speed = 0.20013495877571966
final orientation_flips = 0
```

These are runtime smoke tests only. They verify the new ordering, adapter
exchange, reference/current monitor paths, non-adaptive steps, and grid-speed
limiting. They do not replace the required GCL/free-stream preservation test.

Mathematical ALE validation was added in
`examples/applications/ale_in_step_shock_validation.py`. It uses a 1D
conservative finite-volume ALE update for Burgers' equation,

```text
h_i^{n+1} U_i^{n+1}
=
h_i^n U_i^n
- dt (G_{i+1/2}^{ALE} - G_{i-1/2}^{ALE}),

G^{ALE}(U; w) = F(U) - w U,
F(U) = U^2 / 2.
```

The shock test is the Riemann problem `U_L = 1`, `U_R = 0`, with exact
Rankine-Hugoniot shock speed

```text
s = (F(U_L) - F(U_R)) / (U_L - U_R) = 0.5.
```

The adaptive run rebuilds the mesh target from the current numerical solution
and performs adaptation on every time step before the ALE update. The fixed
coarse run uses the same cell count without mesh movement. The fine reference is
a fixed mesh with eight times more cells.

```bash
uv run python -m examples.applications.ale_in_step_shock_validation \
  --output-dir outputs/ale_in_step_shock_validation \
  --coarse-cells 120 --fine-cells 960 --final-time 0.4
```

Observed:

```text
GCL/free-stream:
  ALE max constant-state error = 8.007905449858299e-11
  naive no-mesh-flux max error = 3052807.809125752

Coarse fixed:
  shock_speed = 0.5016850462716824
  front_width_10_90 = 0.022465909668797746
  L1 exact = 0.006321510138418451
  L1 fine = 0.006310714894061117

Coarse adaptive every step:
  shock_speed = 0.5000126680713995
  front_width_10_90 = 0.0017674780875776008
  L1 exact = 0.0004511869821934215
  L1 fine = 0.00037219357558186737
  mass_error = -2.831068712794149e-15
  bounds_violation = 0.0
  near_shock_dx_ratio = 0.06848181991818914
```

Interpretation: for this scalar conservation-law test, the direct ALE update is
conservative, preserves a constant state under mesh motion, tracks the analytic
shock speed, and the every-step adaptive mesh strongly reduces front smearing.
This validates the ALE ordering at the finite-volume model-problem level. It is
still not a proof that the Firedrake Navier-Stokes discretization satisfies a
discrete GCL; that needs a Firedrake-specific constant-state or manufactured
solution test.

Firedrake every-step smoke:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder \
  --output-dir outputs/ale_in_step_firedrake_every_step \
  --steps 6 --adapt-every 1 --save-every 3 \
  --adapter-steps 2 --adapter-lr 1e-4 \
  --monitor-frame current --adapter-profile regularized
```

Observed:

```text
coupling_scheme = direct_ale_in_step
steps = 6
adapt_every = 1
adaptations = 6
max_grid_speed = 0.2001351717355604
final orientation_flips = 0
all adaptations accepted
```

Firedrake-specific free-stream/GCL check:

```bash
firedrake-run python3 -m examples.firedrake._ale_gcl_validation_firedrake \
  --output-dir outputs/firedrake_ale_gcl_validation \
  --nx 12 --ny 12 --steps 20 --dt 0.01 --amplitude 0.035
```

This test uses the same Firedrake ordering as the adapter path:

```text
set mesh.coordinates = x_{n+1}
set u_grid = (x_{n+1} - x_n) / dt
solve with the ALE-relative advective form
```

For the vector field it mirrors the current Navier-Stokes-style advective
operator:

```text
(u - u_n) / dt + ((u_n - u_grid) . grad) u_mid = 0.
```

It also runs scalar ALE checks on the same moving mesh:

- a constant scalar free-stream check;
- a smooth non-constant manufactured scalar
  `q(x, y) = 1 + 0.2 sin(2 pi x) sin(2 pi y)`.

For the smooth scalar the physical velocity is zero, so the exact Eulerian
solution remains the same function of physical coordinates while the mesh moves.
The mesh motion is a smooth fixed-boundary interior deformation, so cell volumes
change while the physical domain area stays constant.

Observed:

```text
max_vector_linf_error = 8.326672684688674e-17
max_scalar_linf_error = 0.0
max_scalar_mass_error = 1.3322676295501878e-15
max_smooth_scalar_linf_error = 0.0006910197632761239
max_smooth_scalar_l2_error = 0.00022279574056658868
max_smooth_scalar_mass_error = 1.1482707043986018e-05
max_domain_area_error = 8.881784197001252e-16
max_grid_speed = 0.31852447817922547
max_orientation_flips = 0
```

Refinement check for the smooth non-constant scalar:

```text
nx=12, steps=20, dt=0.01:
  max_smooth_scalar_l2_error = 0.00022279574056658868
  max_smooth_scalar_linf_error = 0.0006910197632761239
  max_smooth_scalar_mass_error = 1.1482707043986018e-05

nx=24, steps=40, dt=0.005:
  max_smooth_scalar_l2_error = 0.00005835879144132784
  max_smooth_scalar_linf_error = 0.00017283509811161224
  max_smooth_scalar_mass_error = 0.00000588598842488075

nx=48, steps=80, dt=0.0025:
  max_smooth_scalar_l2_error = 0.00001549330151801674
  max_smooth_scalar_linf_error = 0.000043189512370700456
  max_smooth_scalar_mass_error = 0.0000029613386138827025
```

The L2 error drops by factors of about `3.82` and `3.77` under uniform
refinement with `dt` halved, which is the expected convergence behavior for this
smooth CG manufactured check.

Stronger mesh motion with the smooth scalar:

```bash
firedrake-run python3 -m examples.firedrake._ale_gcl_validation_firedrake \
  --output-dir outputs/firedrake_ale_gcl_validation_stronger \
  --nx 24 --ny 24 --steps 40 --dt 0.005 --amplitude 0.07
```

Observed:

```text
max_vector_linf_error = 8.326672684688674e-17
max_scalar_linf_error = 0.0
max_scalar_mass_error = 1.4432899320127035e-15
max_smooth_scalar_l2_error = 0.00012672281520216354
max_smooth_scalar_linf_error = 0.0003796327405991562
max_smooth_scalar_mass_error = 0.000022992029668378322
max_domain_area_error = 8.881784197001252e-16
max_grid_speed = 0.6375696619111462
max_orientation_flips = 0
```

Interpretation: the specific Firedrake advective ALE form used by the
Navier-Stokes cylinder path preserves a free-stream constant state under mesh
motion to roundoff. This addresses the immediate GCL/free-stream concern for
the current `u_adv = u_now - u_grid` ordering. For a smooth non-constant passive
scalar, the same ALE ordering gives convergent physical-coordinate transport and
small mass error. The non-constant scalar mass is not preserved to roundoff,
which is expected for this continuous Galerkin advective form; exact local
conservation of discontinuous fronts would need a DG/FV conservative ALE
transport form rather than this Navier-Stokes-style advective check.

## Relationship To PDE-Constrained Mesh Optimization

The paper "PDE-Constrained High-Order Mesh Optimization"
(`arXiv:2507.01917v1`) optimizes mesh coordinates with a PDE residual
constraint and adjoint sensitivities:

```text
min_x alpha F_P(u(x), x) + F_mu(x)
subject to R_P(u; x) = 0
```

This repository still uses a lagged monitor:

```text
M_n = stop_gradient(Monitor(u_n))
min_x F_monitor(x; M_n) + F_mesh(x)
```

The paper is still useful for the mesh-motion side: displacement filtering,
Jacobian/TMOP-style quality terms, and line-search validity checks. Those are
orthogonal to the ALE ordering fixed here.

## References To Keep In Scope

- ALE methods with rezoning/remap: indirect ALE requires a conservative remap
  stage after mesh optimization.
- Direct moving-mesh ALE: mesh velocity must enter the PDE step whose domain is
  moving.
- Geometric conservation law: a moving-mesh discretization must preserve a
  constant state under mesh motion.
- PDE-constrained mesh optimization: use adjoints for the full
  `du/dx` sensitivity when the target is a true PDE objective rather than a
  lagged monitor.
