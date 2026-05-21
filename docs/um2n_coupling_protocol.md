# UM2N Coupling Protocol

This note records the minimal protocol to reproduce the useful part of the
UM2N paper workflow in this repository.

## What UM2N Actually Does

For the cylinder ALE demo, UM2N runs the PDE in Firedrake. The mesh mover only
produces new coordinates. The PDE coupling is handled by the solver loop:

1. Solve one Navier-Stokes step on the current mesh.
2. Project the current solution back to the reference mesh before computing the
   next monitor.
3. Build a monitor from the velocity-gradient magnitude.
4. Produce new coordinates with the UM2N model.
5. Set the live mesh coordinates to those new coordinates.
6. Compute grid velocity:

   ```text
   u_grid = (x_new - x_old) / dt
   ```

7. Use ALE convection in the next PDE step:

   ```text
   u_adv = u_now - u_grid
   dot(u_adv, grad(u_mid))
   ```

8. Project or assign `u` and `p` onto the new mesh state, then update the stored
   previous adapted coordinates.

The key point is that Monge-Ampere or UM2N only answers "where should the nodes
go?" Solver correctness comes from grid velocity, relative advection or fluxes,
and solution projection/remapping.

## Baseline Comparison Shape

Use the same PDE case, mesh, time step, monitor, and output metrics for both
adapters:

```text
solve PDE step
project solution to reference mesh for monitor
compute monitor
adapt mesh with method A or B
compute grid velocity / mesh flux
remap fields onto the moved mesh
continue PDE solve with ALE-relative velocity or flux
```

The two adaptation methods should differ only in the coordinate generator:

- `monge_ampere`: Movement/Firedrake `MongeAmpereMover` with the same monitor.
- `torch_monitor_weighted_area`: this repository's differentiable fixed-topology
  optimizer with the same monitor.

The first runnable Monge-Ampere baseline is:

```bash
uv run python -m examples.firedrake.monge_ampere_cylinder_adaptation
```

It generates a cylinder-channel mesh, adapts it with Movement, and writes:

```text
outputs/firedrake_monge_ampere_cylinder/monge_ampere_cylinder_adaptation.pvd
outputs/firedrake_monge_ampere_cylinder/metrics.json
```

The first runnable Navier-Stokes/ALE version of the same baseline is:

```bash
uv run python -m examples.firedrake.navier_stokes_monge_ampere_cylinder
```

It writes:

```text
outputs/firedrake_ns_monge_ampere_cylinder/navier_stokes_monge_ampere_cylinder.pvd
outputs/firedrake_ns_monge_ampere_cylinder/metrics.json
```

The older local implementation moved the live PDE mesh toward a stale target in
substeps. That was not the UM2N coupling and should not be used as a reference.
The current UM2N reference-mesh baseline applies at most one correction per
physical step: solve, compute the monitor from the new solution, produce a new
coordinate target, move the mesh once, and use
`u_grid = (x_new - x_old) / dt` only for the next Navier-Stokes step.
This ordering matches the original UM2N-style engineering demo but is not the
preferred mathematical coupling for the differentiable adapter.

The UM2N reference-mesh Monge-Ampere baseline is:

```bash
uv run python -m examples.firedrake.um2n_monge_ampere_cylinder
```

It uses `examples/firedrake/um2n_reference/meshes/cylinder_015.msh` copied from
UM2N and keeps the UM2N cylinder Navier-Stokes/ALE setup. It does not load or
train a neural network. The coordinate generator is Movement's
`MongeAmpereMover`, using the positive `1 + scale * normalized_monitor`
density shape from UM2N's Monge-Ampere dataset generator. The local default is
`scale=0.2` with `--max-grid-speed 5.0`, matching the UM2N ALE demo's practical
guard that avoids accepting mesh moves with very large `u_grid`.

The matching differentiable-adapter run is:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder
```

It keeps the same copied UM2N cylinder mesh, Navier-Stokes splitting, ALE
relative advection, reference-mesh monitor construction, monitor clipping and
normalization, and grid-speed relaxation guard. The only changed component is
the coordinate generator: the Monge-Ampere or UM2N network coordinate update is
replaced by `adapt_monitor_weighted_area`.

The default adapter exchange remains the file-based `.npz` protocol. For
timing-sensitive local Docker runs, use `--adapter-transport tcp`; it keeps the
same adapter service process alive, sends array payloads over one length-prefixed
TCP connection, and caches fixed connectivity after the first request.

The differentiable-adapter Firedrake path now uses a direct ALE-in-step order:
build the monitor from the accepted state at the beginning of an adaptation
step, compute the target coordinates, set
`u_grid = (x_target - x_current) / dt`, solve the Navier-Stokes step with
`u_adv = u_now - u_grid`, and then accept the new solution on the moved mesh.
The rationale and remaining GCL checks are recorded in
`docs/ale_in_step_notes.md`.

For faster engineering runs where exact UM2N reference-frame parity is not the
goal, skip the reference-mesh projection and compute the monitor on the live
mesh:

```bash
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder --monitor-frame current
```

The default `--monitor-frame reference` is kept for comparison against the UM2N
pipeline. It temporarily restores the initial coordinates and projects the
current adapted solution onto that reference mesh before monitor construction.
That makes the monitor input live in the same frame as the UM2N model expects,
but it costs an additional Firedrake projection solve at every adaptation event.
The `current` mode removes that projection; it is cheaper and is the right
default for quick parameter scans of this adapter, but its monitor is no longer
bit-for-bit the UM2N reference-frame monitor.

For source-level debugging, the upstream UM2N repository is kept as a sibling
checkout at:

```text
/Users/sergei/PycharmProjects/UM2N
```

Log these metrics separately:

- `pde_solve_s`
- `monitor_s`
- `adapter_s`
- `projection_or_remap_s`
- `mesh_motion_s`
- physical quality metric, e.g. drag/lift history for cylinder flow
- mesh quality metric, e.g. tangled cells, min quality, high-monitor area ratio

## Current Repository Status

`examples.firedrake.navier_stokes_monge_ampere_cylinder` is the current
UM2N-style solver-coupled baseline. It is intentionally short by default, so it
is a reproducibility and integration check rather than a benchmark-quality wake
simulation.

`examples.openfoam.cylinder_flow_adaptive` writes `metrics.json` with adapter
timing and coarse adaptation metrics. This is enough for a first timing schema,
but not yet a complete UM2N-equivalent solver coupling.

`examples.openfoam.ale_mesh_flux_correction` is the small gatekeeper test. It
checks mesh-flux correction on a constant scalar and now also on a nonconstant
linear scalar field. The cylinder demo should not be treated as physically
validated until this remap/ALE diagnostic behaves sensibly.

## Next OpenFOAM Step

The next OpenFOAM code step should be a dedicated remap stage for fields moved
onto externally supplied `targetPoints`. For Navier-Stokes this means at least:

1. Compute mesh flux from `x_old -> x_new`.
2. Remap `U` component-wise with the mesh-motion ALE transport equation.
3. Remap/project `p` consistently enough for the next pressure correction.
4. Continue PIMPLE with relative fluxes, not with a bare point overwrite.

Only after this should the cylinder case be used for a speed comparison against
Monge-Ampere adaptation.
