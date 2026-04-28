# Examples

The examples are organized by role:

- `templates/`: reusable setup helpers for experiments and tests.
- `basics/`: compact examples for area equalization, monitor adaptation, and mesh repair.
- `applications/`: longer workflows that look closer to real use cases.
- `openfoam/`: Docker-based OpenFOAM coupling prototypes.
- `firedrake/`: Firedrake/Movement baselines run through the external
  `firedrake-run` environment.

Run examples from the repository root with module syntax:

```bash
uv run python -m examples.basics.gmsh_area_equalization
uv run python -m examples.basics.gmsh_mesh_repair
uv run python -m examples.applications.advecting_monitor_2d
uv run python -m examples.applications.solution_driven_transient_adaptation_2d
uv run python -m examples.applications.transient_moving_ridge_2d
uv run python -m examples.applications.gmsh_tet_mesh_repair
uv run python -m examples.firedrake.monge_ampere_cylinder_adaptation
uv run python -m examples.firedrake.navier_stokes_monge_ampere_cylinder
uv run python -m examples.firedrake.um2n_monge_ampere_cylinder
uv run python -m examples.openfoam.ale_mesh_flux_correction
uv run python -m examples.openfoam.cylinder_flow_adaptive
```

The OpenFOAM examples use the local Docker image `opencfd/openfoam-default:2512`.
`ale_mesh_flux_correction` is a tiny scalar-transport diagnostic for constant
and linear fields under mesh motion. `cylinder_flow_adaptive` copies the standard
OpenFOAM `pimpleFoam/laminar/cylinder2D` tutorial, solves the transient cylinder
flow, computes a monitor from the velocity-gradient magnitude, moves the
fixed-topology mesh with adapter-produced `targetPoints`, continues the
Navier-Stokes solve on the moved mesh, and writes `metrics.json` for later
adapter timing comparisons.

The Firedrake examples require the external `firedrake-run` wrapper.
`monge_ampere_cylinder_adaptation` is a mesh-only Movement baseline.
`navier_stokes_monge_ampere_cylinder` runs a short cylinder Navier-Stokes solve,
builds a monitor from the velocity-gradient magnitude, adapts a target mesh with
Movement's `MongeAmpereMover`, then continues the solve while moving the live
mesh with ALE-relative convection `u - u_grid`.
`um2n_monge_ampere_cylinder` uses the copied UM2N cylinder mesh and ALE
Navier-Stokes setup, but replaces the UM2N neural coordinate prediction with
Movement's Monge-Ampere mover. It applies one mesh correction per adaptation
step and limits the resulting grid velocity with `--max-grid-speed`.
