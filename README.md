# diff-mesh-adapter

Differentiable fixed-topology mesh adaptation for meshes initialized by external mesh generators.

The project does not differentiate a mesh generator. Instead, a generator such as Gmsh creates an initial mesh, then this package keeps connectivity fixed and optimizes nodal coordinates with Torch autograd.

V1 supports 2D ordered cell blocks and 3D tetrahedral meshes. The 2D workflows demonstrate Gmsh triangle-mesh area equalization and monitor-based r-adaptation for advecting features. The 3D workflow demonstrates fixed-topology tetrahedral mesh repair with ParaView VTU/PVD output.

Optimization keeps boundary nodes fixed and can combine monitor equidistribution, movement penalty, cell-shape penalty, displacement-smoothness penalty, and positive-measure barrier terms.

Design notes are kept in [docs/adaptation_design.md](docs/adaptation_design.md).
The UM2N/ALE comparison protocol is in [docs/um2n_coupling_protocol.md](docs/um2n_coupling_protocol.md).

## Repository layout

- `src/diff_mesh_adapter/`: library code. `adapt.py`, `geometry.py`, and `mesh.py` are the core fixed-topology adapter; `io.py` reads external meshes; `visualization.py` writes plots/VTK artifacts; `monitors.py` contains reusable analytic monitor functions.
- `examples/templates/`: reusable experiment setup helpers, such as small Gmsh mesh generators and mesh perturbation helpers.
- `examples/basics/`: small demonstrations of one idea at a time.
- `examples/applications/`: larger application-style workflows.
- `examples/firedrake/`: Firedrake/Movement baselines run through the external `firedrake-run` environment.
- `examples/openfoam/`: Docker-based OpenFOAM coupling checks and real solver demos.
- `tests/`: unit tests and integration tests for the examples.

## Install

```bash
uv sync --extra dev
```

## Example

```bash
uv run python -m examples.basics.gmsh_area_equalization
uv run python -m examples.applications.advecting_monitor_2d
uv run python -m examples.applications.solution_driven_transient_adaptation_2d
uv run python -m examples.applications.transient_moving_ridge_2d
uv run python -m examples.applications.gmsh_tet_mesh_repair
uv run python -m examples.firedrake.monge_ampere_cylinder_adaptation
uv run python -m examples.firedrake.navier_stokes_monge_ampere_cylinder
uv run python -m examples.firedrake.um2n_monge_ampere_cylinder
uv run python -m examples.firedrake.um2n_diff_adapter_cylinder
uv run python -m examples.openfoam.ale_mesh_flux_correction
uv run python -m examples.openfoam.cylinder_flow_adaptive
```

## Test

```bash
uv run pytest
```
