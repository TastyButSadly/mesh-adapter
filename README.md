# diff-mesh-adapter

Differentiable fixed-topology mesh adaptation for meshes initialized by external mesh generators.

The project does not differentiate a mesh generator. Instead, a generator such as Gmsh creates an initial mesh, then this package keeps connectivity fixed and optimizes nodal coordinates with Torch autograd.

V1 supports 2D ordered cell blocks, with a Gmsh triangle-mesh workflow for cell-area equalization.

## Install

```bash
uv sync --extra dev
```

## Example

```bash
uv run python examples/gmsh_area_equalization.py
```

## Test

```bash
uv run pytest
```
