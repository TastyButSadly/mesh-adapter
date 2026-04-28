# UM2N Reference Files

This folder keeps the local files copied from the UM2N repository for the
cylinder Navier-Stokes reference run.

- `meshes/cylinder_015.msh`: UM2N cylinder mesh used by `test_flow_past_cylinder_demo_ale.py`.
- `source/flow_past_cylinder_demo_ale_source.py`: original UM2N ALE demo for reference.

The runnable local baseline is:

```bash
uv run python -m examples.firedrake.um2n_monge_ampere_cylinder
```

It does not load or train a neural model. It keeps the UM2N Navier-Stokes setup
and uses Movement's `MongeAmpereMover` with the positive
`1 + scale * normalized_monitor` density shape used by UM2N's Monge-Ampere
dataset generator. The local default is `scale=0.2` plus `--max-grid-speed 5.0`
so each mesh update is a single UM2N-style correction with bounded grid
velocity.
