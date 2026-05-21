# V-Formation ALE Benchmark

This note records the three-run V-formation comparison after switching the
differentiable Firedrake adapter to direct ALE-in-step ordering.

## Reusable Runner

The reusable CLI is:

```bash
scripts/run_v_formation_compare.sh
```

It wraps:

```bash
uv run python -m examples.firedrake.v_formation_compare
```

The script creates one run directory containing:

- `meshes/`: copied or generated coarse/fine `.msh` files;
- `fine/fixed/`: fine fixed reference with `.pvd`/`.vtu` ParaView series;
- `coarse/fixed/`: coarse fixed solution with `.pvd`/`.vtu` series;
- `coarse/adapted/`: coarse adaptive solution with `.pvd`/`.vtu` series;
- `comparison/comparison_table.csv`;
- `comparison/comparison_table.md`;
- `comparison/error_summary.json`;
- `comparison/vorticity_comparison.png`;
- `comparison/PARAVIEW_SERIES.md`.

If `--coarse-mesh` and `--fine-mesh` are not provided, the script generates
uniform-size V-formation meshes. It deliberately does not add the wake/cylinder
distance refinement used by `multi_cylinder_meshes.py`; Gmsh is run with a
constant background mesh size and boundary/curvature mesh-size extension
disabled. The cylinder boundaries still need enough segments to represent the
holes, but there is no deliberate refinement halo around the cylinders or wake.

Example with existing meshes and aggressive adaptive rerun:

```bash
scripts/run_v_formation_compare.sh \
  --output-dir outputs/vformation_manual/adapt5_gs50 \
  --coarse-mesh outputs/multi_cylinder_mesh_family/v_formation/v_formation_coarse.msh \
  --fine-mesh outputs/multi_cylinder_mesh_family/v_formation/v_formation_fine.msh \
  --dt 0.002 --steps 500 --save-every 50 \
  --adapt-every 5 \
  --monitor-frame current \
  --monitor-kind velocity-gradient \
  --monitor-scale 3.0 \
  --adapter-profile regularized \
  --adapter-steps 12 \
  --adapter-lr 8e-4 \
  --max-grid-speed 50.0 \
  --adaptation-relaxation 1.0
```

Example generating uniform coarse/fine meshes:

```bash
scripts/run_v_formation_compare.sh \
  --run-id uniform_c005_f0016 \
  --coarse-mesh-size 0.05 \
  --fine-mesh-size 0.016
```

To reuse existing fixed runs and rerun only `coarse/adapted` in the same run
directory:

```bash
scripts/run_v_formation_compare.sh \
  --output-dir outputs/vformation_manual/adapt5_gs50 \
  --coarse-mesh outputs/multi_cylinder_mesh_family/v_formation/v_formation_coarse.msh \
  --fine-mesh outputs/multi_cylinder_mesh_family/v_formation/v_formation_fine.msh \
  --only-adaptive \
  --force-adaptive \
  --adapt-every 5 \
  --max-grid-speed 50.0
```

To rebuild only tables and plots from an existing run:

```bash
scripts/run_v_formation_compare.sh \
  --output-dir outputs/vformation_manual/adapt5_gs50 \
  --postprocess-only
```

## Setup

Case: `v_formation` from `examples/firedrake/multi_cylinder_meshes.py`.

Runs:

- `fine/fixed`: fine mesh, no adaptation, used as reference.
- `coarse/fixed`: coarse mesh, no adaptation.
- `coarse/adapted`: coarse mesh, differentiable adapter every 20 steps.

Common parameters:

```text
dt = 0.002
steps = 500
final_time = 1.0
save_every = 50
monitor_frame = current
monitor_scale = 3.0
adapter_profile = regularized
adapter_steps = 12
adapter_lr = 8e-4
max_grid_speed = 1.0
```

Output directory:

```text
outputs/v_formation_ale_benchmark
```

## Runtime

```text
run              vertices  cells   adaptations  wall_time_s  solve_s
fine/fixed       17874     34894   0            206.85       190.51
coarse/fixed     1291      2366    0             28.05        18.39
coarse/adapted   1291      2366    25            35.30        19.13
```

The adaptive coarse run adds about `25.8%` wall-time overhead relative to the
coarse fixed run. It remains about `5.86x` faster than the fine fixed reference.

All 25 adaptations were accepted. Final orientation flips: `0`.

## Error Against Fine Fixed Reference

Postprocessing follows the UM2N comparison pattern more closely than the older
Cartesian-grid sampler: each coarse/adapted field is linearly interpolated to
the fine-reference VTU nodes, then L2/relative L2 are integrated over the
fine-reference triangles. The comparison image uses triangular `tripcolor`
rendering rather than `imshow`, so field plots do not show Cartesian pixels.
To match the UM2N paper style, vorticity and vorticity-error maps use fixed
clipped color limits of `[-100, 100]` and omit colorbars; the CSV/JSON errors
remain computed from unclipped fields.
The summary below excludes `t = 0`.

```text
metric                     coarse/fixed   coarse/adapted   change
mean_velocity_l2           0.0911609      0.0888820        -2.50%
mean_velocity_relative_l2  0.0781568      0.0761798        -2.53%
mean_velocity_linf         0.937897       0.947941         +1.07%

mean_vorticity_l2          9.89258        9.89749          +0.05%
mean_vorticity_relative_l2 0.281363       0.281428         +0.02%
mean_vorticity_linf        415.135        417.475          +0.56%
```

Worst-time L2 errors:

```text
metric             coarse/fixed   coarse/adapted   change
max_velocity_l2    0.170640       0.176795         +3.61%
max_vorticity_l2   11.9072        12.0467          +1.17%
```

At final time `t = 1.0`:

```text
metric                  coarse/fixed   coarse/adapted
velocity_l2             0.170640       0.176795
velocity_relative_l2    0.142998       0.148156
velocity_linf           1.09377        1.04220
vorticity_l2            11.9072        12.0467
vorticity_relative_l2   0.328996       0.332851
vorticity_linf          478.536       467.819
```

## Interpretation

On this V-formation benchmark, coarse adaptation gives a small improvement in
mean velocity L2 error over the saved time series, but vorticity errors and
worst-time L2 errors do not improve. The current adapter configuration
therefore is not yet an accuracy win.

The result suggests that the ALE-in-step plumbing is stable on the benchmark,
but the adaptation policy probably needs tuning for this flow: monitor choice,
adaptation frequency, relaxation, and max-grid-speed limits should be swept
before claiming that adaptation reliably beats the fixed coarse mesh.

## Aggressive Every-5 Adaptation

The following more aggressive configuration was also tested:

```text
mesh = outputs/multi_cylinder_mesh_family/v_formation/v_formation_coarse.msh
dt = 0.002
steps = 500
save_every = 50
adapt_every = 5
monitor_frame = current
monitor_kind = velocity-gradient
monitor_scale = 3.0
adapter_profile = regularized
adapter_steps = 12
adapter_lr = 8e-4
max_grid_speed = 50.0
adaptation_relaxation = 1.0
```

Output:

```text
outputs/v_formation_ale_benchmark/coarse/adapted_every5_gs50
```

Runtime and mesh diagnostics:

```text
run                         adaptations  wall_time_s  max_grid_speed  max_displacement  orientation_flips
coarse/fixed                0             28.05        0.0             0.0               0
coarse/adapted_20_gs1       25            35.30        1.0000          0.05489           0
coarse/adapted_5_gs50       100           42.53        4.9095          0.20757           0
```

Error against `fine/fixed`, excluding `t = 0`:

```text
metric                     coarse/fixed   adapted_20_gs1   adapted_5_gs50
mean_velocity_l2           0.0689081      0.0672128        0.0861498
max_velocity_l2            0.129148       0.133896         0.160895
mean_vorticity_l2          10.0112        9.93085          10.8086
max_vorticity_l2           12.2260        12.2949          14.1193
mean_velocity_linf         0.820879       0.803661         0.823542
mean_vorticity_linf        299.871        297.560         271.161
```

Relative to `coarse/fixed`, the aggressive run changes the main errors by:

```text
mean_velocity_l2   +25.02%
max_velocity_l2    +24.58%
mean_vorticity_l2   +7.96%
max_vorticity_l2   +15.49%
mean_vorticity_linf -9.57%
wall_time_s        +51.62%
```

Interpretation: `adapt_every = 5` with `max_grid_speed = 50` is too aggressive
for this setup. It accepts all mesh moves and keeps valid orientation, but the
large accumulated displacement worsens L2 accuracy relative to both the fixed
coarse run and the milder `adapt_every = 20`, `max_grid_speed = 1` run. It only
improves the vorticity Linf metric, which is not enough to justify the accuracy
and timing cost.
