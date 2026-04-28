from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    MeshState,
    adapt_monitor_weighted_area,
    cell_abs_areas,
    cell_centroids,
    cell_edge_lengths,
)

IMAGE = "opencfd/openfoam-default:2512"
TUTORIAL_CASE = "$FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/cylinder2D"


@dataclass(frozen=True)
class FoamPatch:
    name: str
    type: str
    start_face: int
    n_faces: int


@dataclass(frozen=True)
class FoamMesh2D:
    mesh: MeshState
    openfoam_points: np.ndarray
    point_to_node: np.ndarray
    faces: list[list[int]]
    owner: np.ndarray
    neighbour: np.ndarray
    patches: dict[str, FoamPatch]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real OpenFOAM cylinder-flow adaptation check.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openfoam_cylinder_adaptive"))
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--pre-end-time", type=float, default=40.0)
    parser.add_argument("--post-duration", type=float, default=20.0)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--write-interval", type=float, default=1.0)
    parser.add_argument("--motion-substeps", type=int, default=8)
    parser.add_argument("--motion-delta-t", type=float, default=0.01)
    parser.add_argument("--adapt-steps", type=int, default=35)
    parser.add_argument("--monitor-alpha", type=float, default=2.0)
    parser.add_argument("--monitor-smoothing", type=int, default=2)
    parser.add_argument("--skip-foam-to-vtk", action="store_true")
    args = parser.parse_args()

    paths = run_case(
        args.output_dir,
        image=args.image,
        pre_end_time=args.pre_end_time,
        post_duration=args.post_duration,
        delta_t=args.delta_t,
        write_interval=args.write_interval,
        motion_substeps=args.motion_substeps,
        motion_delta_t=args.motion_delta_t,
        adapt_steps=args.adapt_steps,
        monitor_alpha=args.monitor_alpha,
        monitor_smoothing=args.monitor_smoothing,
        run_foam_to_vtk=not args.skip_foam_to_vtk,
    )
    print(f"OpenFOAM case: {paths['case']}")
    print(f"ParaView marker: {paths['foam']}")
    print(f"adapter series: {paths['adapter_pvd']}")
    if paths.get("vtk") is not None:
        print(f"foamToVTK output: {paths['vtk']}")


def run_case(
    output_dir: str | Path,
    *,
    image: str = IMAGE,
    pre_end_time: float = 40.0,
    post_duration: float = 20.0,
    delta_t: float = 0.05,
    write_interval: float = 1.0,
    motion_substeps: int = 8,
    motion_delta_t: float = 0.01,
    adapt_steps: int = 35,
    monitor_alpha: float = 2.0,
    monitor_smoothing: int = 2,
    run_foam_to_vtk: bool = True,
) -> dict[str, Path | float | None]:
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    case_dir = output / "case"
    timings: dict[str, float] = {}

    started_at = time.perf_counter()
    _copy_tutorial_case(output, image)
    _write_solver(output, image)
    _docker_run(image, output, "cd /work/adaptivePimpleFoam && wmake")
    timings["setup_and_build_s"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    _docker_run(image, output, "cd /work/case && ./Allrun.pre")
    (case_dir / "constant" / "targetPoints").unlink(missing_ok=True)
    timings["case_preparation_s"] = time.perf_counter() - started_at

    write_interval_steps = _write_interval_steps(write_interval, delta_t)
    _write_control_dict(
        case_dir / "system" / "controlDict",
        start_from="startTime",
        end_time=pre_end_time,
        delta_t=delta_t,
        write_interval_steps=write_interval_steps,
    )
    started_at = time.perf_counter()
    _docker_run(image, output, "cd /work/case && adaptivePimpleFoam")
    timings["pre_adaptation_solve_s"] = time.perf_counter() - started_at

    adapt_time = _latest_time(case_dir)
    started_at = time.perf_counter()
    foam_mesh = _load_foam_mesh_2d(case_dir)
    velocity = _read_vector_field(case_dir / _time_name(adapt_time) / "U", foam_mesh.mesh.num_cells)
    monitor = _velocity_gradient_monitor(foam_mesh, velocity, alpha=monitor_alpha, smoothing_steps=monitor_smoothing)
    timings["monitor_s"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    adapted = _adapt_mesh(foam_mesh.mesh, monitor, steps=adapt_steps)
    timings["adapter_s"] = time.perf_counter() - started_at

    adapter_pvd = _write_adapter_artifacts(output / "adapter", foam_mesh.mesh, adapted.mesh, velocity, monitor, adapt_time)

    started_at = time.perf_counter()
    _run_motion_ramp(
        case_dir,
        output,
        image,
        foam_mesh,
        adapted.mesh.points.detach().cpu().numpy(),
        substeps=motion_substeps,
        delta_t=motion_delta_t,
    )
    timings["mesh_motion_ramp_s"] = time.perf_counter() - started_at

    final_time = _latest_time(case_dir) + post_duration
    _write_control_dict(
        case_dir / "system" / "controlDict",
        start_from="latestTime",
        end_time=final_time,
        delta_t=delta_t,
        write_interval_steps=write_interval_steps,
    )
    (case_dir / "constant" / "targetPoints").unlink(missing_ok=True)
    started_at = time.perf_counter()
    _docker_run(image, output, "cd /work/case && adaptivePimpleFoam")
    timings["post_adaptation_solve_s"] = time.perf_counter() - started_at

    foam_marker = case_dir / "openfoam_cylinder_adaptive.foam"
    foam_marker.write_text("", encoding="utf-8")
    vtk_dir: Path | None = None
    if run_foam_to_vtk:
        started_at = time.perf_counter()
        _docker_run(image, output, "cd /work/case && foamToVTK -time '0:'")
        timings["foam_to_vtk_s"] = time.perf_counter() - started_at
        vtk_dir = case_dir / "VTK"

    high_gradient_area_ratio = _high_gradient_area_ratio(foam_mesh.mesh, adapted.mesh, monitor)
    metrics_path = output / "metrics.json"
    _write_metrics(
        metrics_path,
        timings=timings,
        adapt_time=adapt_time,
        final_time=_latest_time(case_dir),
        initial_loss=adapted.initial_loss,
        final_loss=adapted.final_loss,
        high_gradient_area_ratio=high_gradient_area_ratio,
        settings={
            "adapter": "torch_monitor_weighted_area",
            "pre_end_time": pre_end_time,
            "post_duration": post_duration,
            "delta_t": delta_t,
            "write_interval": write_interval,
            "motion_substeps": motion_substeps,
            "motion_delta_t": motion_delta_t,
            "adapt_steps": adapt_steps,
            "monitor_alpha": monitor_alpha,
            "monitor_smoothing": monitor_smoothing,
        },
    )
    print(f"adapted at t={adapt_time:.6g}; final OpenFOAM time={_latest_time(case_dir):.6g}")
    print(f"adapter loss: {adapted.initial_loss:.6g} -> {adapted.final_loss:.6g}")
    print(f"mean high-gradient cell area ratio: {high_gradient_area_ratio:.6g}")
    print(f"metrics: {metrics_path}")

    return {
        "case": case_dir,
        "foam": foam_marker,
        "adapter_pvd": adapter_pvd,
        "vtk": vtk_dir,
        "metrics": metrics_path,
        "adapt_time": adapt_time,
        "high_gradient_area_ratio": high_gradient_area_ratio,
    }


def _adapt_mesh(mesh: MeshState, monitor: np.ndarray, *, steps: int):
    monitor_tensor = torch.as_tensor(monitor, dtype=mesh.points.dtype, device=mesh.points.device)
    reference_edges = cell_edge_lengths(mesh.points, mesh.cell_blocks).clamp_min(1e-12)

    def monitor_fn(points: torch.Tensor, cell_blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return monitor_tensor.to(device=points.device, dtype=points.dtype)

    config = AdaptationConfig(
        steps=steps,
        lr=1.0e-4,
        movement_weight=8.0e-2,
        smoothness_weight=2.5e-1,
        shape_weight=8.0e-2,
        quality_barrier_weight=8.0,
        boundary_quality_barrier_weight=30.0,
        min_cell_quality=0.42,
        min_step_cell_quality=0.35,
        edge_length_weight=1.0e-1,
        edge_length_barrier_weight=2.0,
        boundary_edge_length_barrier_weight=20.0,
        max_edge_stretch=1.25,
        min_edge_compression=0.75,
        max_step_edge_stretch=1.35,
        min_step_edge_compression=0.65,
        barrier_weight=3.0,
        grad_clip=0.12,
        early_stopping_patience=14,
        early_stopping_min_delta=1e-4,
        early_stopping_relative=False,
    )
    result = adapt_monitor_weighted_area(mesh, monitor_fn, config, reference_points=mesh.points)
    final_edges = cell_edge_lengths(result.mesh.points, result.mesh.cell_blocks).clamp_min(1e-12)
    max_edge_ratio = float((final_edges / reference_edges).max())
    min_edge_ratio = float((final_edges / reference_edges).min())
    print(f"edge ratio after adaptation: min={min_edge_ratio:.4g} max={max_edge_ratio:.4g}")
    return result


def _copy_tutorial_case(output_dir: Path, image: str) -> None:
    _docker_run(image, output_dir, f"cp -a {TUTORIAL_CASE} /work/case")


def _run_motion_ramp(
    case_dir: Path,
    output_dir: Path,
    image: str,
    foam_mesh: FoamMesh2D,
    adapted_points: np.ndarray,
    *,
    substeps: int,
    delta_t: float,
) -> None:
    if substeps <= 0:
        raise ValueError("motion_substeps must be positive")
    initial_points = foam_mesh.mesh.points.detach().cpu().numpy()
    for step in range(1, substeps + 1):
        fraction = step / substeps
        target_points = initial_points + fraction * (adapted_points - initial_points)
        _write_target_points(
            case_dir / "constant" / "targetPoints",
            foam_mesh.openfoam_points,
            foam_mesh.point_to_node,
            target_points,
        )
        current_time = _latest_time(case_dir)
        _write_control_dict(
            case_dir / "system" / "controlDict",
            start_from="latestTime",
            end_time=current_time + delta_t,
            delta_t=delta_t,
            write_interval_steps=1,
        )
        print(f"mesh-motion ramp {step}/{substeps}: t={current_time:.6g}->{current_time + delta_t:.6g}")
        _docker_run(image, output_dir, "cd /work/case && adaptivePimpleFoam")


def _write_solver(output_dir: Path, image: str) -> None:
    solver_dir = output_dir / "adaptivePimpleFoam"
    solver_dir.mkdir(parents=True, exist_ok=True)
    _docker_run(
        image,
        output_dir,
        "cp $FOAM_APP/solvers/incompressible/pimpleFoam/createFields.H "
        "$FOAM_APP/solvers/incompressible/pimpleFoam/UEqn.H "
        "$FOAM_APP/solvers/incompressible/pimpleFoam/pEqn.H "
        "$FOAM_APP/solvers/incompressible/pimpleFoam/setRDeltaT.H "
        "/work/adaptivePimpleFoam/ && "
        "cp $FOAM_APP/solvers/incompressible/pimpleFoam/correctPhi.H "
        "/work/adaptivePimpleFoam/pimpleCorrectPhi.H",
    )
    (solver_dir / "Make").mkdir(exist_ok=True)
    (solver_dir / "Make" / "files").write_text(
        """adaptivePimpleFoam.C

EXE = $(FOAM_USER_APPBIN)/adaptivePimpleFoam
""",
        encoding="utf-8",
    )
    (solver_dir / "Make" / "options").write_text(
        """EXE_INC = \\
    -I$(LIB_SRC)/finiteVolume/lnInclude \\
    -I$(LIB_SRC)/meshTools/lnInclude \\
    -I$(LIB_SRC)/sampling/lnInclude \\
    -I$(LIB_SRC)/TurbulenceModels/turbulenceModels/lnInclude \\
    -I$(LIB_SRC)/TurbulenceModels/incompressible/lnInclude \\
    -I$(LIB_SRC)/transportModels \\
    -I$(LIB_SRC)/transportModels/incompressible/singlePhaseTransportModel \\
    -I$(LIB_SRC)/dynamicMesh/lnInclude \\
    -I$(LIB_SRC)/dynamicFvMesh/lnInclude \\
    -I$(LIB_SRC)/regionFaModels/lnInclude

EXE_LIBS = \\
    -lfiniteVolume \\
    -lfvOptions \\
    -lmeshTools \\
    -lsampling \\
    -lturbulenceModels \\
    -lincompressibleTurbulenceModels \\
    -lincompressibleTransportModels \\
    -ldynamicMesh \\
    -ldynamicFvMesh \\
    -ltopoChangerFvMesh \\
    -latmosphericModels \\
    -lregionFaModels \\
    -lfiniteArea
""",
        encoding="utf-8",
    )
    (solver_dir / "adaptivePimpleFoam.C").write_text(
        r'''#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "pointIOField.H"

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "pimpleFoam with an optional externally supplied constant/targetPoints "
        "mesh move before the PIMPLE solve."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    bool adapterMoveDone = false;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                bool movedByAdapter = false;
                const fileName targetPointsPath = runTime.constantPath()/"targetPoints";

                if (isFile(targetPointsPath) && !adapterMoveDone)
                {
                    pointIOField targetPoints
                    (
                        IOobject
                        (
                            "targetPoints",
                            runTime.constant(),
                            mesh,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE
                        )
                    );

                    if (targetPoints.size() != mesh.points().size())
                    {
                        FatalErrorInFunction
                            << "constant/targetPoints contains " << targetPoints.size()
                            << " points, but the mesh contains " << mesh.points().size()
                            << exit(FatalError);
                    }

                    Info<< "Moving mesh to constant/targetPoints" << nl << endl;
                    mesh.movePoints(targetPoints);
                    mesh.write();
                    movedByAdapter = true;
                    adapterMoveDone = true;
                }
                else
                {
                    mesh.controlledUpdate();
                }

                if (mesh.changing() || movedByAdapter)
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        phi = mesh.Sf() & Uf();

                        #include "pimpleCorrectPhi.H"

                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "UEqn.H"

            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        runTime.write();
        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}
''',
        encoding="utf-8",
    )


def _write_control_dict(
    path: Path,
    *,
    start_from: str,
    end_time: float,
    delta_t: float,
    write_interval_steps: int,
) -> None:
    path.write_text(
        _foam_header("dictionary", "controlDict")
        + f"""
application     adaptivePimpleFoam;
startFrom       {start_from};
startTime       0;
stopAt          endTime;
endTime         {_foam_float(end_time)};
deltaT          {_foam_float(delta_t)};
writeControl    timeStep;
writeInterval   {write_interval_steps};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
adjustTimeStep  no;
maxCo           1;
maxDeltaT       {_foam_float(delta_t)};
""",
        encoding="utf-8",
    )


def _load_foam_mesh_2d(case_dir: Path) -> FoamMesh2D:
    poly = case_dir / "constant" / "polyMesh"
    openfoam_points = _read_points(poly / "points")
    faces = _read_faces(poly / "faces")
    owner = _read_label_list(poly / "owner")
    neighbour = _read_label_list(poly / "neighbour")
    patches = _read_boundary(poly / "boundary")

    empty_patches = [patch for patch in patches.values() if patch.type == "empty"]
    if not empty_patches:
        raise ValueError("Expected a one-cell-thick 2D OpenFOAM mesh with empty front/back patches")
    base_patch = empty_patches[0]

    node_by_xy: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []

    def node_index(point_index: int) -> int:
        x, y, _ = openfoam_points[point_index]
        key = _xy_key(x, y)
        existing = node_by_xy.get(key)
        if existing is not None:
            return existing
        node_by_xy[key] = len(nodes)
        nodes.append((float(x), float(y)))
        return len(nodes) - 1

    cells_by_owner: dict[int, list[int]] = {}
    for face_index in range(base_patch.start_face, base_patch.start_face + base_patch.n_faces):
        cells_by_owner[int(owner[face_index])] = [node_index(point_index) for point_index in faces[face_index]]

    cells = [cells_by_owner[cell_index] for cell_index in sorted(cells_by_owner)]
    if len(cells) != int(owner.max()) + 1:
        raise ValueError("Could not recover one 2D polygon for every OpenFOAM cell")

    nodes_per_cell = {len(cell) for cell in cells}
    if len(nodes_per_cell) != 1:
        raise ValueError(f"Mixed OpenFOAM face sizes are not supported in this demo: {sorted(nodes_per_cell)}")

    point_to_node = np.full(openfoam_points.shape[0], -1, dtype=np.int64)
    for point_index, point in enumerate(openfoam_points):
        point_to_node[point_index] = node_by_xy[_xy_key(point[0], point[1])]

    boundary_nodes = np.zeros(len(nodes), dtype=bool)
    for patch in patches.values():
        if patch.type == "empty":
            continue
        for face_index in range(patch.start_face, patch.start_face + patch.n_faces):
            for point_index in faces[face_index]:
                boundary_nodes[point_to_node[point_index]] = True

    mesh = MeshState(
        points=torch.as_tensor(np.asarray(nodes), dtype=torch.float64),
        cell_blocks=(torch.as_tensor(np.asarray(cells), dtype=torch.long),),
        boundary_nodes=torch.as_tensor(boundary_nodes),
    )
    return FoamMesh2D(
        mesh=mesh,
        openfoam_points=openfoam_points,
        point_to_node=point_to_node,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        patches=patches,
    )


def _velocity_gradient_monitor(
    foam_mesh: FoamMesh2D,
    velocity: np.ndarray,
    *,
    alpha: float,
    smoothing_steps: int,
) -> np.ndarray:
    centroids = cell_centroids(foam_mesh.mesh.points, foam_mesh.mesh.cell_blocks).detach().cpu().numpy()
    raw = np.zeros(foam_mesh.mesh.num_cells, dtype=np.float64)
    counts = np.zeros(foam_mesh.mesh.num_cells, dtype=np.float64)

    for face_index, right_cell in enumerate(foam_mesh.neighbour):
        left_cell = foam_mesh.owner[face_index]
        distance = np.linalg.norm(centroids[right_cell] - centroids[left_cell])
        jump = np.linalg.norm(velocity[right_cell, :2] - velocity[left_cell, :2])
        value = jump / max(distance, 1e-12)
        raw[left_cell] += value
        raw[right_cell] += value
        counts[left_cell] += 1.0
        counts[right_cell] += 1.0

    cylinder = foam_mesh.patches.get("cylinder")
    if cylinder is not None:
        for face_index in range(cylinder.start_face, cylinder.start_face + cylinder.n_faces):
            cell = foam_mesh.owner[face_index]
            face_xy = foam_mesh.openfoam_points[foam_mesh.faces[face_index], :2].mean(axis=0)
            distance = np.linalg.norm(centroids[cell] - face_xy)
            raw[cell] += np.linalg.norm(velocity[cell, :2]) / max(distance, 1e-12)
            counts[cell] += 1.0

    raw = raw / np.maximum(counts, 1.0)
    raw = _smooth_cell_values(foam_mesh, raw, iterations=max(smoothing_steps, 0))
    scale = max(float(np.quantile(raw, 0.92)), float(raw.mean()), 1e-12)
    normalized = np.clip(raw / scale, 0.0, 2.0)
    return 1.0 + alpha * normalized


def _smooth_cell_values(foam_mesh: FoamMesh2D, values: np.ndarray, *, iterations: int) -> np.ndarray:
    smoothed = values.astype(np.float64, copy=True)
    for _ in range(iterations):
        accumulated = smoothed.copy()
        counts = np.ones_like(smoothed)
        for face_index, right_cell in enumerate(foam_mesh.neighbour):
            left_cell = foam_mesh.owner[face_index]
            accumulated[left_cell] += smoothed[right_cell]
            accumulated[right_cell] += smoothed[left_cell]
            counts[left_cell] += 1.0
            counts[right_cell] += 1.0
        smoothed = accumulated / counts
    return smoothed


def _write_adapter_artifacts(
    output_dir: Path,
    initial_mesh: MeshState,
    adapted_mesh: MeshState,
    velocity: np.ndarray,
    monitor: np.ndarray,
    time: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    displacement = torch.linalg.norm(adapted_mesh.points - initial_mesh.points, dim=1).detach().cpu().numpy()
    initial_areas = cell_abs_areas(initial_mesh.points, initial_mesh.cell_blocks).detach().cpu().numpy()
    adapted_areas = cell_abs_areas(adapted_mesh.points, adapted_mesh.cell_blocks).detach().cpu().numpy()
    speed = np.linalg.norm(velocity[:, :2], axis=1)

    before = _write_mesh_vtu(
        initial_mesh,
        output_dir / "cylinder_adapter_before.vtu",
        cell_data={
            "velocity_magnitude": speed,
            "solution_gradient_monitor": monitor,
            "cell_area": initial_areas,
        },
        point_data={
            "mesh_displacement": np.zeros(initial_mesh.num_points),
            "is_fixed_boundary": initial_mesh.boundary_nodes.detach().cpu().numpy().astype(np.int8),
        },
    )
    after = _write_mesh_vtu(
        adapted_mesh,
        output_dir / "cylinder_adapter_after.vtu",
        cell_data={
            "velocity_magnitude_before_move": speed,
            "solution_gradient_monitor": monitor,
            "cell_area": adapted_areas,
            "cell_area_ratio_to_initial": adapted_areas / np.maximum(initial_areas, 1e-30),
        },
        point_data={
            "mesh_displacement": displacement,
            "is_fixed_boundary": adapted_mesh.boundary_nodes.detach().cpu().numpy().astype(np.int8),
        },
    )
    return _write_pvd(output_dir / "cylinder_adapter_motion.pvd", [(time, before), (time + 1e-9, after)])


def _write_mesh_vtu(
    mesh: MeshState,
    path: Path,
    *,
    cell_data: dict[str, np.ndarray],
    point_data: dict[str, np.ndarray],
) -> Path:
    import meshio

    points = mesh.points.detach().cpu().numpy()
    points = np.column_stack([points, np.zeros(points.shape[0])])
    cells = mesh.cell_blocks[0].detach().cpu().numpy().astype(np.int64, copy=False)
    if cells.shape[1] == 3:
        cell_type = "triangle"
    elif cells.shape[1] == 4:
        cell_type = "quad"
    else:
        cell_type = "polygon"
    meshio.write(
        path,
        meshio.Mesh(
            points=points,
            cells=[meshio.CellBlock(cell_type, cells)],
            cell_data={name: [np.asarray(values)] for name, values in cell_data.items()},
            point_data={name: np.asarray(values) for name, values in point_data.items()},
        ),
    )
    return path


def _write_target_points(path: Path, original_points: np.ndarray, point_to_node: np.ndarray, adapted_points: np.ndarray) -> None:
    moved = original_points.copy()
    moved[:, :2] = adapted_points[point_to_node]
    path.write_text(
        _foam_header("vectorField", "targetPoints", location="constant")
        + f"\n{len(moved)}\n(\n"
        + "".join(f"({point[0]:.12g} {point[1]:.12g} {point[2]:.12g})\n" for point in moved)
        + ");\n",
        encoding="utf-8",
    )


def _read_points(path: Path) -> np.ndarray:
    body = _foam_list_body(path.read_text(encoding="utf-8"))
    points = [tuple(float(value) for value in match.group(1).split()) for match in re.finditer(r"\(([^()]+)\)", body)]
    return np.asarray(points, dtype=np.float64)


def _read_faces(path: Path) -> list[list[int]]:
    body = _foam_list_body(path.read_text(encoding="utf-8"))
    return [[int(value) for value in match.group(2).split()] for match in re.finditer(r"(\d+)\s*\(([^()]*)\)", body)]


def _read_label_list(path: Path) -> np.ndarray:
    body = _foam_list_body(path.read_text(encoding="utf-8"))
    return np.asarray([int(value) for value in body.split()], dtype=np.int64)


def _read_boundary(path: Path) -> dict[str, FoamPatch]:
    text = _strip_foam_comments(path.read_text(encoding="utf-8"))
    body = _foam_list_body(text)
    patches: dict[str, FoamPatch] = {}
    for match in re.finditer(r"([A-Za-z0-9_\"|().-]+)\s*\{([^{}]*)\}", body, re.S):
        name = match.group(1).strip('"')
        patch_body = match.group(2)
        patch_type = _read_dict_value(patch_body, "type")
        n_faces = int(_read_dict_value(patch_body, "nFaces"))
        start_face = int(_read_dict_value(patch_body, "startFace"))
        patches[name] = FoamPatch(name=name, type=patch_type, start_face=start_face, n_faces=n_faces)
    return patches


def _read_vector_field(path: Path, expected_size: int) -> np.ndarray:
    text = _strip_foam_comments(path.read_text(encoding="utf-8"))
    uniform = re.search(r"internalField\s+uniform\s+\(([^()]+)\)\s*;", text)
    if uniform is not None:
        value = np.asarray([float(part) for part in uniform.group(1).split()], dtype=np.float64)
        return np.repeat(value[None, :], expected_size, axis=0)

    nonuniform = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if nonuniform is None:
        raise ValueError(f"Could not parse vector internalField in {path}")
    size = int(nonuniform.group(1))
    values = [tuple(float(value) for value in match.group(1).split()) for match in re.finditer(r"\(([^()]+)\)", nonuniform.group(2))]
    if size != expected_size or len(values) != expected_size:
        raise ValueError(f"Field {path} has {len(values)} values, expected {expected_size}")
    return np.asarray(values, dtype=np.float64)


def _foam_list_body(text: str) -> str:
    text = _strip_foam_comments(text)
    match = re.search(r"(?:^|\n)\s*\d+\s*\(\s*(.*?)\s*\)\s*;?\s*$", text, re.S)
    if match is None:
        raise ValueError("Could not find an OpenFOAM list body")
    return match.group(1)


def _strip_foam_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _read_dict_value(text: str, key: str) -> str:
    match = re.search(rf"\b{re.escape(key)}\s+([^;]+);", text)
    if match is None:
        raise ValueError(f"Missing {key!r} in OpenFOAM dictionary")
    return match.group(1).strip()


def _latest_time(case_dir: Path) -> float:
    times = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            times.append(float(path.name))
        except ValueError:
            pass
    if not times:
        raise ValueError(f"No OpenFOAM time directories in {case_dir}")
    return max(times)


def _time_name(time_value: float) -> str:
    return f"{time_value:.12g}"


def _write_interval_steps(write_interval: float, delta_t: float) -> int:
    if write_interval <= 0.0:
        raise ValueError("write_interval must be positive")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")
    return max(int(round(write_interval / delta_t)), 1)


def _high_gradient_area_ratio(initial_mesh: MeshState, adapted_mesh: MeshState, monitor: np.ndarray) -> float:
    threshold = float(np.quantile(monitor, 0.90))
    mask = monitor >= threshold
    initial_area = cell_abs_areas(initial_mesh.points, initial_mesh.cell_blocks).detach().cpu().numpy()
    adapted_area = cell_abs_areas(adapted_mesh.points, adapted_mesh.cell_blocks).detach().cpu().numpy()
    return float(adapted_area[mask].mean() / initial_area[mask].mean())


def _write_metrics(
    path: Path,
    *,
    timings: dict[str, float],
    adapt_time: float,
    final_time: float,
    initial_loss: float,
    final_loss: float,
    high_gradient_area_ratio: float,
    settings: dict[str, float | int | str],
) -> None:
    payload = {
        "settings": settings,
        "timings_s": timings,
        "adapt_time": adapt_time,
        "final_time": final_time,
        "adapter_initial_loss": initial_loss,
        "adapter_final_loss": final_loss,
        "high_gradient_area_ratio": high_gradient_area_ratio,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pvd(path: Path, datasets: list[tuple[float, Path]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ElementTree.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ElementTree.SubElement(root, "Collection")
    for time_value, dataset_path in datasets:
        ElementTree.SubElement(
            collection,
            "DataSet",
            timestep=f"{time_value:.12g}",
            group="",
            part="0",
            file=dataset_path.relative_to(path.parent).as_posix(),
        )
    tree = ElementTree.ElementTree(root)
    ElementTree.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _xy_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 12), round(float(y), 12))


def _foam_header(class_name: str, object_name: str, *, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location is not None else ""
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
    )


def _foam_float(value: float) -> str:
    return f"{value:.12g}"


def _docker_run(image: str, output_dir: Path, command: str) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{output_dir}:/work",
            image,
            "bash",
            "-lc",
            "source /usr/lib/openfoam/openfoam2512/etc/bashrc && "
            "export FOAM_USER_APPBIN=/work/platforms/$WM_OPTIONS/bin && "
            "mkdir -p \"$FOAM_USER_APPBIN\" && "
            "export PATH=\"$FOAM_USER_APPBIN:$PATH\" && "
            f"{command}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
