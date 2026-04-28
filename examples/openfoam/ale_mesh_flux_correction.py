from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from diff_mesh_adapter import (
    AdaptationConfig,
    MeshState,
    adapt_cell_area_equalization,
    cell_abs_measures,
    cell_centroids,
)

IMAGE = "opencfd/openfoam-default:2512"
CONTAINER_ROOT = Path("/work")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal OpenFOAM ALE mesh-flux check.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openfoam_ale_mesh_flux"))
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    mesh, target_mesh = _adapted_quad_mesh()
    solver_dir = output_dir / "aleScalarTransportFoam"
    _write_solver(solver_dir)

    _docker_run(args.image, output_dir, "cd /work/aleScalarTransportFoam && wmake")

    constant_initial = torch.full((mesh.num_cells,), 2.5, dtype=mesh.points.dtype)
    corrected_values = _run_case(
        output_dir,
        args.image,
        mesh,
        target_mesh,
        name="constant_with_mesh_flux",
        initial_values=constant_initial,
    )
    corrected = float(np.max(np.abs(corrected_values - 2.5)))
    naive = _naive_constant_state_error(mesh, target_mesh)

    linear_initial = _linear_cell_average(mesh)
    linear_expected = _linear_cell_average(target_mesh).detach().cpu().numpy()
    linear_corrected_values = _run_case(
        output_dir,
        args.image,
        mesh,
        target_mesh,
        name="linear_with_mesh_flux",
        initial_values=linear_initial,
    )
    linear_no_flux_values = linear_initial.detach().cpu().numpy()
    linear_corrected = float(np.max(np.abs(linear_corrected_values - linear_expected)))
    linear_no_flux = float(np.max(np.abs(linear_no_flux_values - linear_expected)))

    print(f"OpenFOAM image: {args.image}")
    print(f"case directory: {output_dir}")
    print(f"max constant-state error with mesh flux: {corrected:.6e}")
    print(f"max constant-state error without mesh flux: {naive:.6e}")
    print(f"max linear-field remap error with mesh flux: {linear_corrected:.6e}")
    print(f"max linear-field remap error without mesh flux: {linear_no_flux:.6e}")

    if corrected > 1e-10:
        raise SystemExit("mesh-flux corrected OpenFOAM run did not preserve the constant field")
    if naive <= 1e-3:
        raise SystemExit("naive volume-ratio diagnostic unexpectedly preserved the constant field")
    if linear_corrected >= linear_no_flux:
        print("warning: mesh-flux corrected OpenFOAM run did not improve the linear-field remap")


def _run_case(
    output_dir: Path,
    image: str,
    mesh: MeshState,
    target_mesh: MeshState,
    *,
    name: str,
    initial_values: torch.Tensor,
) -> np.ndarray:
    case_dir = output_dir / name
    _write_case(case_dir, mesh, target_mesh, initial_values=initial_values)

    _docker_run(image, output_dir, f"cd /work/{name} && blockMesh >/dev/null && aleScalarTransportFoam")
    values = _read_internal_scalar_values(case_dir / "0.1" / "T")
    return np.asarray(values, dtype=np.float64)


def _naive_constant_state_error(mesh: MeshState, target_mesh: MeshState) -> float:
    old_measures = cell_abs_measures(mesh.points, mesh.cell_blocks)
    new_measures = cell_abs_measures(target_mesh.points, target_mesh.cell_blocks)
    naive_values = 2.5 * old_measures / new_measures
    return float((naive_values - 2.5).abs().max())


def _linear_cell_average(mesh: MeshState) -> torch.Tensor:
    centroids = cell_centroids(mesh.points, mesh.cell_blocks)
    return 1.0 + centroids[:, 0] + 0.5 * centroids[:, 1]


def _docker_run(image: str, output_dir: Path, command: str) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
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


def _adapted_quad_mesh() -> tuple[MeshState, MeshState]:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.38, 0.32],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        dtype=torch.long,
    )
    boundary_nodes = torch.tensor([True, True, True, True, False, True, True, True, True])
    mesh = MeshState(points=points, cell_blocks=(cells,), boundary_nodes=boundary_nodes)
    result = adapt_cell_area_equalization(
        mesh,
        AdaptationConfig(steps=160, lr=2e-2, movement_weight=0.0, shape_weight=0.0),
    )
    return mesh, result.mesh


def _write_solver(path: Path) -> None:
    (path / "Make").mkdir(parents=True)
    (path / "Make" / "files").write_text(
        """aleScalarTransportFoam.C

EXE = $(FOAM_USER_APPBIN)/aleScalarTransportFoam
""",
        encoding="utf-8",
    )
    (path / "Make" / "options").write_text(
        """EXE_INC = \\
    -I$(LIB_SRC)/finiteVolume/lnInclude \\
    -I$(LIB_SRC)/meshTools/lnInclude

EXE_LIBS = \\
    -lfiniteVolume \\
    -lmeshTools
""",
        encoding="utf-8",
    )
    (path / "aleScalarTransportFoam.C").write_text(
        r'''#include "fvCFD.H"
#include "pointIOField.H"

int main(int argc, char *argv[])
{
    argList::addNote("Minimal ALE scalar transport check with externally supplied target points.");

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );

    const Switch useMeshFlux
    (
        transportProperties.getOrDefault<Switch>("useMeshFlux", true)
    );

    surfaceScalarField phi
    (
        IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimVolume/dimTime, 0)
    );

    while (runTime.run())
    {
        ++runTime;

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

        mesh.movePoints(targetPoints);

        if (useMeshFlux)
        {
            phi = -mesh.phi();
        }
        else
        {
            phi = dimensionedScalar("zero", dimVolume/dimTime, 0);
        }

        fvScalarMatrix TEqn
        (
            fvm::ddt(T)
          + fvm::div(phi, T)
        );

        TEqn.solve();

        Info<< "T min/max = " << min(T).value() << " " << max(T).value() << nl << endl;
        runTime.write();
    }

    return 0;
}
''',
        encoding="utf-8",
    )


def _write_case(
    path: Path,
    mesh: MeshState,
    target_mesh: MeshState,
    *,
    initial_values: torch.Tensor,
) -> None:
    for subdir in ("0", "constant", "system"):
        (path / subdir).mkdir(parents=True)
    _write_block_mesh_dict(path / "system" / "blockMeshDict", mesh.points)
    _write_target_points(path / "constant" / "targetPoints", target_mesh.points)
    _write_transport_properties(path / "constant" / "transportProperties")
    _write_control_dict(path / "system" / "controlDict")
    _write_fv_schemes(path / "system" / "fvSchemes")
    _write_fv_solution(path / "system" / "fvSolution")
    _write_scalar_field(path / "0" / "T", initial_values)


def _write_block_mesh_dict(path: Path, points_2d: torch.Tensor) -> None:
    vertices = []
    for z in (0.0, 0.05):
        vertices.extend((float(x), float(y), z) for x, y in points_2d.tolist())

    blocks = (
        (0, 1, 4, 3, 9, 10, 13, 12),
        (1, 2, 5, 4, 10, 11, 14, 13),
        (3, 4, 7, 6, 12, 13, 16, 15),
        (4, 5, 8, 7, 13, 14, 17, 16),
    )
    path.write_text(
        _foam_header("dictionary", "blockMeshDict")
        + "\nscale 1;\n\nvertices\n(\n"
        + "".join(f"    ({x:.16g} {y:.16g} {z:.16g})\n" for x, y, z in vertices)
        + ");\n\nblocks\n(\n"
        + "".join("    hex (" + " ".join(str(i) for i in block) + ") (1 1 1) simpleGrading (1 1 1)\n" for block in blocks)
        + """);\n\nedges\n(\n);\n\nboundary\n(\n"""
        + _patch("xmin", ((0, 9, 12, 3), (3, 12, 15, 6)))
        + _patch("xmax", ((2, 5, 14, 11), (5, 8, 17, 14)))
        + _patch("ymin", ((0, 1, 10, 9), (1, 2, 11, 10)))
        + _patch("ymax", ((6, 15, 16, 7), (7, 16, 17, 8)))
        + _patch("front", ((0, 3, 4, 1), (1, 4, 5, 2), (3, 6, 7, 4), (4, 7, 8, 5)))
        + _patch("back", ((9, 10, 13, 12), (10, 11, 14, 13), (12, 13, 16, 15), (13, 14, 17, 16)))
        + ");\n",
        encoding="utf-8",
    )


def _write_target_points(path: Path, points_2d: torch.Tensor) -> None:
    points = []
    for z in (0.0, 0.05):
        points.extend((float(x), float(y), z) for x, y in points_2d.tolist())
    path.write_text(
        _foam_header("vectorField", "targetPoints", location="constant")
        + f"\n{len(points)}\n(\n"
        + "".join(f"({x:.16g} {y:.16g} {z:.16g})\n" for x, y, z in points)
        + ");\n",
        encoding="utf-8",
    )


def _write_transport_properties(path: Path) -> None:
    path.write_text(_foam_header("dictionary", "transportProperties") + "\nuseMeshFlux true;\n", encoding="utf-8")


def _write_control_dict(path: Path) -> None:
    path.write_text(
        _foam_header("dictionary", "controlDict")
        + """
application     aleScalarTransportFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.1;
deltaT          0.1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  12;
runTimeModifiable false;
""",
        encoding="utf-8",
    )


def _write_fv_schemes(path: Path) -> None:
    path.write_text(
        _foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default Gauss linear;
}

divSchemes
{
    default none;
    div(phi,T) Gauss upwind;
}

laplacianSchemes
{
    default none;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default corrected;
}
""",
        encoding="utf-8",
    )


def _write_fv_solution(path: Path) -> None:
    path.write_text(
        _foam_header("dictionary", "fvSolution")
        + """
solvers
{
    T
    {
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-14;
        relTol 0;
    }
}
""",
        encoding="utf-8",
    )


def _write_scalar_field(path: Path, values: torch.Tensor) -> None:
    value_array = values.detach().cpu().numpy().astype(float)
    internal_field = (
        "internalField   nonuniform List<scalar>\n"
        f"{len(value_array)}\n"
        "(\n"
        + "".join(f"{value:.16g}\n" for value in value_array)
        + ");\n"
    )
    path.write_text(
        _foam_header("volScalarField", "T", location="0")
        + f"""
dimensions      [0 0 0 0 0 0 0];
{internal_field}
boundaryField
{{
    xmin  {{ type zeroGradient; }}
    xmax  {{ type zeroGradient; }}
    ymin  {{ type zeroGradient; }}
    ymax  {{ type zeroGradient; }}
    front {{ type zeroGradient; }}
    back  {{ type zeroGradient; }}
}}
""",
        encoding="utf-8",
    )


def _patch(name: str, faces: tuple[tuple[int, int, int, int], ...]) -> str:
    return (
        f"    {name}\n"
        "    {\n"
        "        type patch;\n"
        "        faces\n"
        "        (\n"
        + "".join("            (" + " ".join(str(i) for i in face) + ")\n" for face in faces)
        + "        );\n"
        "    }\n"
    )


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


def _read_internal_scalar_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform is not None:
        return [float(uniform.group(1))]

    nonuniform = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", text, re.S)
    if nonuniform is None:
        raise ValueError(f"Could not parse internalField in {path}")
    return [float(value) for value in nonuniform.group(1).split()]


if __name__ == "__main__":
    main()
