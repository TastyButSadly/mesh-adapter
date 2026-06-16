"""Compute L2 error between two Firedrake solutions on different meshes.

Runs INSIDE Docker (firedrake-run). Uses Firedrake's native L2 projection:
  1. Load reference (fine) mesh and solution
  2. Load current (coarse/adapted) mesh and solution
  3. Project coarse solution onto fine mesh function space
  4. Compute L2 error via fd.assemble((u_coarse_proj - u_fine)**2 * dx)

This is the exact FEM error, not an interpolation approximation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import firedrake as fd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute L2 error between two Firedrake solutions.")
    parser.add_argument("--coarse-pvd", type=Path, required=True, help="PVD file for coarse/adapted solution")
    parser.add_argument("--fine-pvd", type=Path, required=True, help="PVD file for fine reference solution")
    parser.add_argument("--coarse-mesh", type=Path, required=True, help="Coarse mesh .msh file")
    parser.add_argument("--fine-mesh", type=Path, required=True, help="Fine mesh .msh file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--fields", nargs="+", default=["velocity"], help="Fields to compare")
    args = parser.parse_args()

    print(f"Loading fine mesh: {args.fine_mesh}")
    fine_mesh = fd.Mesh(str(args.fine_mesh))
    fine_V = fd.VectorFunctionSpace(fine_mesh, "CG", 2)
    fine_Q = fd.FunctionSpace(fine_mesh, "CG", 1)

    print(f"Loading coarse mesh: {args.coarse_mesh}")
    coarse_mesh = fd.Mesh(str(args.coarse_mesh))
    coarse_V = fd.VectorFunctionSpace(coarse_mesh, "CG", 2)
    coarse_Q = fd.FunctionSpace(coarse_mesh, "CG", 1)

    fine_series = _load_pvd(args.fine_pvd)
    coarse_series = _load_pvd(args.coarse_pvd)
    common_times = sorted(set(fine_series) & set(coarse_series))
    print(f"Common timesteps: {common_times}")

    results = {}
    for field in args.fields:
        field_results = []
        for t in common_times:
            print(f"  Computing error for {field} at t={t:.3f}...")

            # Load fine solution
            fine_func = fd.Function(fine_V if field == "velocity" else fine_Q, name=field)
            _load_function_from_vtu(fine_func, fine_series[t], field)

            # Load coarse solution
            coarse_func = fd.Function(coarse_V if field == "velocity" else coarse_Q, name=field)
            _load_function_from_vtu(coarse_func, coarse_series[t], field)

            # Project coarse solution onto fine mesh
            fine_V_coarse = fd.VectorFunctionSpace(fine_mesh, "CG", 2) if field == "velocity" else fd.FunctionSpace(fine_mesh, "CG", 1)
            coarse_on_fine = fd.Function(fine_V_coarse, name=f"{field}_projected")
            coarse_on_fine.project(coarse_func)

            # Compute L2 error
            l2_error = float(np.sqrt(fd.assemble((coarse_on_fine - fine_func) ** 2 * fd.dx(domain=fine_mesh))))

            # Compute L2 norm of reference solution
            ref_norm = float(np.sqrt(fd.assemble(fine_func ** 2 * fd.dx(domain=fine_mesh))))

            # Compute L-infinity error (nodal)
            linf_error = float(np.abs(coarse_on_fine.dat.data_ro - fine_func.dat.data_ro).max())

            rel_l2 = l2_error / max(ref_norm, 1e-14)

            field_results.append({
                "time": t,
                "l2_error": l2_error,
                "relative_l2": rel_l2,
                "linf_error": linf_error,
                "ref_norm": ref_norm,
            })
            print(f"    L2={l2_error:.6e}, rel_L2={rel_l2:.6e}, Linf={linf_error:.6e}")

        # Summary across timesteps (exclude t=0)
        valid = [r for r in field_results if r["time"] > 0.0]
        if valid:
            results[f"mean_{field}_relative_l2"] = float(np.mean([r["relative_l2"] for r in valid]))
            results[f"max_{field}_relative_l2"] = float(np.max([r["relative_l2"] for r in valid]))
            results[f"mean_{field}_l2"] = float(np.mean([r["l2_error"] for r in valid]))
            results[f"max_{field}_l2"] = float(np.max([r["l2_error"] for r in valid]))
            results[f"mean_{field}_linf"] = float(np.mean([r["linf_error"] for r in valid]))
            results[f"max_{field}_linf"] = float(np.max([r["linf_error"] for r in valid]))

    results["timesteps"] = common_times
    results["per_timestep"] = field_results if args.fields else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to: {args.output}")
    for k, v in sorted(results.items()):
        if k not in ("timesteps", "per_timestep"):
            print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")


def _load_pvd(pvd_path: Path) -> dict[float, Path]:
    entries = re.findall(r'<DataSet timestep="([^"]+)" file="([^"]+)" />', pvd_path.read_text())
    return {round(float(time), 12): pvd_path.parent / file_name for time, file_name in entries}


def _load_function_from_vtu(func: fd.Function, vtu_path: Path, field_name: str):
    """Load Firedrake function from VTU file by reading nodal values."""
    import struct

    raw = vtu_path.read_bytes()
    header = raw[:raw.index(b"<AppendedData")].decode(errors="ignore")
    underscore = raw.index(b"_", raw.index(b"<AppendedData"))

    piece = re.search(r'<Piece NumberOfPoints="(\d+)"', header)
    num_points = int(piece.group(1))

    def read_array(name, dtype):
        offset = int(re.search(rf'Name="{re.escape(name)}"[^>]*offset="(\d+)"', header).group(1))
        start = underscore + 1 + offset
        nbytes = struct.unpack("<I", raw[start:start + 4])[0]
        return np.frombuffer(raw[start + 4:start + 4 + nbytes], dtype=dtype).copy()

    # Find the field in point data
    pd_match = re.search(r"<PointData[^>]*>(.*?)</PointData>", header, re.S)
    if not pd_match:
        raise ValueError(f"No PointData in {vtu_path}")

    for da in re.findall(r"<DataArray[^>]*Name=\"([^\"]+)\"[^>]*NumberOfComponents=\"([^\"]*)\"[^>]*/>", pd_match.group(1)):
        name, components = da[0], int(da[1]) if da[1] else 1
        if name == field_name:
            dtype = np.float64
            values = read_array(name, dtype)
            if components > 1:
                values = values.reshape(num_points, components)
            # Assign to Firedrake function
            if values.ndim == 2 and values.shape[1] == func.dat.data_ro.shape[1]:
                func.dat.data[:] = values
            elif values.ndim == 1 and func.dat.data_ro.ndim == 1:
                func.dat.data[:] = values
            else:
                # Try direct assignment (might need reshaping)
                func.dat.data[:] = values.reshape(func.dat.data_ro.shape)
            return
    raise ValueError(f"Field '{field_name}' not found in {vtu_path}")


if __name__ == "__main__":
    main()
