from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

CHANNEL_LENGTH = 2.2
CHANNEL_HEIGHT = 0.41

CYLINDERS = {
    "multiple_cylinders": (
        (0.20, 0.20),
        (0.45, 0.28),
        (0.70, 0.14),
        (0.95, 0.28),
        (1.20, 0.14),
    ),
    "v_formation": (
        (0.34, 0.205),
        (0.62, 0.270),
        (0.62, 0.140),
        (0.90, 0.315),
        (0.90, 0.095),
    ),
}
CYLINDER_RADIUS = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark runs against fine fixed reference.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("outputs/multi_cylinder_benchmark"))
    parser.add_argument("--case", choices=tuple(CYLINDERS), required=True)
    parser.add_argument("--reference-mode", choices=("fixed", "adapted"), default="fixed")
    parser.add_argument("--resolutions", nargs="+", default=("coarse", "medium"))
    parser.add_argument("--compare-modes", nargs="+", choices=("fixed", "adapted"), default=("fixed", "adapted"))
    parser.add_argument("--nx", type=int, default=420)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    case_dir = args.benchmark_dir / args.case
    reference = _load_series(case_dir / "fine" / args.reference_mode / "um2n_diff_adapter_cylinder.pvd")
    sample_points = _sample_points(args.case, args.nx, args.ny)
    ref_samples = {time: _sample_frame(frame, sample_points) for time, frame in reference.items()}

    results: dict[str, object] = {
        "case": args.case,
        "reference": f"fine/{args.reference_mode}",
        "sample_points": int(sample_points.shape[0]),
        "comparisons": [],
    }
    for resolution in args.resolutions:
        for mode in args.compare_modes:
            pvd_path = case_dir / resolution / mode / "um2n_diff_adapter_cylinder.pvd"
            if not pvd_path.exists():
                continue
            series = _load_series(pvd_path)
            common_times = sorted(set(reference) & set(series))
            rows = []
            for time in common_times:
                ref = ref_samples[time]
                sample = _sample_frame(series[time], sample_points)
                rows.append(_error_row(time, sample, ref))
            summary = _summary(rows)
            results["comparisons"].append(
                {
                    "resolution": resolution,
                    "mode": mode,
                    "pvd": str(pvd_path),
                    "num_times": len(rows),
                    "summary": summary,
                    "times": rows,
                }
            )
            if args.gif and rows:
                _write_comparison_gif(
                    args.case,
                    sample_points,
                    reference,
                    series,
                    common_times,
                    case_dir / resolution / mode / "comparison_against_fine.gif",
                )

    output_path = case_dir / "error_summary.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    for comparison in results["comparisons"]:
        print(comparison["resolution"], comparison["mode"], comparison["summary"])


def _load_series(pvd_path: Path) -> dict[float, Path]:
    entries = re.findall(r'<DataSet timestep="([^"]+)" file="([^"]+)" />', pvd_path.read_text())
    return {round(float(time), 12): pvd_path.parent / file_name for time, file_name in entries}


def _sample_points(case_name: str, nx: int, ny: int) -> np.ndarray:
    x = np.linspace(0.0, CHANNEL_LENGTH, nx)
    y = np.linspace(0.0, CHANNEL_HEIGHT, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    mask = np.ones(points.shape[0], dtype=bool)
    for cx, cy in CYLINDERS[case_name]:
        mask &= (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 > (CYLINDER_RADIUS * 1.02) ** 2
    return points[mask]


def _sample_frame(vtu_path: Path, sample_points: np.ndarray) -> dict[str, np.ndarray]:
    import matplotlib.tri as tri

    points, triangles, point_data = _read_vtu(vtu_path)
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)
    sampled: dict[str, np.ndarray] = {}
    for name in ("vorticity", "velocity"):
        if name not in point_data:
            continue
        values = point_data[name]
        if values.ndim == 1:
            interp = tri.LinearTriInterpolator(triangulation, values)
            sampled[name] = np.asarray(interp(sample_points[:, 0], sample_points[:, 1]).filled(np.nan))
        else:
            comps = []
            for component in range(min(values.shape[1], 2)):
                interp = tri.LinearTriInterpolator(triangulation, values[:, component])
                comps.append(np.asarray(interp(sample_points[:, 0], sample_points[:, 1]).filled(np.nan)))
            sampled[name] = np.column_stack(comps)
    return sampled


def _read_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    raw = path.read_bytes()
    header = raw[: raw.index(b"<AppendedData")].decode(errors="ignore")
    underscore = raw.index(b"_", raw.index(b"<AppendedData"))
    piece = re.search(r'<Piece NumberOfPoints="(\d+)" NumberOfCells="(\d+)">', header)
    num_points = int(piece.group(1))
    num_cells = int(piece.group(2))

    def read_array(name: str, dtype: np.dtype) -> np.ndarray:
        offset = int(re.search(rf'Name="{re.escape(name)}"[^>]*offset="(\d+)"', header).group(1))
        start = underscore + 1 + offset
        nbytes = struct.unpack("<I", raw[start : start + 4])[0]
        return np.frombuffer(raw[start + 4 : start + 4 + nbytes], dtype=dtype).copy()

    points = read_array("firedrake_default_topology_coordinates", np.float64).reshape(num_points, 3)
    connectivity = read_array("connectivity", np.int32)
    types = read_array("types", np.uint8)
    if set(types.tolist()) == {69}:
        conn6 = connectivity.reshape(num_cells, 6)
        c0, c1, c2, c3, c4, c5 = [conn6[:, i] for i in range(6)]
        triangles = np.empty((num_cells * 4, 3), dtype=np.int32)
        triangles[0::4] = np.stack([c0, c3, c5], axis=1)
        triangles[1::4] = np.stack([c3, c1, c4], axis=1)
        triangles[2::4] = np.stack([c5, c4, c2], axis=1)
        triangles[3::4] = np.stack([c3, c4, c5], axis=1)
    else:
        triangles = connectivity.reshape(num_cells, 3)

    point_data: dict[str, np.ndarray] = {}
    for name, vtk_type, components in _point_data_arrays(header):
        dtype = np.float64 if vtk_type == "Float64" else np.float32
        values = read_array(name, dtype)
        if components > 1:
            values = values.reshape(num_points, components)
        point_data[name] = values
    return points, triangles, point_data


def _point_data_arrays(header: str) -> list[tuple[str, str, int]]:
    match = re.search(r"<PointData[^>]*>(.*?)</PointData>", header, re.S)
    if not match:
        return []
    arrays = []
    for data_array in re.findall(r"<DataArray[^>]*/>", match.group(1)):
        name = re.search(r'Name="([^"]+)"', data_array).group(1)
        vtk_type = re.search(r'type="([^"]+)"', data_array).group(1)
        components_match = re.search(r'NumberOfComponents="([^"]*)"', data_array)
        components = int(components_match.group(1)) if components_match and components_match.group(1) else 1
        arrays.append((name, vtk_type, components))
    return arrays


def _error_row(time: float, sample: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    row: dict[str, float] = {"time": float(time)}
    for name in ("vorticity", "velocity"):
        if name not in sample or name not in reference:
            continue
        current = sample[name]
        ref = reference[name]
        valid = np.isfinite(current).all(axis=-1) & np.isfinite(ref).all(axis=-1) if current.ndim == 2 else np.isfinite(current) & np.isfinite(ref)
        diff = current[valid] - ref[valid]
        ref_values = ref[valid]
        row[f"{name}_l2"] = float(np.sqrt(np.mean(diff**2)))
        row[f"{name}_relative_l2"] = float(np.sqrt(np.mean(diff**2)) / max(np.sqrt(np.mean(ref_values**2)), 1.0e-14))
        row[f"{name}_linf"] = float(np.max(np.abs(diff))) if diff.size else float("nan")
    return row


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    summary: dict[str, float] = {}
    for key in rows[0]:
        if key == "time":
            continue
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        summary[f"mean_{key}"] = float(np.nanmean(values))
        summary[f"max_{key}"] = float(np.nanmax(values))
    return summary


def _write_comparison_gif(
    case_name: str,
    sample_points: np.ndarray,
    reference: dict[float, Path],
    series: dict[float, Path],
    times: list[float],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    frames: list[Image.Image] = []
    initial_points, initial_triangles, _ = _read_vtu(series[times[0]])
    for time in times:
        current_points, current_triangles, current_data = _read_vtu(series[time])
        fig, axes = plt.subplots(4, 1, figsize=(14, 9), dpi=180, constrained_layout=True)
        vorticity_limit = np.nanpercentile(np.abs(current_data["vorticity"]), 99)
        speed = np.linalg.norm(current_data["velocity"][:, :2], axis=1)
        speed_limit = np.nanpercentile(speed, 99)
        fig.suptitle(f"t = {time:.3f}", fontsize=13)
        _plot_mesh(axes[0], initial_points, initial_triangles, "Initial mesh", case_name)
        _plot_mesh(axes[1], current_points, current_triangles, "Adapted mesh", case_name)
        _plot_scalar(
            axes[2],
            current_points,
            current_triangles,
            current_data["vorticity"],
            "Vorticity",
            "coolwarm",
            -vorticity_limit,
            vorticity_limit,
            case_name,
            show_mesh=True,
        )
        _plot_scalar(
            axes[3],
            current_points,
            current_triangles,
            speed,
            "Velocity",
            "viridis",
            0.0,
            speed_limit,
            case_name,
            show_mesh=True,
        )
        fig.canvas.draw()
        frame = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(frame)
        if time == times[-1]:
            fig.savefig(output_path.with_name(output_path.stem + "_final.png"))
        plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=False)


def _plot_mesh(
    ax,
    points: np.ndarray,
    triangles: np.ndarray,
    title: str,
    case_name: str,
) -> None:
    import matplotlib.pyplot as plt

    ax.triplot(points[:, 0], points[:, 1], triangles, color="black", linewidth=0.12, alpha=0.75)
    for cx, cy in CYLINDERS[case_name]:
        circle = plt.Circle((cx, cy), CYLINDER_RADIUS, facecolor="white", edgecolor="black", linewidth=0.8, zorder=10)
        ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, CHANNEL_LENGTH)
    ax.set_ylim(0.0, CHANNEL_HEIGHT)
    ax.set_ylabel(title, rotation=0, labelpad=70, ha="right", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_scalar(
    ax,
    points: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    case_name: str,
    *,
    show_mesh: bool,
) -> None:
    import matplotlib.pyplot as plt

    sc = ax.tripcolor(
        points[:, 0],
        points[:, 1],
        triangles,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if show_mesh:
        ax.triplot(points[:, 0], points[:, 1], triangles, color="black", linewidth=0.12, alpha=0.35)
    for cx, cy in CYLINDERS[case_name]:
        circle = plt.Circle((cx, cy), CYLINDER_RADIUS, facecolor="white", edgecolor="black", linewidth=0.8, zorder=10)
        ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, CHANNEL_LENGTH)
    ax.set_ylim(0.0, CHANNEL_HEIGHT)
    ax.set_ylabel(title, rotation=0, labelpad=70, ha="right", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(sc, ax=ax, shrink=0.82)


if __name__ == "__main__":
    main()
