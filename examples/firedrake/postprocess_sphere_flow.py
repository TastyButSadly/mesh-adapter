from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np

from examples.firedrake.sphere_flow_mesh import CHANNEL_HALF_WIDTH, CHANNEL_LENGTH, SPHERE_CENTER, SPHERE_RADIUS


def main() -> None:
    parser = argparse.ArgumentParser(description="Render central-slice sphere-flow visualizations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plane-y", type=float, default=0.0)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    pvd_path = args.output_dir / "sphere_flow_diff_adapter.pvd"
    series = _load_series(pvd_path)
    output_path = args.output_dir / "sphere_flow_slice.gif"
    _write_slice_gif(series, args.plane_y, output_path)
    print(output_path)
    print(output_path.with_name(output_path.stem + "_final.png"))


def _load_series(pvd_path: Path) -> dict[float, Path]:
    entries = re.findall(r'<DataSet timestep="([^"]+)" file="([^"]+)" />', pvd_path.read_text())
    return {round(float(time), 12): pvd_path.parent / file_name for time, file_name in entries}


def _write_slice_gif(series: dict[float, Path], plane_y: float, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    times = sorted(series)
    initial_points, initial_tetrahedra, initial_data = _read_vtu(series[times[0]])
    initial_slice = _slice_tetrahedra(initial_points, initial_tetrahedra, initial_data, plane_y)
    frames: list[Image.Image] = []
    for time in times:
        points, tetrahedra, point_data = _read_vtu(series[time])
        current_slice = _slice_tetrahedra(points, tetrahedra, point_data, plane_y)
        speed = np.linalg.norm(current_slice["velocity"], axis=1)
        vorticity = current_slice["vorticity_magnitude"]
        speed_limit = np.nanpercentile(speed, 99)
        vorticity_limit = np.nanpercentile(vorticity, 99)

        fig = plt.figure(figsize=(12, 9), dpi=180)
        grid = fig.add_gridspec(4, 2, width_ratios=[1.0, 0.025], hspace=0.08, wspace=0.05)
        axes = [fig.add_subplot(grid[row, 0]) for row in range(4)]
        color_axes = [fig.add_subplot(grid[row, 1]) for row in range(4)]
        color_axes[0].axis("off")
        color_axes[1].axis("off")
        fig.suptitle(f"t = {time:.3f}, y = {plane_y:.3f}", fontsize=13)
        _plot_slice_mesh(axes[0], initial_slice, "Initial mesh")
        _plot_slice_mesh(axes[1], current_slice, "Adapted mesh")
        _plot_slice_scalar(axes[2], color_axes[2], current_slice, vorticity, "Vorticity", "magma", 0.0, vorticity_limit)
        _plot_slice_scalar(axes[3], color_axes[3], current_slice, speed, "Velocity", "viridis", 0.0, speed_limit)
        fig.subplots_adjust(left=0.17, right=0.92, top=0.94, bottom=0.04)
        fig.canvas.draw()
        frame = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(frame)
        if time == times[-1]:
            fig.savefig(output_path.with_name(output_path.stem + "_final.png"))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=False)


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
    if set(types.tolist()) == {10}:
        tetrahedra = connectivity.reshape(num_cells, 4)
    elif set(types.tolist()) == {71}:
        tetrahedra = connectivity.reshape(num_cells, 10)[:, :4]
    else:
        raise ValueError(f"Unsupported VTK cell types in {path}: {sorted(set(types.tolist()))}")

    point_data: dict[str, np.ndarray] = {}
    for name, vtk_type, components in _point_data_arrays(header):
        dtype = np.float64 if vtk_type == "Float64" else np.float32
        values = read_array(name, dtype)
        if components > 1:
            values = values.reshape(num_points, components)
        point_data[name] = values
    return points, tetrahedra, point_data


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


def _slice_tetrahedra(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    point_data: dict[str, np.ndarray],
    plane_y: float,
) -> dict[str, np.ndarray]:
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    slice_points: list[tuple[float, float]] = []
    slice_velocity: list[np.ndarray] = []
    slice_vorticity: list[float] = []
    slice_triangles: list[tuple[int, int, int]] = []
    velocity = point_data["velocity"]
    vorticity = point_data["vorticity_magnitude"]

    for cell in tetrahedra:
        vertices = points[cell]
        distances = vertices[:, 1] - plane_y
        intersections: list[tuple[np.ndarray, np.ndarray, float]] = []
        for a, b in edge_pairs:
            da = distances[a]
            db = distances[b]
            if (da < 0.0 and db < 0.0) or (da > 0.0 and db > 0.0):
                continue
            denom = da - db
            if abs(denom) < 1.0e-14:
                continue
            weight = da / denom
            if weight < -1.0e-12 or weight > 1.0 + 1.0e-12:
                continue
            weight = float(np.clip(weight, 0.0, 1.0))
            point = vertices[a] + weight * (vertices[b] - vertices[a])
            vel = velocity[cell[a]] + weight * (velocity[cell[b]] - velocity[cell[a]])
            vort = float(vorticity[cell[a]] + weight * (vorticity[cell[b]] - vorticity[cell[a]]))
            intersections.append((point, vel, vort))
        if len(intersections) < 3:
            continue
        unique = _unique_intersections(intersections)
        if len(unique) not in (3, 4):
            continue
        base = len(slice_points)
        for point, vel, vort in unique:
            slice_points.append((float(point[0]), float(point[2])))
            slice_velocity.append(vel)
            slice_vorticity.append(vort)
        slice_triangles.append((base, base + 1, base + 2))
        if len(unique) == 4:
            slice_triangles.append((base, base + 2, base + 3))

    return {
        "points": np.asarray(slice_points, dtype=float),
        "triangles": np.asarray(slice_triangles, dtype=np.int32),
        "velocity": np.asarray(slice_velocity, dtype=float),
        "vorticity_magnitude": np.asarray(slice_vorticity, dtype=float),
    }


def _unique_intersections(intersections: list[tuple[np.ndarray, np.ndarray, float]]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    unique: list[tuple[np.ndarray, np.ndarray, float]] = []
    for point, velocity, vorticity in intersections:
        if any(np.linalg.norm(point - existing[0]) < 1.0e-10 for existing in unique):
            continue
        unique.append((point, velocity, vorticity))
    if len(unique) <= 3:
        return unique
    center = np.mean([point[[0, 2]] for point, _velocity, _vorticity in unique], axis=0)
    return sorted(unique, key=lambda item: np.arctan2(item[0][2] - center[1], item[0][0] - center[0]))


def _plot_slice_mesh(ax, slice_data: dict[str, np.ndarray], title: str) -> None:
    ax.triplot(
        slice_data["points"][:, 0],
        slice_data["points"][:, 1],
        slice_data["triangles"],
        color="black",
        linewidth=0.18,
        alpha=0.75,
    )
    _format_slice_axis(ax, title)


def _plot_slice_scalar(
    ax,
    cax,
    slice_data: dict[str, np.ndarray],
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    sc = ax.tripcolor(
        slice_data["points"][:, 0],
        slice_data["points"][:, 1],
        slice_data["triangles"],
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.triplot(slice_data["points"][:, 0], slice_data["points"][:, 1], slice_data["triangles"], color="black", linewidth=0.08, alpha=0.25)
    _format_slice_axis(ax, title)
    ax.figure.colorbar(sc, cax=cax)


def _format_slice_axis(ax, title: str) -> None:
    import matplotlib.pyplot as plt

    circle = plt.Circle((SPHERE_CENTER[0], SPHERE_CENTER[2]), SPHERE_RADIUS, facecolor="white", edgecolor="black", linewidth=0.8, zorder=10)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, CHANNEL_LENGTH)
    ax.set_ylim(-CHANNEL_HALF_WIDTH, CHANNEL_HALF_WIDTH)
    ax.set_ylabel(title, rotation=0, labelpad=70, ha="right", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    main()
