from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from examples.firedrake._adapter_tcp_protocol import recv_message, send_message


DEFAULT_MESH = Path("examples/firedrake/um2n_reference/meshes/cylinder_015.msh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UM2N cylinder setup with differentiable fixed-topology adaptation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/um2n_diff_adapter_cylinder"))
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--firedrake-run", default="firedrake-run")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--adapt-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--monitor-kind", choices=("velocity-gradient", "wake-vorticity"), default="velocity-gradient")
    parser.add_argument("--monitor-frame", choices=("reference", "current"), default="reference")
    parser.add_argument(
        "--monitor-build",
        choices=("firedrake-smoothed", "raw-gradient", "graph-gradient", "raw-vorticity", "composite-grad-vort"),
        default="firedrake-smoothed",
    )
    parser.add_argument("--monitor-scale", type=float, default=0.2)
    parser.add_argument("--monitor-graph-smoothing-steps", type=int, default=4)
    parser.add_argument("--monitor-graph-smoothing-weight", type=float, default=0.5)
    parser.add_argument("--adaptation-relaxation", type=float, default=1.0)
    parser.add_argument("--max-grid-speed", type=float, default=5.0)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=8e-4)
    parser.add_argument("--adapter-profile", choices=("regularized", "monitor-only"), default="regularized")
    parser.add_argument("--adapter-preset", choices=("custom", "accurate", "fast", "faster"), default="custom")
    parser.add_argument("--adapter-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--adapter-transport", choices=("npz", "tcp"), default="tcp")
    parser.add_argument("--adapter-tcp-host", default="host.docker.internal")
    parser.add_argument("--adapter-torch-threads", type=int, default=1)
    parser.add_argument("--adapter-poll-interval", type=float, default=0.001)
    args = parser.parse_args()
    adapter_poll_interval = max(args.adapter_poll_interval, 0.001)
    adapter_steps = _adapter_steps_for_preset(args.adapter_preset, args.adapter_steps)

    cwd = Path.cwd().resolve()
    output_dir = args.output_dir
    try:
        command_output_dir = _container_visible_path(output_dir, cwd, "--output-dir")
        command_mesh = _container_visible_path(args.mesh, cwd, "--mesh")
    except ValueError as exc:
        parser.error(str(exc))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    exchange_dir = output_dir / "adapter_exchange"
    exchange_dir.mkdir(parents=True)
    command_exchange_dir = command_output_dir / "adapter_exchange"
    tcp_server = None
    tcp_port = None
    if args.adapter_transport == "tcp":
        tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_server.bind(("127.0.0.1", 0))
        tcp_server.listen(1)
        tcp_server.settimeout(0.1)
        tcp_port = tcp_server.getsockname()[1]

    command = [
        args.firedrake_run,
        "python3",
        "-m",
        "examples.firedrake._um2n_diff_adapter_cylinder_firedrake",
        "--mesh",
        command_mesh.as_posix(),
        "--output-dir",
        command_output_dir.as_posix(),
        "--dt",
        str(args.dt),
        "--steps",
        str(args.steps),
        "--adapt-every",
        str(args.adapt_every),
        "--save-every",
        str(args.save_every),
        "--monitor-kind",
        args.monitor_kind,
        "--monitor-frame",
        args.monitor_frame,
        "--monitor-build",
        args.monitor_build,
        "--monitor-scale",
        str(args.monitor_scale),
        "--monitor-graph-smoothing-steps",
        str(args.monitor_graph_smoothing_steps),
        "--monitor-graph-smoothing-weight",
        str(args.monitor_graph_smoothing_weight),
        "--adaptation-relaxation",
        str(args.adaptation_relaxation),
        "--max-grid-speed",
        str(args.max_grid_speed),
        "--adapter-steps",
        str(adapter_steps),
        "--adapter-lr",
        str(args.adapter_lr),
        "--adapter-profile",
        args.adapter_profile,
        "--adapter-preset",
        args.adapter_preset,
        "--adapter-dtype",
        args.adapter_dtype,
        "--adapter-transport",
        args.adapter_transport,
        "--adapter-exchange-dir",
        command_exchange_dir.as_posix(),
        "--adapter-poll-interval",
        str(adapter_poll_interval),
        "--adapter-torch-threads",
        str(args.adapter_torch_threads),
    ]
    if args.adapter_transport == "tcp":
        command.extend(["--adapter-tcp-host", args.adapter_tcp_host, "--adapter-tcp-port", str(tcp_port)])
    stop_event = threading.Event()
    if args.adapter_transport == "tcp":
        worker = threading.Thread(
            target=_serve_adapter_tcp,
            args=(tcp_server, stop_event, args.adapter_torch_threads),
            daemon=True,
        )
    else:
        worker = threading.Thread(
            target=_serve_adapter_requests,
            args=(exchange_dir, stop_event, adapter_poll_interval, args.adapter_torch_threads),
            daemon=True,
        )
    worker.start()
    process = subprocess.Popen(command)
    try:
        return_code = process.wait()
    finally:
        stop_event.set()
        worker.join(timeout=5.0)
        if tcp_server is not None:
            tcp_server.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"mesh: {args.mesh}")
    print(f"ParaView series: {output_dir / 'um2n_diff_adapter_cylinder.pvd'}")
    print(f"final plot: {output_dir / 'final_vorticity_mesh.png'}")
    print(f"metrics: {output_dir / 'metrics.json'}")


def _serve_adapter_requests(exchange_dir: Path, stop_event: threading.Event, poll_interval: float, adapter_torch_threads: int) -> None:
    adapt_coordinates = None
    topology_cache = None
    load_error = None
    try:
        adapt_coordinates, cache_cls = _load_adapter(adapter_torch_threads)
        topology_cache = cache_cls()
    except Exception as exc:
        load_error = exc
    processed: set[Path] = set()
    while not stop_event.is_set():
        for request_path in sorted(exchange_dir.glob("request_*.npz")):
            if request_path in processed:
                continue
            response_path = exchange_dir / request_path.name.replace("request_", "response_", 1)
            if response_path.exists():
                processed.add(request_path)
                continue
            try:
                if load_error is not None:
                    raise load_error
                service_start = time.perf_counter()
                data = np.load(request_path)
                cells_np = data["cells"] if "cells" in data else data["triangles"]
                points, info = adapt_coordinates(
                    points_np=data["points"],
                    cells_np=cells_np,
                    monitor_np=data["monitor"] if "monitor" in data else None,
                    steps=int(data["steps"]),
                    lr=float(data["lr"]),
                    profile=str(data["profile"]) if "profile" in data else "regularized",
                    dtype=_npz_string(data, "dtype", "float64"),
                    cell_monitor_np=data["cell_monitor"] if "cell_monitor" in data else None,
                    topology_cache=topology_cache,
                )
                info["service_s"] = time.perf_counter() - service_start
                _write_adapter_response(response_path, points, info)
            except Exception as exc:
                _write_error_response(response_path, f"{type(exc).__name__}: {exc}")
            processed.add(request_path)
        time.sleep(poll_interval)


def _serve_adapter_tcp(server: socket.socket, stop_event: threading.Event, adapter_torch_threads: int) -> None:
    adapt_coordinates = None
    topology_cache = None
    load_error = None
    try:
        adapt_coordinates, cache_cls = _load_adapter(adapter_torch_threads)
        topology_cache = cache_cls()
    except Exception as exc:
        load_error = exc
    cached_cells_np = None
    while not stop_event.is_set():
        try:
            connection, _address = server.accept()
        except socket.timeout:
            continue
        except OSError:
            return
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(None)
            while not stop_event.is_set():
                try:
                    message = recv_message(connection)
                except Exception as exc:
                    _send_tcp_error(connection, exc)
                    continue
                if message is None:
                    break
                scalars, arrays = message
                service_start = time.perf_counter()
                try:
                    if load_error is not None:
                        raise load_error
                    if "cells" in arrays:
                        cached_cells_np = arrays["cells"]
                    elif "triangles" in arrays:
                        cached_cells_np = arrays["triangles"]
                    if cached_cells_np is None:
                        raise ValueError("TCP adapter request did not include initial cells/triangles")
                    points, info = adapt_coordinates(
                        points_np=arrays["points"],
                        cells_np=cached_cells_np,
                        monitor_np=arrays.get("monitor"),
                        steps=int(scalars["steps"]),
                        lr=float(scalars["lr"]),
                        profile=str(scalars.get("profile", "regularized")),
                        dtype=str(scalars.get("dtype", "float64")),
                        cell_monitor_np=arrays.get("cell_monitor"),
                        topology_cache=topology_cache,
                    )
                    info["service_s"] = time.perf_counter() - service_start
                    send_message(
                        connection,
                        scalars={
                            "ok": True,
                            "initial_loss": info["initial_loss"],
                            "final_loss": info["final_loss"],
                            "steps_completed": info["steps_completed"],
                            "early_stopped": info["early_stopped"],
                            "service_s": info["service_s"],
                        },
                        arrays={"points": points},
                    )
                except Exception as exc:
                    _send_tcp_error(connection, exc)


def _send_tcp_error(connection: socket.socket, exc: Exception) -> None:
    send_message(connection, scalars={"ok": False, "error": f"{type(exc).__name__}: {exc}"}, arrays={})


def _load_adapter(adapter_torch_threads: int):
    if adapter_torch_threads > 0:
        import torch

        torch.set_num_threads(adapter_torch_threads)
        try:
            torch.set_num_interop_threads(max(1, adapter_torch_threads))
        except RuntimeError:
            pass

    from examples.firedrake._diff_adapter_subprocess import (
        AdapterTopologyCache,
        adapt_coordinates,
    )

    return adapt_coordinates, AdapterTopologyCache


def _adapter_steps_for_preset(preset: str, custom_steps: int) -> int:
    if preset == "custom":
        return custom_steps
    if preset == "accurate":
        return 12
    if preset == "fast":
        return 6
    if preset == "faster":
        return 4
    raise ValueError(f"Unknown adapter preset: {preset}")


def _npz_string(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def _container_visible_path(path: Path, cwd: Path, argument_name: str) -> Path:
    if not path.is_absolute():
        return path
    try:
        return path.resolve().relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} must be relative to the repository or inside {cwd}; "
            "firedrake-run only mounts this workspace into the container."
        ) from exc


def _write_adapter_response(path: Path, points: np.ndarray, info: dict[str, float | int | bool]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            points=points,
            initial_loss=np.array(info["initial_loss"], dtype=np.float64),
            final_loss=np.array(info["final_loss"], dtype=np.float64),
            steps_completed=np.array(info["steps_completed"], dtype=np.int64),
            early_stopped=np.array(info["early_stopped"], dtype=bool),
            service_s=np.array(info.get("service_s", np.nan), dtype=np.float64),
        )
    tmp_path.rename(path)


def _write_error_response(path: Path, message: str) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, error=np.array(message))
    tmp_path.rename(path)


if __name__ == "__main__":
    main()
