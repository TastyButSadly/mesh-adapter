#!/usr/bin/env python3
"""Docker entrypoint: starts adapter service in-process, then runs sweep."""
import sys
import subprocess
import threading
import time
from pathlib import Path


def main():
    # Find exchange dir from args
    exchange_dir = None
    for i, arg in enumerate(sys.argv):
        if arg == "--adapter-exchange-dir" and i + 1 < len(sys.argv):
            exchange_dir = Path(sys.argv[i + 1])

    # Start adapter service thread
    stop_event = threading.Event()
    if exchange_dir:
        exchange_dir.mkdir(parents=True, exist_ok=True)
        worker = threading.Thread(
            target=_serve_adapter,
            args=(exchange_dir, stop_event),
            daemon=True,
        )
        worker.start()
        print(f"[entrypoint] Adapter service started on {exchange_dir}")

    # Run sweep script as subprocess
    result = subprocess.run(
        [sys.executable, "-m", "scripts.diploma_pareto_sweep"] + sys.argv[1:],
        check=False,
    )

    stop_event.set()
    if exchange_dir:
        worker.join(timeout=10.0)

    sys.exit(result.returncode)


def _serve_adapter(exchange_dir: Path, stop_event: threading.Event) -> None:
    """NPZ-based adapter service thread."""
    try:
        import numpy as np
        from examples.firedrake._diff_adapter_subprocess import adapt_coordinates
    except ImportError as e:
        print(f"[adapter] Cannot import adapter: {e}")
        return

    processed = set()
    while not stop_event.is_set():
        for req in sorted(exchange_dir.glob("request_*.npz")):
            if req in processed:
                continue
            resp = exchange_dir / req.name.replace("request_", "response_", 1)
            if resp.exists():
                processed.add(req)
                continue
            try:
                data = np.load(req)
                cells_key = "cells" if "cells" in data else "triangles"
                cell_monitor = data["cell_monitor"] if "cell_monitor" in data else None
                points, info = adapt_coordinates(
                    points_np=data["points"],
                    cells_np=data[cells_key],
                    monitor_np=data["monitor"] if "monitor" in data else None,
                    cell_monitor_np=cell_monitor,
                    steps=int(data["steps"]),
                    lr=float(data["lr"]),
                    profile=str(data["profile"]) if "profile" in data else "regularized",
                )
                tmp = resp.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez(f, points=points,
                             initial_loss=np.array(info["initial_loss"]),
                             final_loss=np.array(info["final_loss"]),
                             steps_completed=np.array(info["steps_completed"]),
                             early_stopped=np.array(info["early_stopped"]),
                             service_s=np.array(0.0))
                tmp.rename(resp)
            except Exception as e:
                print(f"[adapter] Error processing {req.name}: {e}")
            processed.add(req)
        time.sleep(0.02)


if __name__ == "__main__":
    main()
