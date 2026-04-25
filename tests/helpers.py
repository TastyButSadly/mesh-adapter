from __future__ import annotations

from pathlib import Path

from diff_mesh_adapter.adapt import AdaptationResult
from diff_mesh_adapter.mesh import MeshState
from diff_mesh_adapter.visualization import save_adaptation_artifacts


def save_demo_artifacts(initial_mesh: MeshState, result: AdaptationResult, output_dir: str | Path, prefix: str, title: str) -> dict[str, Path]:
    return save_adaptation_artifacts(
        initial_mesh,
        result,
        output_dir,
        prefix=prefix,
        title=title,
        include_gif=True,
        fps=12,
    )


def assert_artifacts_exist(paths: dict[str, Path]) -> None:
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
