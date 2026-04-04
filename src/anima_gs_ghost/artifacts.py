"""Artifact management — sequence run directories and manifests."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .layout import SequenceLayout


@dataclass
class RunManifest:
    sequence: str
    run_id: str
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    stages_completed: list[str] = field(default_factory=list)
    config_path: str | None = None
    device: str = "cuda:1"
    metrics: dict[str, float] = field(default_factory=dict)


def init_run_dir(root: Path, sequence: str, run_id: str | None = None) -> tuple[SequenceLayout, RunManifest]:
    """Create artifact directory structure for a reconstruction run.

    Args:
        root: Artifacts root (e.g. /mnt/artifacts-datai).
        sequence: Sequence identifier.
        run_id: Optional explicit run ID; defaults to timestamp.

    Returns:
        Tuple of (layout, manifest).
    """
    if run_id is None:
        run_id = f"{sequence}_{int(time.time())}"

    layout = SequenceLayout(root=root / "runs" / "project_gs_ghost", sequence=run_id)

    # Create all directories
    for d in [
        layout.ghost_build,
        layout.object_output,
        layout.combined_output,
        layout.viewer_output,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(sequence=sequence, run_id=run_id)
    return layout, manifest


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    """Write the run manifest to disk.

    Returns:
        Path to the manifest JSON file.
    """
    path = run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    return path


def load_manifest(path: Path) -> RunManifest:
    """Load a run manifest from disk."""
    with open(path) as f:
        data = json.load(f)
    return RunManifest(**data)
