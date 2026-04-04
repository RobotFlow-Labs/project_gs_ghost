"""Viewer asset export for reconstructed scenes."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def export_viewer_assets(
    run_dir: Path,
    object_gaussians: dict[str, torch.Tensor] | None = None,
    hand_gaussians: dict[str, torch.Tensor] | None = None,
    camera_trajectory: list[dict] | None = None,
) -> dict[str, Path]:
    """Export viewer-compatible assets from reconstruction results.

    Args:
        run_dir: Run artifact directory.
        object_gaussians: Object Gaussian state dict.
        hand_gaussians: Hand Gaussian state dict.
        camera_trajectory: List of camera parameter dicts.

    Returns:
        Dict mapping asset names to file paths.
    """
    viewer_dir = run_dir / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    if object_gaussians is not None:
        path = viewer_dir / "object_gaussians.pt"
        torch.save(object_gaussians, path)
        exported["object_gaussians"] = path

    if hand_gaussians is not None:
        path = viewer_dir / "hand_gaussians.pt"
        torch.save(hand_gaussians, path)
        exported["hand_gaussians"] = path

    if camera_trajectory is not None:
        path = viewer_dir / "cameras.json"
        with open(path, "w") as f:
            json.dump(camera_trajectory, f, indent=2)
        exported["cameras"] = path

    # Write asset index
    index_path = viewer_dir / "assets.json"
    with open(index_path, "w") as f:
        json.dump({k: str(v) for k, v in exported.items()}, f, indent=2)
    exported["index"] = index_path

    return exported
