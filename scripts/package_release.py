"""Package a GS-GHOST release bundle for HuggingFace upload.

Usage:
    uv run python scripts/package_release.py --checkpoint /path/to/best.pth
"""

from __future__ import annotations

from pathlib import Path

import tyro


def main(
    checkpoint: Path = Path("/mnt/artifacts-datai/checkpoints/project_gs_ghost/best.pth"),
    output_dir: Path = Path("/mnt/artifacts-datai/exports/project_gs_ghost"),
    config: Path = Path("configs/default.toml"),
) -> None:
    """Package release bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Config: {config}")

    if not checkpoint.exists():
        print(f"WARNING: Checkpoint not found at {checkpoint}")
        print("Run training first, then re-run this script.")
        return

    # Copy checkpoint
    import shutil
    shutil.copy2(checkpoint, output_dir / "model.pth")

    # Copy config
    if config.exists():
        shutil.copy2(config, output_dir / "config.toml")

    print(f"Release bundle ready at {output_dir}")
    print("Next: push to HF with:")
    print(f"  huggingface-cli upload ilessio-aiflowlab/project_gs_ghost {output_dir} . --private")


if __name__ == "__main__":
    tyro.cli(main)
