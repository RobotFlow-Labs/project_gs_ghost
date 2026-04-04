"""CLI entrypoint for running the GHOST pipeline on a single sequence.

Usage:
    uv run python scripts/run_sequence.py --sequence my_seq --frames-dir /path/to/frames
    uv run python scripts/run_sequence.py --config configs/default.toml --sequence arctic_s03_box
"""

from __future__ import annotations

import logging
from pathlib import Path

import tyro

from anima_gs_ghost.config import GhostSettings
from anima_gs_ghost.pipeline import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main(
    sequence: str,
    frames_dir: Path = Path("data/frames"),
    config: Path = Path("configs/default.toml"),
    device: str = "cuda:1",
) -> None:
    """Run GHOST reconstruction on a single sequence."""
    if config.exists():
        cfg = GhostSettings.from_toml(config)
    else:
        cfg = GhostSettings()

    manifest_path = run_full_pipeline(cfg, frames_dir, sequence, device)
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    tyro.cli(main)
