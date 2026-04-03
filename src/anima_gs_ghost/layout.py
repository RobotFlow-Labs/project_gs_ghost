"""Canonical runtime layout for GS-GHOST artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SequenceLayout:
    root: Path
    sequence: str

    @property
    def sequence_root(self) -> Path:
        return self.root / self.sequence

    @property
    def ghost_build(self) -> Path:
        return self.sequence_root / "ghost_build"

    @property
    def output_root(self) -> Path:
        return self.sequence_root / "output"

    @property
    def object_output(self) -> Path:
        return self.output_root / "object"

    @property
    def combined_output(self) -> Path:
        return self.output_root / "combined"

    @property
    def viewer_output(self) -> Path:
        return self.sequence_root / "viewer"


def sequence_layout(root: Path, sequence: str) -> dict[str, Path]:
    layout = SequenceLayout(root=root, sequence=sequence)
    return {
        "sequence_root": layout.sequence_root,
        "ghost_build": layout.ghost_build,
        "object_output": layout.object_output,
        "combined_output": layout.combined_output,
        "viewer_output": layout.viewer_output,
    }

