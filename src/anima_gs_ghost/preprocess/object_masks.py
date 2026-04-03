"""Pure-Python wrapper around the paper's SAM2 object mask preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Literal

from anima_gs_ghost.config import GhostSettings
from anima_gs_ghost.layout import SequenceLayout

ObjectMaskMethod = Literal["sam2"]
SUPPORTED_OBJECT_MASK_METHODS: tuple[ObjectMaskMethod, ...] = ("sam2",)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MaskPrompt:
    x: int
    y: int
    label: int = 1

    def to_cli_arg(self) -> str:
        prefix = "+" if self.label > 0 else "-"
        return f"{prefix}{self.x},{self.y}"


@dataclass(frozen=True)
class ObjectMaskRequest:
    sequence: str
    data_root: Path
    prompts: tuple[MaskPrompt, ...]
    method: ObjectMaskMethod | None = None


@dataclass(frozen=True)
class ObjectMaskOutputs:
    mask_dir: Path
    rgba_dir: Path
    rgb_dir: Path


@dataclass(frozen=True)
class ObjectMaskResult:
    method: ObjectMaskMethod
    command: tuple[str, ...]
    working_directory: Path
    script_path: Path
    outputs: ObjectMaskOutputs
    executed: bool


class ObjectMaskStage:
    """Wrap the paper's SAM2 preprocessing behind a stable Python API."""

    def __init__(
        self,
        settings: GhostSettings | None = None,
        repo_root: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.settings = settings or GhostSettings()
        self.repo_root = (repo_root or (_project_root() / "repositories" / "GHOST")).resolve()
        self.python_executable = python_executable or sys.executable

    @property
    def script_path(self) -> Path:
        return self.repo_root / "preprocess" / "sam_object.py"

    @property
    def working_directory(self) -> Path:
        return self.script_path.parent

    def run(self, request: ObjectMaskRequest, execute: bool = False) -> ObjectMaskResult:
        method = request.method or self.settings.pipeline.object_mask_method
        if method not in SUPPORTED_OBJECT_MASK_METHODS:
            raise ValueError(
                f"Unsupported object-mask method '{method}'. Expected one of {SUPPORTED_OBJECT_MASK_METHODS}."
            )
        if not request.prompts:
            raise ValueError("ObjectMaskStage requires at least one click prompt.")

        layout = SequenceLayout(root=request.data_root, sequence=request.sequence)
        outputs = ObjectMaskOutputs(
            mask_dir=layout.ghost_build / "obj_bin",
            rgba_dir=layout.ghost_build / "obj_rgba",
            rgb_dir=layout.ghost_build / "obj_rgb",
        )
        for path in (outputs.mask_dir, outputs.rgba_dir, outputs.rgb_dir):
            path.mkdir(parents=True, exist_ok=True)

        command = (
            self.python_executable,
            str(self.script_path),
            request.sequence,
            *(prompt.to_cli_arg() for prompt in request.prompts),
        )

        if execute:
            subprocess.run(command, cwd=self.working_directory, check=True)

        return ObjectMaskResult(
            method=method,
            command=command,
            working_directory=self.working_directory,
            script_path=self.script_path,
            outputs=outputs,
            executed=execute,
        )
