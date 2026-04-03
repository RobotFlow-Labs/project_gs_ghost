"""Pure-Python wrappers for the paper's SfM entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Literal

from anima_gs_ghost.config import GhostSettings
from anima_gs_ghost.layout import SequenceLayout

SfmMethod = Literal["hloc", "vggsfm"]
SUPPORTED_SFM_METHODS: tuple[SfmMethod, ...] = ("hloc", "vggsfm")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SfmRequest:
    sequence: str
    data_root: Path
    method: SfmMethod | None = None
    num_pairs: int = 50
    window_size: int = 100


@dataclass(frozen=True)
class SfmOutputs:
    image_dir: Path
    sfm_dir: Path
    pairs_path: Path
    features_path: Path


@dataclass(frozen=True)
class SfmResult:
    method: SfmMethod
    command: tuple[str, ...]
    working_directory: Path
    script_path: Path
    outputs: SfmOutputs
    executed: bool


class SfmStage:
    """Centralized interface for HLoc and VGGSfM preprocessing entrypoints."""

    def __init__(
        self,
        settings: GhostSettings | None = None,
        repo_root: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.settings = settings or GhostSettings()
        self.repo_root = (repo_root or (_project_root() / "repositories" / "GHOST")).resolve()
        self.python_executable = python_executable or sys.executable

    def run(self, request: SfmRequest, execute: bool = False) -> SfmResult:
        method = request.method or self.settings.pipeline.sfm_method
        if method not in SUPPORTED_SFM_METHODS:
            raise ValueError(
                f"Unsupported SfM method '{method}'. Expected one of {SUPPORTED_SFM_METHODS}."
            )

        layout = SequenceLayout(root=request.data_root, sequence=request.sequence)
        outputs = SfmOutputs(
            image_dir=layout.ghost_build / "obj_rgb",
            sfm_dir=layout.ghost_build / "sfm",
            pairs_path=layout.ghost_build / "sfm" / "pairs-netvlad.txt",
            features_path=layout.ghost_build / "sfm" / "features.h5",
        )
        outputs.sfm_dir.mkdir(parents=True, exist_ok=True)

        script_path, command = self._build_command(request, method)
        working_directory = script_path.parent

        if execute:
            subprocess.run(command, cwd=working_directory, check=True)

        return SfmResult(
            method=method,
            command=command,
            working_directory=working_directory,
            script_path=script_path,
            outputs=outputs,
            executed=execute,
        )

    def _build_command(self, request: SfmRequest, method: SfmMethod) -> tuple[Path, tuple[str, ...]]:
        if method == "hloc":
            script_path = self.repo_root / "preprocess" / "hloc_colmap_sfm.py"
            command = (
                self.python_executable,
                str(script_path),
                "--seq_name",
                request.sequence,
                "--num_pairs",
                str(request.num_pairs),
                "--window_size",
                str(request.window_size),
            )
            return script_path, command

        script_path = self.repo_root / "preprocess" / "vggsfm_video.py"
        command = (
            self.python_executable,
            str(script_path),
            f"SCENE_DIR=../data/{request.sequence}",
            f"init_window_size={request.window_size}",
            f"window_size={request.window_size}",
            "camera_type=SIMPLE_PINHOLE",
            "query_method=sp+sift",
        )
        return script_path, command
