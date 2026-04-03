"""Asset validation helpers for paper, weights, and datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import GhostSettings


@dataclass(frozen=True)
class AssetCheck:
    name: str
    path: Path
    exists: bool


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def core_asset_checks(settings: GhostSettings) -> list[AssetCheck]:
    project_root = Path.cwd()
    checks = [
        AssetCheck(
            name="paper_pdf",
            path=project_root / settings.project.paper_pdf,
            exists=(project_root / settings.project.paper_pdf).exists(),
        ),
        AssetCheck(
            name="datasets_root",
            path=settings.data.datasets_root,
            exists=settings.data.datasets_root.exists(),
        ),
        AssetCheck(
            name="models_root",
            path=settings.data.models_root,
            exists=settings.data.models_root.exists(),
        ),
    ]
    return checks


def missing_core_assets(settings: GhostSettings) -> list[AssetCheck]:
    return [check for check in core_asset_checks(settings) if not check.exists]

