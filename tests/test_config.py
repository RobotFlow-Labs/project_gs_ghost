from pathlib import Path

from anima_gs_ghost.assets import core_asset_checks
from anima_gs_ghost.config import GhostSettings


def test_default_sfm_method() -> None:
    assert GhostSettings().pipeline.sfm_method in {"hloc", "vggsfm"}


def test_python_target_matches_project_defaults() -> None:
    settings = GhostSettings()
    assert settings.project.name == "anima-gs-ghost"
    assert settings.project.paper_arxiv == "2603.18912"


def test_core_asset_checks_include_paper_pdf() -> None:
    checks = core_asset_checks(GhostSettings())
    assert any(check.name == "paper_pdf" for check in checks)
    assert all(isinstance(check.path, Path) for check in checks)

