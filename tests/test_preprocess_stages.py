from pathlib import Path

import pytest

from anima_gs_ghost.config import GhostSettings
from anima_gs_ghost.preprocess.object_masks import MaskPrompt, ObjectMaskRequest, ObjectMaskStage
from anima_gs_ghost.preprocess.sfm import SfmRequest, SfmStage


def test_object_mask_stage_builds_sam2_outputs(tmp_path: Path) -> None:
    request = ObjectMaskRequest(
        sequence="demo-seq",
        data_root=tmp_path,
        prompts=(MaskPrompt(x=24, y=32, label=1), MaskPrompt(x=12, y=8, label=0)),
    )

    result = ObjectMaskStage().run(request)

    assert result.method == "sam2"
    assert result.outputs.mask_dir == tmp_path / "demo-seq" / "ghost_build" / "obj_bin"
    assert result.outputs.rgb_dir.exists()
    assert "+24,32" in result.command
    assert "-12,8" in result.command


def test_sfm_stage_uses_configured_default_method(tmp_path: Path) -> None:
    settings = GhostSettings.from_toml("configs/default.toml")

    result = SfmStage(settings=settings).run(
        SfmRequest(sequence="demo-seq", data_root=tmp_path, window_size=42)
    )

    assert result.method == settings.pipeline.sfm_method
    assert result.script_path.name == "vggsfm_video.py"
    assert "window_size=42" in result.command
    assert result.outputs.sfm_dir.exists()


def test_sfm_stage_supports_hloc_command_shape(tmp_path: Path) -> None:
    result = SfmStage().run(
        SfmRequest(sequence="demo-seq", data_root=tmp_path, method="hloc", num_pairs=12, window_size=20)
    )

    assert result.script_path.name == "hloc_colmap_sfm.py"
    assert "--seq_name" in result.command
    assert "12" in result.command
    assert "20" in result.command


def test_sfm_stage_rejects_unknown_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported SfM method"):
        SfmStage().run(SfmRequest(sequence="demo-seq", data_root=tmp_path, method="bad"))  # type: ignore[arg-type]
