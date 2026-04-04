"""Tests for pipeline orchestration, artifacts, and viewer — PRD-03."""

import tempfile
from pathlib import Path



class TestArtifacts:
    def test_init_run_dir(self):
        from anima_gs_ghost.artifacts import init_run_dir
        with tempfile.TemporaryDirectory() as tmp:
            layout, manifest = init_run_dir(Path(tmp), "test_seq", "run_001")
            assert layout.ghost_build.exists()
            assert layout.object_output.exists()
            assert manifest.sequence == "test_seq"
            assert manifest.run_id == "run_001"

    def test_write_and_load_manifest(self):
        from anima_gs_ghost.artifacts import RunManifest, write_manifest, load_manifest
        with tempfile.TemporaryDirectory() as tmp:
            m = RunManifest(sequence="s1", run_id="r1")
            m.stages_completed = ["preprocess"]
            path = write_manifest(Path(tmp), m)
            assert path.exists()
            loaded = load_manifest(path)
            assert loaded.sequence == "s1"
            assert loaded.stages_completed == ["preprocess"]


class TestPipeline:
    def test_preprocess_returns_outputs(self):
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.layout import SequenceLayout
        from anima_gs_ghost.pipeline import preprocess_sequence
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GhostSettings()
            layout = SequenceLayout(root=Path(tmp), sequence="test")
            outputs = preprocess_sequence(cfg, layout, Path(tmp))
            # Currently returns stub outputs
            assert outputs is not None

    def test_run_full_pipeline_smoke(self):
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.pipeline import run_full_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            cfg = GhostSettings()
            cfg.data.artifacts_root = Path(tmp)
            manifest_path = run_full_pipeline(cfg, Path(tmp), "smoke_test", "cpu")
            assert Path(manifest_path).exists()


class TestViewer:
    def test_export_viewer_assets(self):
        import torch
        from anima_gs_ghost.viewer import export_viewer_assets
        with tempfile.TemporaryDirectory() as tmp:
            exported = export_viewer_assets(
                Path(tmp),
                object_gaussians={"xyz": torch.randn(10, 3)},
                camera_trajectory=[{"fov": 60}],
            )
            assert "object_gaussians" in exported
            assert "cameras" in exported
            assert "index" in exported
