"""Tests for preflight checks — PRD-07."""


class TestPreflight:
    def test_arxiv_id_correct(self):
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.preflight import check_arxiv_id
        cfg = GhostSettings()
        result = check_arxiv_id(cfg)
        assert result.passed is True

    def test_wrong_arxiv_id(self):
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.preflight import check_arxiv_id
        cfg = GhostSettings()
        cfg.project.paper_arxiv = "2503.14397"
        result = check_arxiv_id(cfg)
        assert result.passed is False

    def test_cuda_check_runs(self):
        from anima_gs_ghost.preflight import check_cuda_available
        result = check_cuda_available()
        # Just verify it runs without error
        assert result.name == "cuda"

    def test_verify_environment(self):
        from anima_gs_ghost.preflight import verify_environment
        results = verify_environment()
        assert len(results) > 0
        # At least arxiv_id should pass
        arxiv_result = [r for r in results if r.name == "arxiv_id"][0]
        assert arxiv_result.passed is True

    def test_print_report(self):
        from anima_gs_ghost.preflight import verify_environment, print_preflight_report
        results = verify_environment()
        # Should not raise
        print_preflight_report(results)


class TestReporting:
    def test_build_run_manifest(self):
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.reporting import build_run_manifest
        cfg = GhostSettings()
        report = build_run_manifest(cfg, {"PSNR": 25.0}, "cuda:1", 3600.0)
        assert report.paper_arxiv == "2603.18912"
        assert report.metrics["PSNR"] == 25.0

    def test_write_training_report(self):
        import tempfile
        from pathlib import Path
        from anima_gs_ghost.config import GhostSettings
        from anima_gs_ghost.reporting import build_run_manifest, write_training_report
        cfg = GhostSettings()
        report = build_run_manifest(cfg, {"PSNR": 25.0})
        with tempfile.TemporaryDirectory() as tmp:
            md_path = write_training_report(report, Path(tmp))
            assert md_path.exists()
            content = md_path.read_text()
            assert "PSNR" in content
