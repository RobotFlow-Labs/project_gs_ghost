"""Tests for evaluation modules — PRD-04."""

import torch
import pytest


class TestBenchmarks:
    def test_arctic_sequences_count(self):
        from anima_gs_ghost.eval.benchmarks import ARCTIC_SEQS
        assert len(ARCTIC_SEQS) == 9

    def test_check_target_pass(self):
        from anima_gs_ghost.eval.benchmarks import MetricTarget, check_target
        t = MetricTarget("PSNR", 25.0, 1.0, lower_is_better=False)
        assert check_target(24.5, t) is True
        assert check_target(23.0, t) is False

    def test_check_target_lower_is_better(self):
        from anima_gs_ghost.eval.benchmarks import MetricTarget, check_target
        t = MetricTarget("LPIPS", 0.03, 0.01, lower_is_better=True)
        assert check_target(0.03, t) is True
        assert check_target(0.05, t) is False


class TestRendering:
    def test_psnr_identical(self):
        from anima_gs_ghost.eval.rendering import psnr
        img = torch.rand(3, 32, 32)
        assert psnr(img, img) > 50

    def test_psnr_different(self):
        from anima_gs_ghost.eval.rendering import psnr
        a = torch.zeros(3, 32, 32)
        b = torch.ones(3, 32, 32)
        p = psnr(a, b)
        # MSE=1 → PSNR=0 dB; just check it's finite and less than identical
        assert p < 50
        assert p == pytest.approx(0.0, abs=0.1)

    def test_ssim_identical(self):
        from anima_gs_ghost.eval.rendering import ssim_metric
        img = torch.rand(3, 64, 64)
        s = ssim_metric(img, img)
        assert s > 0.99


class TestReport:
    def test_compare_against_paper(self):
        from anima_gs_ghost.eval.report import compare_against_paper
        from anima_gs_ghost.eval.benchmarks import ARCTIC_RENDER_TARGETS
        actual = {"PSNR": 26.0, "SSIM": 0.89, "LPIPS": 0.015}
        report = compare_against_paper(actual, ARCTIC_RENDER_TARGETS)
        assert report["PSNR"]["passed"] is True
        assert report["SSIM"]["passed"] is True

    def test_generate_report_markdown(self):
        from anima_gs_ghost.eval.report import compare_against_paper, generate_report_markdown
        report = compare_against_paper({"PSNR": 20.0})
        md = generate_report_markdown(report)
        assert "PSNR" in md
        assert "Score:" in md


class TestArcticExport:
    def test_export_sequence(self):
        import tempfile
        from pathlib import Path
        from anima_gs_ghost.eval.arctic_export import export_sequence
        with tempfile.TemporaryDirectory() as tmp:
            path = export_sequence(
                torch.randn(10, 778, 3),
                torch.randn(10, 778, 3),
                torch.randn(10, 100, 3),
                "arctic_s03_box_grab_01_1",
                Path(tmp),
            )
            assert path.exists()
            assert path.suffix == ".pt"

    def test_export_invalid_sequence_raises(self):
        import tempfile
        from pathlib import Path
        from anima_gs_ghost.eval.arctic_export import export_sequence
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Unknown ARCTIC"):
                export_sequence(
                    torch.randn(1, 778, 3),
                    torch.randn(1, 778, 3),
                    torch.randn(1, 100, 3),
                    "invalid_seq",
                    Path(tmp),
                )
