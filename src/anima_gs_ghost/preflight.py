"""Production preflight checks — verify paper ID, weights, datasets, deps."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .config import GhostSettings


@dataclass
class PreflightResult:
    name: str
    passed: bool
    message: str


def check_paper_identity(cfg: GhostSettings) -> PreflightResult:
    """Verify the correct paper PDF is present."""
    pdf_path = Path(cfg.project.paper_pdf)
    if not pdf_path.exists():
        # Try relative to project root
        project_root = Path(__file__).resolve().parents[2]
        pdf_path = project_root / cfg.project.paper_pdf

    if pdf_path.exists():
        return PreflightResult("paper_pdf", True, f"Found: {pdf_path}")
    return PreflightResult("paper_pdf", False, f"Missing: {cfg.project.paper_pdf}")


def check_arxiv_id(cfg: GhostSettings) -> PreflightResult:
    """Verify the arxiv ID is correct."""
    expected = "2603.18912"
    if cfg.project.paper_arxiv == expected:
        return PreflightResult("arxiv_id", True, f"Correct: {expected}")
    return PreflightResult(
        "arxiv_id", False,
        f"Wrong arXiv ID: {cfg.project.paper_arxiv} (expected {expected})"
    )


def check_cuda_available() -> PreflightResult:
    """Verify CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return PreflightResult("cuda", True, f"Available: {name}")
        return PreflightResult("cuda", False, "CUDA not available")
    except ImportError:
        return PreflightResult("cuda", False, "torch not installed")


def check_shared_rasterizer() -> PreflightResult:
    """Verify the shared Gaussian rasterizer is importable."""
    rast_path = Path("/mnt/forge-data/shared_infra/cuda_extensions/gaussian_semantic_rasterization")
    if not rast_path.exists():
        return PreflightResult("rasterizer", False, f"Not found: {rast_path}")

    try:
        sys.path.insert(0, str(rast_path.parent))
        from gaussian_semantic_rasterization import GaussianRasterizer  # noqa: F401
        return PreflightResult("rasterizer", True, "Shared rasterizer importable")
    except Exception as e:
        return PreflightResult("rasterizer", False, f"Import error: {e}")


def check_model_weights(cfg: GhostSettings) -> list[PreflightResult]:
    """Check that required model weights exist on disk."""
    results = []
    models = {
        "sam2_large": cfg.data.models.sam2_large,
        "sam2_base": cfg.data.models.sam2_base,
        "dinov2_vitb14": cfg.data.models.dinov2_vitb14,
    }
    for name, path in models.items():
        if path.exists():
            results.append(PreflightResult(f"model_{name}", True, f"Found: {path}"))
        else:
            results.append(PreflightResult(f"model_{name}", False, f"Missing: {path}"))
    return results


def verify_environment(cfg: GhostSettings | None = None) -> list[PreflightResult]:
    """Run all preflight checks.

    Returns:
        List of PreflightResult objects.
    """
    if cfg is None:
        cfg = GhostSettings()

    results = [
        check_paper_identity(cfg),
        check_arxiv_id(cfg),
        check_cuda_available(),
        check_shared_rasterizer(),
    ]
    results.extend(check_model_weights(cfg))
    return results


def print_preflight_report(results: list[PreflightResult]) -> bool:
    """Print a formatted preflight report. Returns True if all passed."""
    print("\n" + "=" * 60)
    print("  GS-GHOST PREFLIGHT CHECK")
    print("=" * 60)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "!"
        print(f"  [{icon}] {status:4s} | {r.name}: {r.message}")
        if not r.passed:
            all_passed = False

    n_pass = sum(1 for r in results if r.passed)
    print(f"\n  {n_pass}/{len(results)} checks passed")
    print("=" * 60 + "\n")
    return all_passed
