"""CLI for evaluating rendered outputs against ground truth.

Usage:
    uv run python scripts/evaluate_rendering.py --pred-dir output/renders --gt-dir data/gt
"""

from __future__ import annotations

from pathlib import Path

import tyro

from anima_gs_ghost.eval.rendering import evaluate_sequence, write_metrics_json


def main(
    pred_dir: Path,
    gt_dir: Path,
    output: Path = Path("reports/rendering_metrics.json"),
    device: str = "cuda:1",
) -> None:
    """Evaluate rendering quality (PSNR, SSIM, LPIPS)."""
    metrics = evaluate_sequence(pred_dir, gt_dir, device)
    write_metrics_json(metrics, output)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    tyro.cli(main)
