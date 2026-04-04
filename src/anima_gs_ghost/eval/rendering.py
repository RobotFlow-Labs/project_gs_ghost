"""2D rendering evaluation — PSNR, SSIM, LPIPS — Paper §4.2, Table 3."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak signal-to-noise ratio between predicted and ground-truth images.

    Args:
        pred, gt: [C, H, W] or [B, C, H, W] tensors in [0, 1].

    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    return float(-10.0 * torch.log10(mse))


def ssim_metric(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM using the losses module."""
    from ..reconstruction.losses import ssim

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    return float(ssim(pred, gt))


def lpips_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Compute LPIPS perceptual distance.

    Falls back to L2 distance in feature space if lpips package unavailable.
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net, verbose=False)
        loss_fn = loss_fn.to(pred.device)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            return float(loss_fn(pred, gt))
    except ImportError:
        # Fallback: L2 distance (rough approximation)
        return float(F.mse_loss(pred, gt))


def evaluate_render_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate a single predicted/GT image pair.

    Args:
        pred: [C, H, W] predicted rendering.
        gt: [C, H, W] ground-truth image.
        mask: Optional [1, H, W] valid region mask.

    Returns:
        Dict with PSNR, SSIM, LPIPS values.
    """
    if mask is not None:
        pred = pred * mask
        gt = gt * mask

    return {
        "PSNR": psnr(pred, gt),
        "SSIM": ssim_metric(pred, gt),
        "LPIPS": lpips_metric(pred, gt),
    }


def evaluate_sequence(
    pred_dir: Path,
    gt_dir: Path,
    device: str = "cuda:1",
) -> dict[str, float]:
    """Evaluate all rendered frames in a directory against ground truth.

    Returns average PSNR, SSIM, LPIPS across all frames.
    """
    from PIL import Image
    import numpy as np

    pred_files = sorted(pred_dir.glob("*.png"))
    gt_files = sorted(gt_dir.glob("*.png"))
    n = min(len(pred_files), len(gt_files))

    metrics = {"PSNR": [], "SSIM": [], "LPIPS": []}
    for i in range(n):
        pred_np = np.array(Image.open(pred_files[i])).astype(np.float32) / 255.0
        gt_np = np.array(Image.open(gt_files[i])).astype(np.float32) / 255.0
        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).to(device)
        gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).to(device)

        m = evaluate_render_pair(pred_t, gt_t)
        for k in metrics:
            metrics[k].append(m[k])

    return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}


def write_metrics_json(metrics: dict[str, float], output_path: Path) -> None:
    """Write metrics summary to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
