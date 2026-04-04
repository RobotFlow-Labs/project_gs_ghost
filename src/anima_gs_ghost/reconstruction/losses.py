"""Reconstruction losses for object and combined GS stages — §3.3.2.

L_rgb:   photometric loss (L1 + 0.2 * D-SSIM)
L_bkg,h: hand-aware background loss — masks hand region during object stage
L_geo:   geometric consistency between Gaussians and prior mesh
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs().mean()


def _ssim_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """2D Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    g = g / g.sum()
    win_2d = g.unsqueeze(1) * g.unsqueeze(0)  # [K, K] outer product
    return win_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


def ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Structural similarity index between pred and gt images.

    Args:
        pred, gt: [B, C, H, W] images.

    Returns:
        Scalar SSIM value.
    """
    C = pred.shape[1]
    win_2d = _ssim_window(window_size).to(pred.device, pred.dtype)  # [1, 1, K, K]
    win_2d = win_2d.expand(C, 1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(pred, win_2d, padding=pad, groups=C)
    mu2 = F.conv2d(gt, win_2d, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, win_2d, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, win_2d, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, win_2d, padding=pad, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def rgb_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """Photometric loss: L1 + λ * (1 - SSIM)."""
    return (1.0 - lambda_dssim) * l1_loss(pred, gt) + lambda_dssim * (1.0 - ssim(pred, gt))


def background_hand_loss(
    rendered_alpha: torch.Tensor,
    hand_mask: torch.Tensor,
    lambda_bkg: float = 0.3,
) -> torch.Tensor:
    """L_bkg,h — §3.3.2: penalise object Gaussians that render into hand regions.

    Args:
        rendered_alpha: [B, 1, H, W] rendered opacity from object Gaussians.
        hand_mask: [B, 1, H, W] binary hand mask.
        lambda_bkg: Loss weight (default 0.3 from training scripts).

    Returns:
        Scalar loss.
    """
    # Object opacity in hand region should be 0
    hand_opacity = rendered_alpha * hand_mask.float()
    return lambda_bkg * hand_opacity.mean()


def geometric_consistency_loss(
    gaussian_centers: torch.Tensor,
    prior_points: torch.Tensor,
    tau_out: float = 0.05,
    tau_fill: float = 0.005,
    lambda_geo: float = 5.0,
) -> torch.Tensor:
    """L_geo — Eq. (8-9): bidirectional consistency between Gaussians and prior.

    Outlier term: Gaussians far from prior surface are penalised.
    Fill term: Prior points not covered by any Gaussian are penalised.

    Args:
        gaussian_centers: [G, 3] Gaussian positions.
        prior_points: [P, 3] prior mesh surface points.
        tau_out: Distance threshold for outlier penalty (m).
        tau_fill: Distance threshold for fill penalty (m).
        lambda_geo: Loss weight (default 5.0 from paper).

    Returns:
        Scalar loss.
    """
    # Outlier: min dist from each Gaussian to prior
    dists_g2p = torch.cdist(gaussian_centers.unsqueeze(0), prior_points.unsqueeze(0)).squeeze(0)
    min_g2p = dists_g2p.min(dim=-1).values  # [G]
    l_out = F.relu(min_g2p - tau_out).mean()

    # Fill: min dist from each prior point to Gaussians
    min_p2g = dists_g2p.min(dim=0).values  # [P]
    l_fill = F.relu(min_p2g - tau_fill).mean()

    return lambda_geo * (l_out + l_fill)


def combined_object_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    hand_mask: torch.Tensor,
    gaussian_centers: torch.Tensor,
    prior_points: torch.Tensor,
    cfg_lambda_bkg: float = 0.3,
    cfg_lambda_geo: float = 5.0,
    cfg_tau_out: float = 0.05,
    cfg_tau_fill: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Full object-stage loss combining L_rgb + L_bkg,h + L_geo."""
    l_rgb = rgb_loss(pred_rgb, gt_rgb)
    l_bkg = background_hand_loss(rendered_alpha, hand_mask, cfg_lambda_bkg)
    l_geo = geometric_consistency_loss(
        gaussian_centers, prior_points, cfg_tau_out, cfg_tau_fill, cfg_lambda_geo
    )
    total = l_rgb + l_bkg + l_geo
    return {"total": total, "rgb": l_rgb, "bkg_hand": l_bkg, "geo": l_geo}
