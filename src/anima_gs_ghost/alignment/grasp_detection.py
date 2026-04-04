"""Motion-based grasp detection — Eq. (2) from §3.2.1.

Detects which hand(s) are grasping the object by computing the cosine
similarity between the 2D centroid motion of the object mask and each
hand mask across the video sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def centroid_motion(
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute per-frame 2D centroid displacement from binary masks.

    Args:
        masks: [T, H, W] binary mask sequence.

    Returns:
        motion: [T-1, 2] centroid displacement vectors (dy, dx).
    """
    T, H, W = masks.shape
    device = masks.device
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    mask_float = masks.float()
    areas = mask_float.sum(dim=(1, 2)).clamp(min=1.0)
    cy = (mask_float * grid_y.unsqueeze(0)).sum(dim=(1, 2)) / areas
    cx = (mask_float * grid_x.unsqueeze(0)).sum(dim=(1, 2)) / areas
    centroids = torch.stack([cy, cx], dim=-1)  # [T, 2]
    return centroids[1:] - centroids[:-1]  # [T-1, 2]


def grasp_score(
    obj_motion: torch.Tensor,
    hand_motion: torch.Tensor,
) -> float:
    """Cosine similarity between object and hand centroid motion — Eq. (2).

    Args:
        obj_motion: [T-1, 2] object centroid displacement vectors.
        hand_motion: [T-1, 2] hand centroid displacement vectors.

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    obj_flat = obj_motion.reshape(-1)
    hand_flat = hand_motion.reshape(-1)
    if obj_flat.norm() < 1e-8 or hand_flat.norm() < 1e-8:
        return 0.0
    return float(F.cosine_similarity(obj_flat.unsqueeze(0), hand_flat.unsqueeze(0)))


def detect_grasping_hands(
    obj_masks: torch.Tensor,
    hand_masks: torch.Tensor,
    tau_sim: float = 0.5,
) -> list[int]:
    """Detect which hands are grasping the object.

    Args:
        obj_masks: [T, H, W] object mask sequence.
        hand_masks: [T, N_hands, H, W] hand mask sequences.
        tau_sim: Similarity threshold from §3.2.1 (default 0.5).

    Returns:
        List of grasping hand indices.
    """
    obj_motion = centroid_motion(obj_masks)
    T, N_hands, H, W = hand_masks.shape
    grasping = []
    for h in range(N_hands):
        h_motion = centroid_motion(hand_masks[:, h])
        score = grasp_score(obj_motion, h_motion)
        if score > tau_sim:
            grasping.append(h)
    return grasping
