"""Hand-Object alignment losses — Eqs. (3-6) from §3.2.2.

Jointly optimizes object scale and per-frame hand translations so that
hands make contact with the object surface in a physically plausible way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import AlignmentTrainingSettings


def contact_loss(
    hand_joints: torch.Tensor,
    object_points: torch.Tensor,
    contact_indices: torch.Tensor,
) -> torch.Tensor:
    """L_contact — Eq. (3): minimise distance between contact joints and object surface.

    Args:
        hand_joints: [T, N_joints, 3] hand joint positions.
        object_points: [N_obj, 3] object surface points.
        contact_indices: [N_contact] indices of joints involved in grasping.

    Returns:
        Scalar loss.
    """
    contact_joints = hand_joints[:, contact_indices]  # [T, N_contact, 3]
    # Closest point on object surface for each contact joint
    dists = torch.cdist(contact_joints.reshape(-1, 3), object_points)  # [T*N_c, N_obj]
    min_dists = dists.min(dim=-1).values  # [T*N_c]
    return min_dists.mean()


def projection_loss(
    hand_joints: torch.Tensor,
    hand_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """L_proj — Eq. (4): reprojected hand joints should fall inside hand masks.

    Args:
        hand_joints: [T, N_joints, 3] in world frame.
        hand_masks: [T, H, W] binary hand masks.
        intrinsics: [3, 3] camera intrinsic matrix.
        extrinsics: [T, 4, 4] camera extrinsics (world-to-camera).

    Returns:
        Scalar loss.
    """
    T, N_j, _ = hand_joints.shape
    _, H, W = hand_masks.shape

    # Transform to camera frame
    R = extrinsics[:, :3, :3]  # [T, 3, 3]
    t = extrinsics[:, :3, 3:]  # [T, 3, 1]
    cam_pts = torch.bmm(R, hand_joints.transpose(1, 2)) + t  # [T, 3, N_j]

    # Project to pixel coords
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = cam_pts[:, 2:3].clamp(min=1e-4)  # [T, 1, N_j]
    u = (fx * cam_pts[:, 0:1] / z + cx)  # [T, 1, N_j]
    v = (fy * cam_pts[:, 1:2] / z + cy)  # [T, 1, N_j]

    # Normalise to [-1, 1] for grid_sample
    u_norm = 2.0 * u.squeeze(1) / W - 1.0  # [T, N_j]
    v_norm = 2.0 * v.squeeze(1) / H - 1.0  # [T, N_j]
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)  # [T, 1, N_j, 2]

    # Sample mask at projected positions
    mask_float = hand_masks.float().unsqueeze(1)  # [T, 1, H, W]
    sampled = F.grid_sample(mask_float, grid, align_corners=True, padding_mode="zeros")
    sampled = sampled.squeeze(1).squeeze(1)  # [T, N_j]

    # Loss: joints should be inside mask (sampled ≈ 1)
    return (1.0 - sampled).mean()


def temporal_smoothness_loss(
    translations: torch.Tensor,
) -> torch.Tensor:
    """L_temp — Eq. (5): smooth hand translation trajectories.

    Args:
        translations: [T, N_hands, 3] per-frame hand translations.

    Returns:
        Scalar loss.
    """
    if translations.shape[0] < 2:
        return translations.new_tensor(0.0)
    vel = translations[1:] - translations[:-1]  # [T-1, N_h, 3]
    if vel.shape[0] < 2:
        return vel.pow(2).mean()
    accel = vel[1:] - vel[:-1]  # [T-2, N_h, 3]
    return accel.pow(2).mean()


def total_alignment_loss(
    l_contact: torch.Tensor,
    l_proj: torch.Tensor,
    l_temp: torch.Tensor,
    cfg: AlignmentTrainingSettings | None = None,
) -> torch.Tensor:
    """Total HO alignment loss — Eq. (6).

    L_align = λ_contact * L_contact + λ_proj * L_proj + λ_temp * L_temp
    """
    if cfg is None:
        cfg = AlignmentTrainingSettings()
    return (
        cfg.lambda_contact * l_contact
        + cfg.lambda_proj * l_proj
        + cfg.lambda_temp * l_temp
    )


class HOAlignmentOptimizer:
    """Run the 500-step HO alignment optimization from §3.2.2.

    Optimizes object scale and per-frame hand translations to minimise
    the combined alignment loss.
    """

    def __init__(
        self,
        cfg: AlignmentTrainingSettings | None = None,
        device: str = "cuda:1",
    ) -> None:
        self.cfg = cfg or AlignmentTrainingSettings()
        self.device = device

    def optimize(
        self,
        hand_joints: torch.Tensor,
        object_points: torch.Tensor,
        contact_indices: torch.Tensor,
        hand_masks: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        n_hands: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Run alignment optimisation loop.

        Returns:
            Dict with 'scale', 'translations', 'losses' tensors.
        """
        T = hand_joints.shape[0]
        scale = torch.ones(1, device=self.device, requires_grad=True)
        translations = torch.zeros(T, n_hands, 3, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([scale, translations], lr=self.cfg.lr)
        losses = []

        for step in range(self.cfg.iterations):
            optimizer.zero_grad()

            scaled_obj = object_points * scale
            shifted_joints = hand_joints + translations

            lc = contact_loss(shifted_joints, scaled_obj, contact_indices)
            lp = projection_loss(shifted_joints, hand_masks, intrinsics, extrinsics)
            lt = temporal_smoothness_loss(translations)
            loss = total_alignment_loss(lc, lp, lt, self.cfg)

            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        return {
            "scale": scale.detach(),
            "translations": translations.detach(),
            "losses": torch.tensor(losses),
        }
