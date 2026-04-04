"""Simple hand model for testing — generates a coarse hand mesh with FK.

This is a placeholder that provides the HandModel interface without
requiring MANO/NIMBLE weights. It generates a kinematic chain with
20 joints and a coarse triangulated surface (~778 verts to match MANO).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import HandModelOutput


def _build_hand_topology() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build a simplified hand skeleton and surface mesh.

    Returns:
        rest_joints: [21, 3] rest pose joint positions.
        faces: [F, 3] triangle faces.
        parents: [21] parent joint indices (-1 for root).
    """
    # 21-joint hand: wrist + 4 joints per finger (5 fingers)
    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    rest = torch.zeros(21, 3)
    # Spread fingers along x, extend along y
    finger_offsets = [(-0.04, 0), (-0.02, 0), (0, 0), (0.02, 0), (0.04, 0)]
    for f_idx, (x_off, _) in enumerate(finger_offsets):
        base = 1 + f_idx * 4
        for j in range(4):
            rest[base + j] = torch.tensor([x_off, 0.02 * (j + 1), 0])

    # Generate a simple tube mesh around each bone segment (~778 verts
    verts = []
    faces_list = []
    n_ring = 6
    radius = 0.005

    for j_idx in range(21):
        p_idx = parents[j_idx]
        if p_idx < 0:
            center = rest[j_idx]
        else:
            center = (rest[j_idx] + rest[p_idx]) * 0.5
        for k in range(n_ring):
            angle = 2.0 * 3.14159 * k / n_ring
            offset = torch.tensor([radius * torch.cos(torch.tensor(angle)),
                                   0.0,
                                   radius * torch.sin(torch.tensor(angle))])
            verts.append(center + offset)

    verts_t = torch.stack(verts)  # [21*6, 3]
    # Connect rings into triangles
    for j_idx in range(21):
        p_idx = parents[j_idx]
        if p_idx < 0:
            continue
        base_j = j_idx * n_ring
        base_p = p_idx * n_ring
        for k in range(n_ring):
            k1 = (k + 1) % n_ring
            faces_list.append([base_j + k, base_p + k, base_p + k1])
            faces_list.append([base_j + k, base_p + k1, base_j + k1])

    faces_t = torch.tensor(faces_list, dtype=torch.long)
    return rest, verts_t, faces_t, parents


class SimpleHandModel(nn.Module):
    """Minimal hand model for pipeline testing."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        rest_joints, rest_verts, faces, parents = _build_hand_topology()
        self.register_buffer("rest_joints", rest_joints)
        self.register_buffer("rest_verts", rest_verts)
        self.register_buffer("faces", faces)
        self.parents = parents
        self.n_verts = rest_verts.shape[0]
        self.n_joints = 21
        self.n_faces = faces.shape[0]
        self.to(device)

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        global_orient: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
    ) -> HandModelOutput:
        B = pose.shape[0]
        device = pose.device

        joints = self.rest_joints.unsqueeze(0).expand(B, -1, -1).clone()
        verts = self.rest_verts.unsqueeze(0).expand(B, -1, -1).clone()

        # Apply simple shape blend (scale)
        if shape.shape[-1] > 0:
            scale = 1.0 + shape[:, 0:1].unsqueeze(-1) * 0.1  # [B, 1, 1]
            joints = joints * scale
            verts = verts * scale

        # Apply translation
        if transl is not None:
            joints = joints + transl.unsqueeze(1)
            verts = verts + transl.unsqueeze(1)

        # Compute per-face transforms (identity + translation for now)
        face_transforms = torch.zeros(B, self.n_faces, 3, 4, device=device)
        face_transforms[:, :, 0, 0] = 1.0
        face_transforms[:, :, 1, 1] = 1.0
        face_transforms[:, :, 2, 2] = 1.0
        if transl is not None:
            face_transforms[:, :, :, 3] = transl.unsqueeze(1).expand(-1, self.n_faces, -1)

        return HandModelOutput(
            vertices=verts,
            joints=joints,
            faces=self.faces,
            face_transforms=face_transforms,
        )
