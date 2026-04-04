"""Hand model protocol — abstract interface for parametric hand models.

The GHOST paper uses MANO (gated). We abstract the interface so NIMBLE,
Handy, or MANO can be swapped in. The key outputs needed are:
  - vertices: [B, V, 3] hand surface vertices
  - joints: [B, J, 3] joint positions
  - faces: [F, 3] triangle indices
  - face_transforms: [B, F, 3, 4] per-face affine transforms (for rigging)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class HandModelOutput:
    vertices: torch.Tensor  # [B, V, 3]
    joints: torch.Tensor  # [B, J, 3]
    faces: torch.Tensor  # [F, 3] (long)
    face_transforms: torch.Tensor | None = None  # [B, F, 3, 4]


class HandModel(Protocol):
    """Protocol for parametric hand models."""

    n_verts: int
    n_joints: int
    n_faces: int

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        global_orient: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
    ) -> HandModelOutput:
        """Run forward kinematics.

        Args:
            pose: [B, N_pose] pose parameters (joint angles).
            shape: [B, N_shape] shape parameters.
            global_orient: [B, 3] global rotation (axis-angle).
            transl: [B, 3] global translation.

        Returns:
            HandModelOutput with vertices, joints, faces, and optionally
            per-face affine transforms for Gaussian rigging.
        """
        ...

    def to(self, device: str | torch.device) -> "HandModel":
        ...
