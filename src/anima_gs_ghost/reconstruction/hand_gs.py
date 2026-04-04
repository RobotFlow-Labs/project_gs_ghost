"""Canonical hand Gaussian rigging — §3.3.3, Eqs. (10-12).

Gaussian centres are defined on the canonical hand mesh faces and
deformed per-frame via the per-face affine transforms from the hand model.
Each triangle edge is sampled with `gaussians_per_edge` Gaussians.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..hand_model.base import HandModelOutput


def sample_face_barycentrics(n_per_edge: int = 10) -> torch.Tensor:
    """Generate barycentric coordinates for uniformly sampling triangle interiors.

    Uses a grid pattern with n_per_edge points along each edge.

    Returns:
        bary: [M, 3] barycentric coordinates summing to 1.
    """
    coords = []
    for i in range(n_per_edge + 1):
        for j in range(n_per_edge + 1 - i):
            k = n_per_edge - i - j
            coords.append([i / n_per_edge, j / n_per_edge, k / n_per_edge])
    return torch.tensor(coords, dtype=torch.float32)


class HandGaussianRig(nn.Module):
    """Canonical-space hand Gaussians that deform with the hand mesh.

    At init, Gaussians are placed uniformly on canonical mesh faces.
    Per-frame, they are transformed via per-face affine transforms — Eq. (10-12).
    """

    def __init__(
        self,
        n_faces: int,
        gaussians_per_edge: int = 10,
        sh_degree: int = 3,
        device: str = "cuda:1",
    ) -> None:
        super().__init__()
        self.n_faces = n_faces
        self.gaussians_per_edge = gaussians_per_edge
        self.sh_degree = sh_degree
        self.device = device

        # Barycentric sampling pattern
        bary = sample_face_barycentrics(gaussians_per_edge)
        self.register_buffer("barycentrics", bary)  # [M, 3]
        self.n_per_face = bary.shape[0]

        # Learnable per-Gaussian attributes in canonical space
        n_total = n_faces * self.n_per_face
        self._scaling = nn.Parameter(torch.full((n_total, 3), -5.0, device=device))
        self._rotation = nn.Parameter(torch.zeros(n_total, 4, device=device))
        self._rotation.data[:, 0] = 1.0
        self._opacity = nn.Parameter(torch.full((n_total, 1), -2.0, device=device))
        n_sh = (sh_degree + 1) ** 2
        self._features_dc = nn.Parameter(torch.zeros(n_total, 1, 3, device=device))
        self._features_rest = nn.Parameter(torch.zeros(n_total, n_sh - 1, 3, device=device))

    @property
    def n_gaussians(self) -> int:
        return self.n_faces * self.n_per_face

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def canonical_positions(
        self,
        hand_output: HandModelOutput,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """Compute canonical Gaussian positions from mesh face vertices.

        Args:
            hand_output: Output from hand model forward pass (canonical pose).
            batch_idx: Batch element to use.

        Returns:
            positions: [N_total, 3] canonical Gaussian centres.
        """
        verts = hand_output.vertices[batch_idx]  # [V, 3]
        faces = hand_output.faces  # [F, 3]
        face_verts = verts[faces]  # [F, 3, 3]

        # Barycentric interpolation: [F, M, 3] = [F, 1, 3, 3] @ [1, M, 3, 1]
        bary = self.barycentrics.to(verts.device)  # [M, 3]
        positions = torch.einsum("fvd,mv->fmd", face_verts, bary)  # [F, M, 3]
        return positions.reshape(-1, 3)

    def deform(
        self,
        canonical_pos: torch.Tensor,
        face_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-face affine deformation — Eq. (10-12).

        Args:
            canonical_pos: [N_total, 3] canonical positions.
            face_transforms: [F, 3, 4] per-face affine transforms.

        Returns:
            deformed: [N_total, 3] deformed positions.
        """
        F_n = face_transforms.shape[0]
        M = self.n_per_face

        # Reshape to [F, M, 3]
        pos = canonical_pos.reshape(F_n, M, 3)

        # Extract rotation and translation parts
        R = face_transforms[:, :3, :3]  # [F, 3, 3]
        t = face_transforms[:, :3, 3]  # [F, 3]

        # Apply: deformed = R @ canonical + t
        deformed = torch.bmm(pos, R.transpose(1, 2)) + t.unsqueeze(1)
        return deformed.reshape(-1, 3)

    def forward(
        self,
        hand_output: HandModelOutput,
        frame_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Get deformed Gaussian parameters for a specific frame.

        Args:
            hand_output: Hand model output with vertices and face_transforms.
            frame_idx: Which batch element to use.

        Returns:
            Dict with 'xyz', 'opacity', 'scaling', 'rotation', 'features'.
        """
        canonical = self.canonical_positions(hand_output, batch_idx=frame_idx)

        if hand_output.face_transforms is not None:
            ft = hand_output.face_transforms[frame_idx]  # [F, 3, 4]
            xyz = self.deform(canonical, ft)
        else:
            xyz = canonical

        return {
            "xyz": xyz,
            "opacity": self.opacity,
            "scaling": self.scaling,
            "rotation": self.rotation,
            "features": self.features,
        }
