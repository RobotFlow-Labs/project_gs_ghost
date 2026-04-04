"""Core Gaussian model — manages Gaussian parameters and densification.

Wraps the shared semantic rasterizer at
/mnt/forge-data/shared_infra/cuda_extensions/gaussian_semantic_rasterization/
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

# Import shared rasterizer
sys.path.insert(0, "/mnt/forge-data/shared_infra/cuda_extensions")
from gaussian_semantic_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x + 1e-8))


class GaussianModel(nn.Module):
    """Differentiable 3D Gaussian representation for a single scene."""

    def __init__(
        self,
        sh_degree: int = 3,
        device: str = "cuda:1",
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.device = device
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=device))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3, device=device))
        self._features_rest = nn.Parameter(torch.empty(0, (sh_degree + 1) ** 2 - 1, 3, device=device))
        self.max_radii2D = torch.empty(0, device=device)
        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)

    @property
    def n_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

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

    def init_from_points(
        self,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        """Initialise Gaussians from a point cloud.

        Args:
            points: [N, 3] point positions.
            colors: [N, 3] RGB colors in [0, 1] (optional).
            scale_factor: Global scale multiplier for initial Gaussian sizes.
        """
        N = points.shape[0]
        device = self.device

        self._xyz = nn.Parameter(points.to(device).float())

        # Initial scale from average nearest-neighbour distance
        dists = torch.cdist(points[:min(N, 5000)].float(), points[:min(N, 5000)].float())
        dists.fill_diagonal_(float("inf"))
        nn_dist = dists.min(dim=-1).values.clamp(min=1e-7)
        mean_dist = nn_dist.mean().item()
        init_scale = math.log(mean_dist * scale_factor)
        self._scaling = nn.Parameter(torch.full((N, 3), init_scale, device=device))

        self._rotation = nn.Parameter(torch.zeros(N, 4, device=device))
        self._rotation.data[:, 0] = 1.0

        self._opacity = nn.Parameter(inverse_sigmoid(torch.full((N, 1), 0.1, device=device)))

        n_sh = (self.sh_degree + 1) ** 2
        if colors is not None:
            # Convert RGB to SH DC coefficient
            sh_dc = (colors.to(device).float() - 0.5) / 0.2821
            self._features_dc = nn.Parameter(sh_dc.unsqueeze(1))
        else:
            self._features_dc = nn.Parameter(torch.zeros(N, 1, 3, device=device))

        self._features_rest = nn.Parameter(torch.zeros(N, n_sh - 1, 3, device=device))
        self.max_radii2D = torch.zeros(N, device=device)
        self.xyz_gradient_accum = torch.zeros(N, 1, device=device)
        self.denom = torch.zeros(N, 1, device=device)

    def render(
        self,
        viewpoint_camera: dict,
        bg_color: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render from a viewpoint using the shared semantic rasterizer.

        Args:
            viewpoint_camera: Dict with keys 'image_height', 'image_width',
                'FoVx', 'FoVy', 'world_view_transform', 'full_proj_transform',
                'camera_center'.
            bg_color: [3] background color tensor.

        Returns:
            Dict with 'color' [3, H, W], 'depth' [1, H, W], 'alpha' [1, H, W],
            'radii' [N].
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=self.device)

        tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
        tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera["image_height"]),
            image_width=int(viewpoint_camera["image_width"]),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color.to(self.device),
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera["world_view_transform"].to(self.device),
            projmatrix=viewpoint_camera["full_proj_transform"].to(self.device),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera["camera_center"].to(self.device),
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means2D = torch.zeros_like(self._xyz, requires_grad=True, device=self.device)

        color, objects, radii, depth, alpha = rasterizer(
            means3D=self._xyz,
            means2D=means2D,
            shs=self.features,
            opacities=self.opacity,
            scales=self.scaling,
            rotations=self.rotation,
        )

        return {
            "color": color,
            "depth": depth,
            "alpha": alpha,
            "radii": radii,
            "means2D": means2D,
            "objects": objects,
        }

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_screen_size: float = 20.0,
    ) -> None:
        """Adaptive density control — split, clone, and prune Gaussians."""
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        grads = grads.squeeze(-1)

        # TODO: implement clone (small + high grad) and split (large + high grad)
        # Prune low-opacity or too-large Gaussians
        prune_mask = (self.opacity.squeeze(-1) < min_opacity) | (
            self.max_radii2D > max_screen_size
        )

        # For now, just prune (clone/split needs careful parameter duplication)
        if prune_mask.any():
            keep = ~prune_mask
            self._xyz = nn.Parameter(self._xyz[keep])
            self._scaling = nn.Parameter(self._scaling[keep])
            self._rotation = nn.Parameter(self._rotation[keep])
            self._opacity = nn.Parameter(self._opacity[keep])
            self._features_dc = nn.Parameter(self._features_dc[keep])
            self._features_rest = nn.Parameter(self._features_rest[keep])
            self.max_radii2D = self.max_radii2D[keep]
            self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
            self.denom = self.denom[keep]

    def state_dict_custom(self) -> dict[str, torch.Tensor]:
        return {
            "xyz": self._xyz.data,
            "scaling": self._scaling.data,
            "rotation": self._rotation.data,
            "opacity": self._opacity.data,
            "features_dc": self._features_dc.data,
            "features_rest": self._features_rest.data,
        }

    def load_state_dict_custom(self, state: dict[str, torch.Tensor]) -> None:
        self._xyz = nn.Parameter(state["xyz"].to(self.device))
        self._scaling = nn.Parameter(state["scaling"].to(self.device))
        self._rotation = nn.Parameter(state["rotation"].to(self.device))
        self._opacity = nn.Parameter(state["opacity"].to(self.device))
        self._features_dc = nn.Parameter(state["features_dc"].to(self.device))
        self._features_rest = nn.Parameter(state["features_rest"].to(self.device))
        N = self._xyz.shape[0]
        self.max_radii2D = torch.zeros(N, device=self.device)
        self.xyz_gradient_accum = torch.zeros(N, 1, device=self.device)
        self.denom = torch.zeros(N, 1, device=self.device)
