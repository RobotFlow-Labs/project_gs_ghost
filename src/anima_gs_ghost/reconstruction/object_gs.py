"""Object-only Gaussian Splatting optimisation — §3.3.2.

Optimises object Gaussians using L_rgb + L_bkg,h + L_geo for 30k iterations.
Uses the shared GRAPHDECO semantic rasterizer.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from ..config import ObjectGSSettings
from .gaussian_model import GaussianModel
from .losses import combined_object_loss


class ObjectGaussianStage:
    """Object-only Gaussian reconstruction stage."""

    def __init__(
        self,
        cfg: ObjectGSSettings | None = None,
        device: str = "cuda:1",
        sh_degree: int = 3,
    ) -> None:
        self.cfg = cfg or ObjectGSSettings()
        self.device = device
        self.model = GaussianModel(sh_degree=sh_degree, device=device)

    def init_from_sfm(
        self,
        points: torch.Tensor,
        colors: torch.Tensor | None = None,
    ) -> None:
        """Initialise Gaussians from SfM point cloud."""
        self.model.init_from_points(points, colors)

    def train(
        self,
        viewpoints: list[dict],
        gt_images: list[torch.Tensor],
        hand_masks: list[torch.Tensor],
        prior_points: torch.Tensor,
        output_dir: Path | None = None,
    ) -> dict[str, list[float]]:
        """Run the 30k-iteration object optimisation.

        Args:
            viewpoints: List of camera dicts for each training view.
            gt_images: List of [3, H, W] ground-truth images.
            hand_masks: List of [1, H, W] hand masks.
            prior_points: [P, 3] aligned prior surface points.
            output_dir: Optional directory for saving checkpoints.

        Returns:
            Dict of loss history lists.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.6e-4)
        n_views = len(viewpoints)
        history: dict[str, list[float]] = {
            "total": [], "rgb": [], "bkg_hand": [], "geo": [],
        }

        for step in tqdm(range(self.cfg.iterations), desc="Object GS"):
            idx = step % n_views
            cam = viewpoints[idx]
            gt = gt_images[idx].to(self.device)
            hmask = hand_masks[idx].to(self.device)

            optimizer.zero_grad()
            out = self.model.render(cam)
            losses = combined_object_loss(
                pred_rgb=out["color"].unsqueeze(0),
                gt_rgb=gt.unsqueeze(0),
                rendered_alpha=out["alpha"].unsqueeze(0),
                hand_mask=hmask.unsqueeze(0),
                gaussian_centers=self.model.xyz,
                prior_points=prior_points.to(self.device),
                cfg_lambda_bkg=self.cfg.lambda_background,
                cfg_lambda_geo=self.cfg.lambda_geo,
                cfg_tau_out=self.cfg.tau_out,
                cfg_tau_fill=self.cfg.tau_fill,
            )
            losses["total"].backward()
            optimizer.step()

            for k in history:
                history[k].append(float(losses[k]))

            # Densification
            if step < self.cfg.densify_until_iter and step % 100 == 0 and step > 500:
                self.model.densify_and_prune()

            # Checkpoint
            if output_dir and step > 0 and step % 500 == 0:
                self._save_checkpoint(output_dir / f"object_step{step:06d}.pth", step)

        if output_dir:
            self._save_checkpoint(output_dir / "object_final.pth", self.cfg.iterations)

        return history

    def _save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model": self.model.state_dict_custom(),
            "n_gaussians": self.model.n_gaussians,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict_custom(ckpt["model"])
        return ckpt["step"]
