"""Train object-stage GS on synthetic data to validate the GPU pipeline.

This creates a synthetic scene (coloured sphere point cloud + random cameras),
runs the object GS optimisation loop on GPU, and validates:
- Shared rasterizer works end-to-end
- Loss decreases
- Checkpointing works
- GPU memory is within target (60-70% of L4 23GB)

Usage:
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/train_synthetic.py
    CUDA_VISIBLE_DEVICES=5 uv run python scripts/train_synthetic.py --iterations 1000 --n-points 50000
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add shared CUDA extensions
sys.path.insert(0, "/mnt/forge-data/shared_infra/cuda_extensions")

from gaussian_semantic_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"  # Maps to CUDA_VISIBLE_DEVICES
    return "cpu"


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x + 1e-8))


def generate_sphere_pointcloud(
    n_points: int = 20000,
    radius: float = 1.0,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a coloured sphere point cloud."""
    # Fibonacci sphere sampling
    golden_ratio = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n_points, dtype=torch.float32)
    theta = 2 * math.pi * indices / golden_ratio
    phi = torch.acos(1 - 2 * (indices + 0.5) / n_points)

    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    points = torch.stack([x, y, z], dim=-1).to(device)

    # Colour by normal direction (RGB = normalised XYZ)
    colors = (points / radius + 1.0) / 2.0
    return points, colors


def generate_cameras(
    n_cameras: int = 50,
    radius: float = 4.0,
    image_h: int = 480,
    image_w: int = 640,
    fov: float = 60.0,
    device: str = "cuda:0",
) -> list[dict]:
    """Generate orbital cameras looking at the origin."""
    cameras = []
    fov_rad = fov * math.pi / 180.0

    for i in range(n_cameras):
        # Orbit around Y axis
        angle = 2 * math.pi * i / n_cameras
        elevation = 0.3 * math.sin(2 * math.pi * i / n_cameras * 3)

        eye = torch.tensor([
            radius * math.cos(angle) * math.cos(elevation),
            radius * math.sin(elevation),
            radius * math.sin(angle) * math.cos(elevation),
        ], device=device)

        target = torch.zeros(3, device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)

        # Look-at view matrix
        forward = F.normalize(target - eye, dim=0)
        right = F.normalize(torch.cross(forward, up), dim=0)
        new_up = torch.cross(right, forward)

        R = torch.stack([right, new_up, -forward], dim=0)  # [3, 3]
        t = -R @ eye  # [3]

        # World-to-camera [4, 4]
        view = torch.eye(4, device=device)
        view[:3, :3] = R
        view[:3, 3] = t

        # Projection matrix
        znear, zfar = 0.1, 100.0
        tanfov = math.tan(fov_rad / 2)
        proj = torch.zeros(4, 4, device=device)
        proj[0, 0] = 1.0 / (tanfov * image_w / image_h)
        proj[1, 1] = 1.0 / tanfov
        proj[2, 2] = -(zfar + znear) / (zfar - znear)
        proj[2, 3] = -2 * zfar * znear / (zfar - znear)
        proj[3, 2] = -1.0

        full_proj = proj @ view

        cameras.append({
            "image_height": image_h,
            "image_width": image_w,
            "FoVx": fov_rad * image_w / image_h,
            "FoVy": fov_rad,
            "world_view_transform": view.T.contiguous(),  # Column-major for rasterizer
            "full_proj_transform": full_proj.T.contiguous(),
            "camera_center": eye,
        })
    return cameras


def render_gt(
    points: torch.Tensor,
    colors: torch.Tensor,
    camera: dict,
    sh_degree: int = 0,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Render a 'ground truth' image by rasterising the point cloud directly."""
    N = points.shape[0]
    bg = torch.zeros(3, device=device)

    # Simple Gaussian params for GT rendering
    scaling = torch.full((N, 3), math.log(0.005), device=device)
    rotation = torch.zeros(N, 4, device=device)
    rotation[:, 0] = 1.0
    opacity = inverse_sigmoid(torch.full((N, 1), 0.99, device=device))

    # SH DC from colours
    sh_dc = ((colors - 0.5) / 0.2821).unsqueeze(1)  # [N, 1, 3]
    n_sh = (sh_degree + 1) ** 2
    sh_rest = torch.zeros(N, n_sh - 1, 3, device=device)
    shs = torch.cat([sh_dc, sh_rest], dim=1)

    tanfovx = math.tan(camera["FoVx"] * 0.5)
    tanfovy = math.tan(camera["FoVy"] * 0.5)

    settings = GaussianRasterizationSettings(
        image_height=camera["image_height"],
        image_width=camera["image_width"],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=camera["world_view_transform"],
        projmatrix=camera["full_proj_transform"],
        sh_degree=sh_degree,
        campos=camera["camera_center"],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2D = torch.zeros_like(points, requires_grad=False)

    color, _, _, _, _ = rasterizer(
        means3D=points,
        means2D=means2D,
        shs=shs,
        opacities=torch.sigmoid(opacity),
        scales=torch.exp(scaling),
        rotations=F.normalize(rotation, dim=-1),
    )
    return color.detach()  # [3, H, W]


def ssim_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    g = g / g.sum()
    win = g.unsqueeze(1) * g.unsqueeze(0)
    return win.unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    C = pred.shape[0]
    win = ssim_window().to(pred.device, pred.dtype).expand(C, 1, -1, -1)
    pad = 5
    p = pred.unsqueeze(0)
    g = gt.unsqueeze(0)
    mu1 = F.conv2d(p, win, padding=pad, groups=C)
    mu2 = F.conv2d(g, win, padding=pad, groups=C)
    s1 = F.conv2d(p * p, win, padding=pad, groups=C) - mu1 * mu1
    s2 = F.conv2d(g * g, win, padding=pad, groups=C) - mu2 * mu2
    s12 = F.conv2d(p * g, win, padding=pad, groups=C) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    )
    return ssim_map.mean()


def main(
    iterations: int = 2000,
    n_points: int = 30000,
    n_cameras: int = 50,
    sh_degree: int = 3,
    image_h: int = 480,
    image_w: int = 640,
    lr: float = 1.6e-4,
    lambda_dssim: float = 0.2,
    densify_until: int = 1000,
    save_every: int = 500,
) -> None:
    device = get_device()
    print(f"[DEVICE] {device} — {torch.cuda.get_device_name() if 'cuda' in device else 'CPU'}")

    # Output dirs
    ckpt_dir = Path("/mnt/artifacts-datai/checkpoints/project_gs_ghost/synthetic")
    log_dir = Path("/mnt/artifacts-datai/logs/project_gs_ghost")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate scene
    print(f"[DATA] Generating {n_points} point sphere + {n_cameras} cameras ({image_h}x{image_w})")
    points, colors = generate_sphere_pointcloud(n_points, device=device)
    cameras = generate_cameras(n_cameras, image_h=image_h, image_w=image_w, device=device)

    # Pre-render GT images
    print("[DATA] Pre-rendering GT images...")
    gt_images = []
    for cam in tqdm(cameras, desc="GT render"):
        gt = render_gt(points, colors, cam, sh_degree=0, device=device)
        gt_images.append(gt)

    # Initialise learnable Gaussians (add noise to sphere points)
    N = n_points
    xyz = torch.nn.Parameter((points + torch.randn_like(points) * 0.05).clone())

    # Initial scale from NN distance
    dists = torch.cdist(points[:5000], points[:5000])
    dists.fill_diagonal_(float("inf"))
    mean_nn = dists.min(dim=-1).values.mean().item()
    init_scale = math.log(mean_nn * 0.5)

    scaling = torch.nn.Parameter(torch.full((N, 3), init_scale, device=device))
    rotation = torch.nn.Parameter(torch.zeros(N, 4, device=device))
    rotation.data[:, 0] = 1.0
    opacity = torch.nn.Parameter(inverse_sigmoid(torch.full((N, 1), 0.1, device=device)))

    n_sh = (sh_degree + 1) ** 2
    sh_dc = ((colors - 0.5) / 0.2821).unsqueeze(1)
    features_dc = torch.nn.Parameter(sh_dc.clone())
    features_rest = torch.nn.Parameter(torch.zeros(N, n_sh - 1, 3, device=device))

    optimizer = torch.optim.Adam([
        {"params": [xyz], "lr": 1.6e-4},
        {"params": [features_dc], "lr": 0.0025},
        {"params": [features_rest], "lr": 0.0025 / 20},
        {"params": [opacity], "lr": 0.05},
        {"params": [scaling], "lr": 0.005},
        {"params": [rotation], "lr": 0.001},
    ])

    bg = torch.zeros(3, device=device)

    # Training loop
    print(f"\n[TRAIN] {iterations} iterations, {N} Gaussians, SH degree {sh_degree}")
    print(f"[TRAIN] Checkpoint dir: {ckpt_dir}")
    t0 = time.time()
    best_psnr = 0.0

    for step in tqdm(range(1, iterations + 1), desc="Training"):
        cam_idx = (step - 1) % n_cameras
        cam = cameras[cam_idx]
        gt = gt_images[cam_idx]

        optimizer.zero_grad()

        tanfovx = math.tan(cam["FoVx"] * 0.5)
        tanfovy = math.tan(cam["FoVy"] * 0.5)

        settings = GaussianRasterizationSettings(
            image_height=cam["image_height"],
            image_width=cam["image_width"],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=cam["world_view_transform"],
            projmatrix=cam["full_proj_transform"],
            sh_degree=sh_degree,
            campos=cam["camera_center"],
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=settings)
        means2D = torch.zeros_like(xyz, requires_grad=True)
        shs = torch.cat([features_dc, features_rest], dim=1)

        color, _, radii, depth, alpha = rasterizer(
            means3D=xyz,
            means2D=means2D,
            shs=shs,
            opacities=torch.sigmoid(opacity),
            scales=torch.exp(scaling),
            rotations=F.normalize(rotation, dim=-1),
        )

        l1 = (color - gt).abs().mean()
        loss = (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(color, gt))
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                mse = F.mse_loss(color, gt)
                psnr_val = -10 * torch.log10(mse).item()
                best_psnr = max(best_psnr, psnr_val)
                mem = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                pct = mem / total * 100
                tqdm.write(
                    f"[Step {step:5d}] loss={loss.item():.5f} "
                    f"PSNR={psnr_val:.2f}dB "
                    f"VRAM={mem:.1f}/{total:.1f}GB ({pct:.0f}%)"
                )

        if step % save_every == 0:
            path = ckpt_dir / f"synthetic_step{step:06d}.pth"
            torch.save({
                "step": step,
                "xyz": xyz.data,
                "scaling": scaling.data,
                "rotation": rotation.data,
                "opacity": opacity.data,
                "features_dc": features_dc.data,
                "features_rest": features_rest.data,
                "n_gaussians": N,
            }, path)
            tqdm.write(f"[CKPT] Saved {path}")

    elapsed = time.time() - t0
    print(f"\n[DONE] {iterations} steps in {elapsed:.1f}s ({iterations/elapsed:.1f} it/s)")
    print(f"[DONE] Best PSNR: {best_psnr:.2f} dB")
    print(f"[DONE] Checkpoints: {ckpt_dir}")

    # Final GPU memory check
    mem = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] VRAM: {mem:.1f}/{total:.1f}GB ({mem/total*100:.0f}%)")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
