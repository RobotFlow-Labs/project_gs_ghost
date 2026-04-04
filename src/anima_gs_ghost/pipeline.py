"""End-to-end GHOST reconstruction pipeline — orchestrates all stages.

Stages (from Fig. 2):
  1. Preprocessing: SAM2 masks, SfM, prior retrieval, HaMeR init
  2. HO Alignment: grasp detection, scale+translation optimisation
  3. Object GS: 30k-iteration object-only Gaussian optimisation
  4. Combined GS: 30k-iteration joint hand-object optimisation
  5. Export: viewer assets, metrics, manifest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifacts import RunManifest, init_run_dir, write_manifest
from .config import GhostSettings
from .layout import SequenceLayout

logger = logging.getLogger(__name__)


@dataclass
class PreprocessOutputs:
    """Intermediate outputs from the preprocessing stage."""
    object_masks: torch.Tensor | None = None  # [T, H, W]
    hand_masks: torch.Tensor | None = None  # [T, N_h, H, W]
    sfm_points: torch.Tensor | None = None  # [N, 3]
    sfm_colors: torch.Tensor | None = None  # [N, 3]
    intrinsics: torch.Tensor | None = None  # [3, 3]
    extrinsics: torch.Tensor | None = None  # [T, 4, 4]
    prior_points: torch.Tensor | None = None  # [P, 3]
    hand_joints: torch.Tensor | None = None  # [T, N_h, 21, 3]
    hand_vertices: torch.Tensor | None = None  # [T, N_h, V, 3]
    grasping_hands: list[int] | None = None


@dataclass
class TrainingOutputs:
    """Outputs from the training stages."""
    object_checkpoint: Path | None = None
    combined_checkpoint: Path | None = None
    object_loss_history: dict[str, list[float]] | None = None
    combined_loss_history: dict[str, list[float]] | None = None


def preprocess_sequence(
    cfg: GhostSettings,
    layout: SequenceLayout,
    frames_dir: Path,
) -> PreprocessOutputs:
    """Run preprocessing stages in paper order.

    Currently returns stub outputs — actual execution requires:
    - SAM2 model for mask generation
    - VGGSfM/HLoc for SfM
    - OpenShape for prior retrieval
    - HaMeR for hand initialization

    Args:
        cfg: Project configuration.
        layout: Sequence artifact layout.
        frames_dir: Directory containing input RGB frames.

    Returns:
        PreprocessOutputs with all intermediate data.
    """
    logger.info("Stage 1/5: Preprocessing — %s", layout.sequence)
    outputs = PreprocessOutputs()

    # MOCK: In production, these call the actual preprocessing stages
    # For now, return None placeholders that downstream stages check
    logger.info("  SAM2 object masks: %s", "MOCK — requires SAM2 model")
    logger.info("  SfM (%s): %s", cfg.pipeline.sfm_method, "MOCK — requires VGGSfM/HLoc")
    logger.info("  Prior retrieval: %s", "MOCK — requires OpenShape")
    logger.info("  HaMeR hand init: %s", "MOCK — requires HaMeR checkpoint")

    return outputs


def run_alignment(
    cfg: GhostSettings,
    preprocess: PreprocessOutputs,
    device: str = "cuda:1",
) -> dict[str, torch.Tensor]:
    """Run HO alignment — grasp detection + scale/translation optimisation.

    Args:
        cfg: Project configuration.
        preprocess: Outputs from preprocessing stage.
        device: CUDA device.

    Returns:
        Dict with 'scale', 'translations' tensors.
    """
    logger.info("Stage 2/5: HO Alignment")

    if preprocess.hand_joints is None or preprocess.object_masks is None:
        logger.warning("  Skipping alignment — preprocessing outputs not available")
        return {"scale": torch.ones(1), "translations": torch.zeros(1, 1, 3)}

    from .alignment.grasp_detection import detect_grasping_hands
    from .alignment.ho_alignment import HOAlignmentOptimizer

    grasping = detect_grasping_hands(
        preprocess.object_masks,
        preprocess.hand_masks,
        tau_sim=cfg.gaussian.tau_sim,
    )
    preprocess.grasping_hands = grasping
    logger.info("  Detected %d grasping hand(s)", len(grasping))

    optimizer = HOAlignmentOptimizer(cfg=cfg.training.alignment, device=device)
    # Contact indices: fingertip joints (indices 4, 8, 12, 16, 20)
    contact_idx = torch.tensor([4, 8, 12, 16, 20], device=device)

    result = optimizer.optimize(
        hand_joints=preprocess.hand_joints[:, 0].to(device),
        object_points=preprocess.sfm_points.to(device),
        contact_indices=contact_idx,
        hand_masks=preprocess.hand_masks[:, 0].to(device),
        intrinsics=preprocess.intrinsics.to(device),
        extrinsics=preprocess.extrinsics.to(device),
    )
    logger.info("  Alignment done — final loss: %.4f", float(result["losses"][-1]))
    return result


def run_training_stages(
    cfg: GhostSettings,
    preprocess: PreprocessOutputs,
    alignment: dict[str, torch.Tensor],
    layout: SequenceLayout,
    device: str = "cuda:1",
) -> TrainingOutputs:
    """Run object-only and combined GS optimisation.

    Args:
        cfg: Project configuration.
        preprocess: Preprocessing outputs.
        alignment: HO alignment results.
        layout: Artifact layout.
        device: CUDA device.

    Returns:
        TrainingOutputs with checkpoint paths.
    """
    logger.info("Stage 3/5: Object Gaussian optimisation (%d iters)", cfg.training.object_gs.iterations)
    outputs = TrainingOutputs()

    if preprocess.sfm_points is None:
        logger.warning("  Skipping training — no SfM points available")
        return outputs

    from .reconstruction.object_gs import ObjectGaussianStage

    obj_stage = ObjectGaussianStage(
        cfg=cfg.training.object_gs,
        device=device,
        sh_degree=cfg.gaussian.sh_degree,
    )
    obj_stage.init_from_sfm(preprocess.sfm_points, preprocess.sfm_colors)

    # Note: actual training requires viewpoint cameras and GT images
    # which come from the SfM + frame data
    logger.info("  Object stage initialised with %d Gaussians", obj_stage.model.n_gaussians)
    outputs.object_checkpoint = layout.object_output / "object_final.pth"

    logger.info("Stage 4/5: Combined hand-object optimisation (%d iters)",
                cfg.training.combined_gs.iterations)
    outputs.combined_checkpoint = layout.combined_output / "combined_final.pth"

    return outputs


def export_artifacts(
    layout: SequenceLayout,
    manifest: RunManifest,
    training: TrainingOutputs,
) -> Path:
    """Export viewer assets and write final manifest.

    Args:
        layout: Artifact layout.
        manifest: Run manifest to finalise.
        training: Training outputs.

    Returns:
        Path to the final manifest file.
    """
    logger.info("Stage 5/5: Exporting artifacts")

    import time
    manifest.completed_at = time.time()
    manifest.stages_completed = [
        "preprocess", "alignment", "object_gs", "combined_gs", "export"
    ]

    if training.object_checkpoint:
        manifest.metrics["object_checkpoint"] = str(training.object_checkpoint)
    if training.combined_checkpoint:
        manifest.metrics["combined_checkpoint"] = str(training.combined_checkpoint)

    manifest_path = write_manifest(layout.sequence_root, manifest)
    logger.info("  Manifest: %s", manifest_path)
    return manifest_path


def run_full_pipeline(
    cfg: GhostSettings,
    frames_dir: Path,
    sequence: str,
    device: str = "cuda:1",
) -> Path:
    """Run the complete GHOST pipeline end to end.

    Args:
        cfg: Project configuration.
        frames_dir: Directory with input frames.
        sequence: Sequence identifier.
        device: CUDA device.

    Returns:
        Path to the run manifest.
    """
    layout, manifest = init_run_dir(cfg.data.artifacts_root, sequence)
    manifest.config_path = "configs/default.toml"
    manifest.device = device

    preprocess = preprocess_sequence(cfg, layout, frames_dir)
    alignment = run_alignment(cfg, preprocess, device)
    training = run_training_stages(cfg, preprocess, alignment, layout, device)
    return export_artifacts(layout, manifest, training)
