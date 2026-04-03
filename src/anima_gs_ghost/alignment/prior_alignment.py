"""Prior-to-mask alignment utilities for GS-GHOST."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


@dataclass(frozen=True)
class PriorTransform:
    quaternion: np.ndarray
    translation: np.ndarray
    scale: np.ndarray

    @staticmethod
    def identity() -> "PriorTransform":
        return PriorTransform(
            quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32),
            scale=np.ones(3, dtype=np.float32),
        )


@dataclass(frozen=True)
class PriorAlignmentHypothesis:
    transform: PriorTransform
    projected_masks: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PriorAlignmentCandidate:
    uid: str
    vertices: np.ndarray
    hypotheses: tuple[PriorAlignmentHypothesis, ...]


@dataclass(frozen=True)
class PriorAlignmentResult:
    uid: str
    transform: PriorTransform
    aligned_vertices: np.ndarray
    mean_iou: float
    loss: float
    projected_masks: tuple[np.ndarray, ...]


class PriorAligner:
    """Evaluate prior candidates using the paper's IoU-style silhouette objective."""

    def loss(self, projected_mask: np.ndarray, object_mask: np.ndarray) -> float:
        """Eq. (1) surrogate: minimize 1 - IoU between projected prior and object mask."""
        return 1.0 - self.iou(projected_mask, object_mask)

    def iou(self, projected_mask: np.ndarray, object_mask: np.ndarray) -> float:
        projected = _as_bool_mask(projected_mask)
        target = _as_bool_mask(object_mask)
        intersection = np.logical_and(projected, target).sum()
        union = np.logical_or(projected, target).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)

    def align(
        self,
        candidates: Iterable[PriorAlignmentCandidate],
        object_masks: tuple[np.ndarray, ...],
    ) -> PriorAlignmentResult:
        best: PriorAlignmentResult | None = None
        for candidate in candidates:
            result = self._evaluate_candidate(candidate, object_masks)
            if best is None or result.loss < best.loss:
                best = result

        if best is None:
            raise ValueError("PriorAligner.align requires at least one candidate.")
        return best

    def _evaluate_candidate(
        self,
        candidate: PriorAlignmentCandidate,
        object_masks: tuple[np.ndarray, ...],
    ) -> PriorAlignmentResult:
        if not candidate.hypotheses:
            raise ValueError(f"Candidate '{candidate.uid}' does not contain any alignment hypotheses.")

        best_result: PriorAlignmentResult | None = None
        for hypothesis in candidate.hypotheses:
            if len(hypothesis.projected_masks) != len(object_masks):
                raise ValueError("Projected mask count must match object mask count for alignment.")

            ious = [
                self.iou(projected_mask, object_mask)
                for projected_mask, object_mask in zip(hypothesis.projected_masks, object_masks)
            ]
            mean_iou = float(sum(ious) / len(ious))
            loss = 1.0 - mean_iou
            aligned_vertices = self.apply_transform(candidate.vertices, hypothesis.transform)
            result = PriorAlignmentResult(
                uid=candidate.uid,
                transform=hypothesis.transform,
                aligned_vertices=aligned_vertices,
                mean_iou=mean_iou,
                loss=loss,
                projected_masks=hypothesis.projected_masks,
            )
            if best_result is None or result.loss < best_result.loss:
                best_result = result

        assert best_result is not None
        return best_result

    def apply_transform(self, vertices: np.ndarray, transform: PriorTransform) -> np.ndarray:
        verts = np.asarray(vertices, dtype=np.float32)
        scaled = verts * transform.scale.reshape(1, 3)
        rotation = self.quaternion_to_matrix(transform.quaternion)
        return scaled @ rotation.T + transform.translation.reshape(1, 3)

    def quaternion_to_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float32)
        norm = math.sqrt(float(w * w + x * x + y * y + z * z))
        if norm == 0:
            raise ValueError("Quaternion norm must be non-zero.")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
