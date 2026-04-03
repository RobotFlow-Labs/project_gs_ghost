"""HaMeR postprocessing helpers for jitter rejection and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class HandFrameStats:
    pose_delta_prev: float = 0.0
    pose_delta_next: float = 0.0
    orient_delta_prev: float = 0.0
    orient_delta_next: float = 0.0
    transl_delta_prev_xy: float = 0.0
    transl_delta_next_xy: float = 0.0
    shape_z_max: float = 0.0


def reject_frame(
    stats: HandFrameStats,
    *,
    pose_thresh: float = 1.0,
    orient_thresh: float = 1.0,
    transl_thresh: float = 2.0,
    shape_thresh: float = 4.0,
) -> bool:
    pose_outlier = (
        (stats.pose_delta_prev > pose_thresh and stats.pose_delta_next > pose_thresh)
        or stats.pose_delta_prev > pose_thresh * 2
        or stats.pose_delta_next > pose_thresh * 2
    )
    orient_outlier = (
        (stats.orient_delta_prev > orient_thresh and stats.orient_delta_next > orient_thresh)
        or stats.orient_delta_prev > orient_thresh * 2
        or stats.orient_delta_next > orient_thresh * 2
    )
    transl_outlier = (
        stats.transl_delta_prev_xy > transl_thresh and stats.transl_delta_next_xy > transl_thresh
    )
    shape_outlier = stats.shape_z_max > shape_thresh
    return pose_outlier or orient_outlier or transl_outlier or shape_outlier


@dataclass(frozen=True)
class HandSequence:
    handedness: Literal["left", "right"]
    frame_ids: np.ndarray
    translations: np.ndarray
    orientations: np.ndarray
    poses: np.ndarray
    shapes: np.ndarray


@dataclass(frozen=True)
class ProcessedHandSequence:
    handedness: Literal["left", "right"]
    frame_ids: np.ndarray
    rejected_frame_ids: tuple[int, ...]
    interpolated_frame_ids: tuple[int, ...]
    translations: np.ndarray
    orientations: np.ndarray
    poses: np.ndarray
    shapes: np.ndarray


class HaMeRPostprocessor:
    """Apply the supplementary jitter rejection and interpolation rules."""

    def __init__(
        self,
        *,
        pose_thresh: float = 1.0,
        orient_thresh: float = 1.0,
        transl_thresh: float = 2.0,
        shape_thresh: float = 4.0,
    ) -> None:
        self.pose_thresh = pose_thresh
        self.orient_thresh = orient_thresh
        self.transl_thresh = transl_thresh
        self.shape_thresh = shape_thresh

    def process(self, sequence: HandSequence) -> ProcessedHandSequence:
        frame_ids = np.asarray(sequence.frame_ids, dtype=int)
        translations = np.asarray(sequence.translations, dtype=np.float32)
        orientations = np.asarray(sequence.orientations, dtype=np.float32)
        poses = np.asarray(sequence.poses, dtype=np.float32)
        shapes = np.asarray(sequence.shapes, dtype=np.float32)

        rejected_indices = self._detect_rejected_indices(
            frame_ids=frame_ids,
            translations=translations,
            orientations=orientations,
            poses=poses,
            shapes=shapes,
        )
        rejected_frame_ids = tuple(int(frame_ids[index]) for index in rejected_indices)

        keep_mask = np.ones(len(frame_ids), dtype=bool)
        keep_mask[rejected_indices] = False
        kept_frame_ids = frame_ids[keep_mask]
        kept_translations = translations[keep_mask]
        kept_orientations = orientations[keep_mask]
        kept_poses = poses[keep_mask]
        kept_shapes = shapes[keep_mask]

        dense_frame_ids = np.arange(int(frame_ids.min()), int(frame_ids.max()) + 1)
        interpolated_frame_ids = tuple(int(fid) for fid in dense_frame_ids if fid not in kept_frame_ids)

        return ProcessedHandSequence(
            handedness=sequence.handedness,
            frame_ids=dense_frame_ids,
            rejected_frame_ids=rejected_frame_ids,
            interpolated_frame_ids=interpolated_frame_ids,
            translations=self._interpolate_array(kept_frame_ids, kept_translations, dense_frame_ids),
            orientations=self._interpolate_array(kept_frame_ids, kept_orientations, dense_frame_ids),
            poses=self._interpolate_array(kept_frame_ids, kept_poses, dense_frame_ids),
            shapes=self._interpolate_array(kept_frame_ids, kept_shapes, dense_frame_ids),
        )

    def _detect_rejected_indices(
        self,
        *,
        frame_ids: np.ndarray,
        translations: np.ndarray,
        orientations: np.ndarray,
        poses: np.ndarray,
        shapes: np.ndarray,
    ) -> list[int]:
        if len(frame_ids) < 3:
            return []

        rejected_indices: list[int] = []
        shape_median = np.median(shapes, axis=0)
        shape_std = np.std(shapes, axis=0) + 1e-6

        for index in range(len(frame_ids)):
            if index == 0 or index == len(frame_ids) - 1:
                continue

            shape_z = np.abs((shapes[index] - shape_median) / shape_std)
            stats = HandFrameStats(
                pose_delta_prev=float(np.linalg.norm(poses[index] - poses[index - 1])),
                pose_delta_next=float(np.linalg.norm(poses[index] - poses[index + 1])),
                orient_delta_prev=float(np.linalg.norm(orientations[index] - orientations[index - 1])),
                orient_delta_next=float(np.linalg.norm(orientations[index] - orientations[index + 1])),
                transl_delta_prev_xy=float(np.linalg.norm(translations[index, :2] - translations[index - 1, :2])),
                transl_delta_next_xy=float(np.linalg.norm(translations[index, :2] - translations[index + 1, :2])),
                shape_z_max=float(shape_z.max()),
            )
            if reject_frame(
                stats,
                pose_thresh=self.pose_thresh,
                orient_thresh=self.orient_thresh,
                transl_thresh=self.transl_thresh,
                shape_thresh=self.shape_thresh,
            ):
                rejected_indices.append(index)

        return rejected_indices

    def _interpolate_array(
        self,
        known_frame_ids: np.ndarray,
        values: np.ndarray,
        query_frame_ids: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        flat = values.reshape(values.shape[0], -1)
        interpolated = np.vstack(
            [
                np.interp(query_frame_ids, known_frame_ids, flat[:, column])
                for column in range(flat.shape[1])
            ]
        ).T
        return interpolated.reshape((len(query_frame_ids),) + values.shape[1:])
