import numpy as np

from anima_gs_ghost.alignment.prior_alignment import (
    PriorAligner,
    PriorAlignmentCandidate,
    PriorAlignmentHypothesis,
    PriorTransform,
)
from anima_gs_ghost.preprocess.hand_init import HaMeRPostprocessor, HandFrameStats, HandSequence, reject_frame


def test_prior_loss_matches_one_minus_iou() -> None:
    aligner = PriorAligner()
    projected = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    target = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)

    loss = aligner.loss(projected, target)

    assert np.isclose(loss, 0.5)


def test_prior_alignment_selects_best_candidate_by_iou() -> None:
    aligner = PriorAligner()
    object_masks = (
        np.asarray(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    perfect = PriorAlignmentCandidate(
        uid="best-prior",
        vertices=vertices,
        hypotheses=(
            PriorAlignmentHypothesis(
                transform=PriorTransform.identity(),
                projected_masks=object_masks,
            ),
        ),
    )
    weak = PriorAlignmentCandidate(
        uid="weak-prior",
        vertices=vertices,
        hypotheses=(
            PriorAlignmentHypothesis(
                transform=PriorTransform.identity(),
                projected_masks=(np.zeros_like(object_masks[0]),),
            ),
        ),
    )

    result = aligner.align((weak, perfect), object_masks)

    assert result.uid == "best-prior"
    assert np.isclose(result.mean_iou, 1.0)
    assert np.isclose(result.loss, 0.0)


def test_prior_alignment_outputs_transformed_vertices() -> None:
    aligner = PriorAligner()
    vertices = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    transform = PriorTransform(
        quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        translation=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        scale=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
    candidate = PriorAlignmentCandidate(
        uid="scaled",
        vertices=vertices,
        hypotheses=(
            PriorAlignmentHypothesis(
                transform=transform,
                projected_masks=(np.asarray([[1]], dtype=np.uint8),),
            ),
        ),
    )

    result = aligner.align((candidate,), (np.asarray([[1]], dtype=np.uint8),))

    assert np.allclose(result.aligned_vertices, np.asarray([[4.0, 3.0, 4.0]], dtype=np.float32))


def test_jitter_reject_frame_flags_translation_spike() -> None:
    stats = HandFrameStats(transl_delta_prev_xy=3.0, transl_delta_next_xy=3.5)

    assert reject_frame(stats)


def test_jitter_postprocess_interpolates_rejected_frame() -> None:
    processor = HaMeRPostprocessor()
    sequence = HandSequence(
        handedness="right",
        frame_ids=np.asarray([0, 1, 2], dtype=int),
        translations=np.asarray([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
        orientations=np.asarray([[0.0], [5.0], [0.2]], dtype=np.float32),
        poses=np.asarray([[0.0], [5.0], [0.2]], dtype=np.float32),
        shapes=np.asarray([[0.1], [0.1], [0.1]], dtype=np.float32),
    )

    result = processor.process(sequence)

    assert result.rejected_frame_ids == (1,)
    assert result.interpolated_frame_ids == (1,)
    assert np.allclose(result.translations[1], np.asarray([1.0, 1.0, 0.0], dtype=np.float32))


def test_jitter_postprocess_handles_left_and_right_symmetrically() -> None:
    processor = HaMeRPostprocessor()
    base = dict(
        frame_ids=np.asarray([0, 1, 2], dtype=int),
        translations=np.asarray([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
        orientations=np.asarray([[0.0], [4.0], [0.1]], dtype=np.float32),
        poses=np.asarray([[0.0], [4.0], [0.1]], dtype=np.float32),
        shapes=np.asarray([[0.1], [0.1], [0.1]], dtype=np.float32),
    )

    right = processor.process(HandSequence(handedness="right", **base))
    left = processor.process(HandSequence(handedness="left", **base))

    assert right.rejected_frame_ids == left.rejected_frame_ids
    assert np.allclose(right.translations, left.translations)
