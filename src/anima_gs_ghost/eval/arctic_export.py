"""ARCTIC evaluation export bridge — §4.2.

Exports ARCTIC-compatible prediction tensors for the HOLD/Bi-CAIR protocol.
Each sequence produces a .pt file with hand and object predictions.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .benchmarks import ARCTIC_SEQS


def export_sequence(
    hand_verts_right: torch.Tensor,
    hand_verts_left: torch.Tensor,
    object_verts: torch.Tensor,
    seq_name: str,
    output_dir: Path,
) -> Path:
    """Export ARCTIC-compatible prediction tensor for one sequence.

    Args:
        hand_verts_right: [T, 778, 3] right hand vertices.
        hand_verts_left: [T, 778, 3] left hand vertices.
        object_verts: [T, N_obj, 3] object vertices.
        seq_name: Sequence identifier (must be in ARCTIC_SEQS).
        output_dir: Output directory for .pt files.

    Returns:
        Path to the exported .pt file.
    """
    if seq_name not in ARCTIC_SEQS:
        raise ValueError(f"Unknown ARCTIC sequence: {seq_name}. Expected one of {ARCTIC_SEQS}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "right_hand_verts": hand_verts_right.cpu(),
        "left_hand_verts": hand_verts_left.cpu(),
        "object_verts": object_verts.cpu(),
        "seq_name": seq_name,
    }
    path = output_dir / f"{seq_name}.pt"
    torch.save(payload, path)
    return path


def export_all_sequences(
    predictions: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
) -> list[Path]:
    """Export predictions for all 9 ARCTIC sequences.

    Args:
        predictions: Dict mapping seq_name -> {"right_hand_verts", "left_hand_verts", "object_verts"}.
        output_dir: Output directory.

    Returns:
        List of exported .pt file paths.
    """
    paths = []
    for seq in ARCTIC_SEQS:
        if seq in predictions:
            p = predictions[seq]
            path = export_sequence(
                p["right_hand_verts"],
                p["left_hand_verts"],
                p["object_verts"],
                seq,
                output_dir,
            )
            paths.append(path)
    return paths
