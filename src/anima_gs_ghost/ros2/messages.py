"""ROS2 message contracts for GS-GHOST.

These are Python dataclass contracts that mirror the ROS2 message types.
Actual .msg definitions would be in a separate anima_msgs package.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GhostRequest:
    """Request to reconstruct a sequence."""
    sequence_path: str = ""
    sequence_name: str = ""
    prompt: str = ""
    sfm_method: str = "vggsfm"
    device: str = "cuda:1"


@dataclass
class GhostStatus:
    """Status update during reconstruction."""
    job_id: str = ""
    sequence: str = ""
    status: str = "unknown"  # queued, running, done, failed
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    manifest_path: str = ""


@dataclass
class GhostResult:
    """Final reconstruction result."""
    job_id: str = ""
    sequence: str = ""
    manifest_path: str = ""
    object_checkpoint: str = ""
    combined_checkpoint: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
