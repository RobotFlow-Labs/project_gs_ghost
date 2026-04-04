"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SequenceRequest(BaseModel):
    sequence_path: str
    sequence_name: str | None = None
    prompt: str | None = None
    sfm_method: str = "vggsfm"
    device: str = "cuda:1"


class JobStatus(BaseModel):
    job_id: str
    status: str = "queued"  # queued, running, done, failed
    sequence: str | None = None
    progress: float = 0.0
    stage: str | None = None
    error: str | None = None
    manifest_path: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    module: str = "anima-gs-ghost"
    version: str = "0.2.0"
    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None


class InfoResponse(BaseModel):
    module: str = "anima-gs-ghost"
    codename: str = "GS-GHOST"
    paper: str = "GHOST: Hand-Object Reconstruction via 3DGS"
    arxiv: str = "2603.18912"
    version: str = "0.2.0"
    capabilities: list[str] = Field(
        default_factory=lambda: [
            "hand_object_reconstruction",
            "gaussian_splatting",
            "monocular_rgb",
        ]
    )
