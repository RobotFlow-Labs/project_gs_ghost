"""Typed project configuration models."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import BaseModel, Field


class ProjectSettings(BaseModel):
    name: str = "anima-gs-ghost"
    codename: str = "GS-GHOST"
    functional_name: str = "GS-GHOST"
    wave: int = 7
    paper_arxiv: str = "2603.18912"
    paper_pdf: str = "papers/2603.18912_GHOST.pdf"


class CudaSettings(BaseModel):
    enabled: bool = True
    train_extra: str = "cuda"
    device: str = "cuda:0"


class MlxSettings(BaseModel):
    enabled: bool = True
    train_extra: str = "mac"


class ComputeSettings(BaseModel):
    backend: str = "auto"
    preferred_backends: list[str] = Field(default_factory=lambda: ["mlx", "cuda", "cpu"])
    precision: str = "fp32"
    cuda: CudaSettings = Field(default_factory=CudaSettings)
    mlx: MlxSettings = Field(default_factory=MlxSettings)


class DataSettings(BaseModel):
    artifacts_root: Path = Path("artifacts")
    shared_volume: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets")
    models_root: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/models")
    datasets_root: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets")
    repos_volume: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/repos/wave7")


class PipelineSettings(BaseModel):
    object_mask_method: str = "sam2"
    sfm_method: str = "vggsfm"
    use_prior: bool = True
    prior_topk: int = 10


class TrainingSettings(BaseModel):
    object_iterations: int = 30_000
    combined_iterations: int = 30_000
    alignment_iterations: int = 500
    prior_alignment_iterations: int = 1_500


class HardwareSettings(BaseModel):
    zed2i: bool = True
    unitree_l2_lidar: bool = True
    cobot_xarm6: bool = False


class GhostSettings(BaseModel):
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)

    @classmethod
    def from_toml(cls, path: str | Path) -> "GhostSettings":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.model_validate(raw)
