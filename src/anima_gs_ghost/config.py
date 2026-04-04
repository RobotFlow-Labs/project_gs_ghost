"""Typed project configuration models for GS-GHOST (GPU server)."""

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
    device: str = "cuda:1"


class MlxSettings(BaseModel):
    enabled: bool = False
    train_extra: str = "mac"


class ComputeSettings(BaseModel):
    backend: str = "cuda"
    preferred_backends: list[str] = Field(default_factory=lambda: ["cuda", "mlx", "cpu"])
    precision: str = "bf16"
    cuda: CudaSettings = Field(default_factory=CudaSettings)
    mlx: MlxSettings = Field(default_factory=MlxSettings)


class ModelPaths(BaseModel):
    sam2_large: Path = Path("/mnt/forge-data/models/sam2.1-hiera-large/sam2.1_hiera_large.pt")
    sam2_base: Path = Path("/mnt/forge-data/models/sam2.1_hiera_base_plus.pt")
    sam_vit_h: Path = Path("/mnt/forge-data/models/sam_vit_h_4b8939.pth")
    dinov2_vitb14: Path = Path("/mnt/forge-data/models/dinov2_vitb14_pretrain.pth")
    hamer_data: Path = Path("/mnt/forge-data/models/hamer_demo_data.tar.gz")


class DataSettings(BaseModel):
    artifacts_root: Path = Path("/mnt/artifacts-datai")
    models_root: Path = Path("/mnt/forge-data/models")
    datasets_root: Path = Path("/mnt/forge-data/datasets")
    repos_volume: Path = Path("/mnt/forge-data/repos")
    shared_infra: Path = Path("/mnt/forge-data/shared_infra")
    models: ModelPaths = Field(default_factory=ModelPaths)


class PipelineSettings(BaseModel):
    object_mask_method: str = "sam2"
    sfm_method: str = "vggsfm"
    use_prior: bool = True
    prior_topk: int = 10


class AlignmentTrainingSettings(BaseModel):
    optimizer: str = "adam"
    lr: float = 0.05
    iterations: int = 500
    lambda_contact: float = 1e3
    lambda_proj: float = 1e-1
    lambda_temp: float = 10.0


class PriorAlignmentSettings(BaseModel):
    optimizer: str = "adamw"
    lr: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)
    iterations: int = 1500


class ObjectGSSettings(BaseModel):
    optimizer: str = "adam"
    iterations: int = 30_000
    sh_degree: int = 3
    densify_until_iter: int = 15_000
    lambda_background: float = 0.3
    lambda_geo: float = 5.0
    tau_out: float = 0.05
    tau_fill: float = 0.005


class CombinedGSSettings(BaseModel):
    optimizer: str = "adam"
    iterations: int = 30_000
    gaussians_per_edge: int = 10
    optimize_hand: bool = True
    transl_lr: float = 1e-4


class TrainingSettings(BaseModel):
    object_iterations: int = 30_000
    combined_iterations: int = 30_000
    alignment_iterations: int = 500
    prior_alignment_iterations: int = 1_500
    batch_size: str | int = "auto"
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    seed: int = 42
    alignment: AlignmentTrainingSettings = Field(default_factory=AlignmentTrainingSettings)
    prior_alignment: PriorAlignmentSettings = Field(default_factory=PriorAlignmentSettings)
    object_gs: ObjectGSSettings = Field(default_factory=ObjectGSSettings)
    combined_gs: CombinedGSSettings = Field(default_factory=CombinedGSSettings)


class GaussianSettings(BaseModel):
    sh_degree: int = 3
    tau_sim: float = 0.5


class CheckpointSettings(BaseModel):
    output_dir: Path = Path("/mnt/artifacts-datai/checkpoints/project_gs_ghost")
    save_every_n_steps: int = 500
    keep_top_k: int = 2
    metric: str = "val_loss"
    mode: str = "min"


class LoggingSettings(BaseModel):
    log_dir: Path = Path("/mnt/artifacts-datai/logs/project_gs_ghost")
    tensorboard_dir: Path = Path("/mnt/artifacts-datai/tensorboard/project_gs_ghost")


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
    gaussian: GaussianSettings = Field(default_factory=GaussianSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)

    @classmethod
    def from_toml(cls, path: str | Path) -> "GhostSettings":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.model_validate(raw)
