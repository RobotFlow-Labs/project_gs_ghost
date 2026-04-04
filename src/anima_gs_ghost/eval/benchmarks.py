"""Benchmark definitions — ARCTIC Bi-CAIR + HO3D eval sequences and metric targets.

Paper: Tables 1, 2, 3.
"""

from __future__ import annotations

from dataclasses import dataclass

# 9 ARCTIC Bi-CAIR allocentric evaluation sequences
ARCTIC_SEQS: list[str] = [
    "arctic_s03_box_grab_01_1",
    "arctic_s03_notebook_grab_01_1",
    "arctic_s03_laptop_grab_01_1",
    "arctic_s03_ketchup_grab_01_1",
    "arctic_s03_espressomachine_grab_01_1",
    "arctic_s03_microwave_grab_01_1",
    "arctic_s03_waffleiron_grab_01_1",
    "arctic_s03_mixer_grab_01_1",
    "arctic_s03_capsulemachine_grab_01_1",
]

HO3D_SEQS: list[str] = [
    "SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14",
    "SB10", "SB14", "AP10", "AP11", "AP12", "AP13", "AP14",
]


@dataclass(frozen=True)
class MetricTarget:
    name: str
    paper_value: float
    tolerance: float
    lower_is_better: bool


# Paper Table 2: ARCTIC 3D interaction metrics
ARCTIC_3D_TARGETS: list[MetricTarget] = [
    MetricTarget("MPJPE_RA_h", 24.07, 1.0, lower_is_better=True),
    MetricTarget("MPJPE_RA_r", 22.71, 1.3, lower_is_better=True),
    MetricTarget("MPJPE_RA_l", 25.42, 0.6, lower_is_better=True),
    MetricTarget("CDICP", 2.26, 0.24, lower_is_better=True),
    MetricTarget("CDr", 13.40, 1.1, lower_is_better=True),
    MetricTarget("CDl", 23.41, 1.6, lower_is_better=True),
    MetricTarget("CDh", 18.40, 1.6, lower_is_better=True),
    MetricTarget("F10mm", 60.88, 2.88, lower_is_better=False),
    MetricTarget("F5mm", 34.67, 2.67, lower_is_better=False),
]

# Paper Table 3: Rendering metrics
ARCTIC_RENDER_TARGETS: list[MetricTarget] = [
    MetricTarget("PSNR", 25.93, 0.93, lower_is_better=False),
    MetricTarget("SSIM", 0.88, 0.02, lower_is_better=False),
    MetricTarget("LPIPS", 0.02, 0.01, lower_is_better=True),
]

HO3D_RENDER_TARGETS: list[MetricTarget] = [
    MetricTarget("PSNR", 21.37, 0.87, lower_is_better=False),
    MetricTarget("SSIM", 0.75, 0.02, lower_is_better=False),
    MetricTarget("LPIPS", 0.03, 0.01, lower_is_better=True),
]


def check_target(actual: float, target: MetricTarget) -> bool:
    """Check if actual metric meets the paper target within tolerance."""
    if target.lower_is_better:
        return actual <= target.paper_value + target.tolerance
    return actual >= target.paper_value - target.tolerance
