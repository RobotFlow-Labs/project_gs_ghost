"""Reproducibility packaging — capture config, git state, metrics for release."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import GhostSettings


@dataclass
class RunReport:
    module: str = "gs-ghost"
    version: str = "0.2.0"
    paper_arxiv: str = "2603.18912"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = ""
    git_branch: str = ""
    config: dict = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    device: str = ""
    training_time_s: float = 0.0
    checkpoint_path: str = ""


def get_git_info() -> dict[str, str]:
    """Get current git commit and branch."""
    info = {"commit": "", "branch": ""}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def build_run_manifest(
    cfg: GhostSettings,
    metrics: dict[str, float],
    device: str = "cuda:1",
    training_time_s: float = 0.0,
    checkpoint_path: str = "",
) -> RunReport:
    """Build a complete run report for reproducibility."""
    git = get_git_info()
    return RunReport(
        paper_arxiv=cfg.project.paper_arxiv,
        git_commit=git["commit"],
        git_branch=git["branch"],
        config=cfg.model_dump(),
        metrics=metrics,
        device=device,
        training_time_s=training_time_s,
        checkpoint_path=checkpoint_path,
    )


def write_training_report(
    report: RunReport,
    output_dir: Path,
) -> Path:
    """Write TRAINING_REPORT.md and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "training_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "module": report.module,
            "version": report.version,
            "paper_arxiv": report.paper_arxiv,
            "timestamp": report.timestamp,
            "git_commit": report.git_commit,
            "git_branch": report.git_branch,
            "device": report.device,
            "training_time_s": report.training_time_s,
            "checkpoint_path": report.checkpoint_path,
            "metrics": report.metrics,
        }, f, indent=2)

    # Markdown
    md_path = output_dir / "TRAINING_REPORT.md"
    lines = [
        "# GS-GHOST Training Report",
        f"\n- **Module:** {report.module} v{report.version}",
        f"- **Paper:** arXiv:{report.paper_arxiv}",
        f"- **Git:** {report.git_commit[:8]} ({report.git_branch})",
        f"- **Device:** {report.device}",
        f"- **Training time:** {report.training_time_s:.0f}s",
        f"- **Checkpoint:** {report.checkpoint_path}",
        "\n## Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in report.metrics.items():
        lines.append(f"| {k} | {v:.4f} |")
    md_path.write_text("\n".join(lines))

    return md_path
