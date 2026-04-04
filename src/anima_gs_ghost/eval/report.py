"""Paper comparison report — compares reproduced metrics against Tables 1, 2, 3."""

from __future__ import annotations

import json
from pathlib import Path

from .benchmarks import (
    ARCTIC_3D_TARGETS,
    ARCTIC_RENDER_TARGETS,
    HO3D_RENDER_TARGETS,
    MetricTarget,
    check_target,
)


def compare_against_paper(
    actual: dict[str, float],
    targets: list[MetricTarget] | None = None,
) -> dict[str, dict]:
    """Compare actual metrics against paper targets.

    Args:
        actual: Dict of metric_name -> actual_value.
        targets: List of MetricTarget to compare against. Defaults to all.

    Returns:
        Dict per metric with actual, paper, delta, pass/fail status.
    """
    if targets is None:
        targets = ARCTIC_3D_TARGETS + ARCTIC_RENDER_TARGETS + HO3D_RENDER_TARGETS

    report = {}
    for target in targets:
        if target.name in actual:
            val = actual[target.name]
            delta = val - target.paper_value
            passed = check_target(val, target)
            report[target.name] = {
                "actual": val,
                "paper": target.paper_value,
                "delta": round(delta, 4),
                "tolerance": target.tolerance,
                "passed": passed,
                "direction": "lower_better" if target.lower_is_better else "higher_better",
            }
        else:
            report[target.name] = {
                "actual": None,
                "paper": target.paper_value,
                "delta": None,
                "tolerance": target.tolerance,
                "passed": False,
                "direction": "lower_better" if target.lower_is_better else "higher_better",
                "note": "MISSING — not evaluated",
            }
    return report


def generate_report_markdown(
    report: dict[str, dict],
    title: str = "GS-GHOST Reproduction Report",
) -> str:
    """Generate a Markdown comparison table."""
    lines = [f"# {title}", "", "| Metric | Paper | Ours | Delta | Status |",
             "|--------|-------|------|-------|--------|"]
    for name, data in report.items():
        actual = f"{data['actual']:.4f}" if data["actual"] is not None else "N/A"
        delta = f"{data['delta']:+.4f}" if data["delta"] is not None else "N/A"
        status = "PASS" if data["passed"] else "FAIL"
        if data.get("note"):
            status = "MISSING"
        lines.append(f"| {name} | {data['paper']} | {actual} | {delta} | {status} |")

    n_pass = sum(1 for d in report.values() if d["passed"])
    n_total = len(report)
    lines.extend(["", f"**Score: {n_pass}/{n_total} metrics within tolerance**"])
    return "\n".join(lines)


def write_report(
    actual: dict[str, float],
    output_dir: Path,
    targets: list[MetricTarget] | None = None,
) -> tuple[Path, Path]:
    """Generate and write both JSON and Markdown reports.

    Returns:
        Tuple of (json_path, md_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = compare_against_paper(actual, targets)

    json_path = output_dir / "paper_comparison.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = output_dir / "PAPER_COMPARISON.md"
    md_path.write_text(generate_report_markdown(report))

    return json_path, md_path
