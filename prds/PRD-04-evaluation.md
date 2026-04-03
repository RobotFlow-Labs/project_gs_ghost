# PRD-04: Evaluation

> Module: GS-GHOST | Priority: P1  
> Depends on: PRD-02, PRD-03  
> Status: ⬜ Not started

## Objective
The implementation reproduces the paper’s reported evaluation protocol and outputs a report that compares ANIMA results against Tables 1, 2, and 3.

## Context (from paper)
The paper evaluates 3D interaction accuracy on ARCTIC, 2D rendering quality on ARCTIC and HO3D, and runtime on a 300-frame ARCTIC sequence.  
**Paper reference**: §4.1, §4.2, §4.3, §4.4  
Key line: "All 3D evaluation metrics follow the official protocol released by HOLD."

## Acceptance Criteria
- [ ] ARCTIC export bridge supports the 9 Bi-CAIR allocentric sequences
- [ ] 2D metrics compute PSNR, SSIM, LPIPS for ARCTIC and HO3D
- [ ] 3D metrics report MPJPE, `CDICP`, `CDr`, `CDl`, `CDh`, `F10mm`, `F5mm`
- [ ] Markdown and JSON report compare reproduced metrics against the paper
- [ ] Test: `uv run pytest tests/test_eval_metrics.py tests/test_arctic_export.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/eval/rendering.py` | PSNR/SSIM/LPIPS computation | §4.2, Table 3 | ~140 |
| `src/anima_gs_ghost/eval/arctic_export.py` | ARCTIC challenge tensor export | §4.2 | ~180 |
| `src/anima_gs_ghost/eval/report.py` | paper-comparison report | §4.4 | ~160 |
| `scripts/evaluate_rendering.py` | eval CLI | §4.2 | ~100 |
| `tests/test_eval_metrics.py` | rendering metric tests | — | ~100 |
| `tests/test_arctic_export.py` | export mapping tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- predicted rendered RGBA frames `[T, H, W, 4]`
- ground-truth RGBA frames `[T, H, W, 4]`
- predicted ARCTIC export tensors

### Outputs
- `metrics_summary.json`
- `paper_gap_report.md`

### Algorithm
```python
# Paper Section 4 — evaluation against reported tables
def summarize_against_paper(results, paper):
    return {
        name: {"actual": results[name], "paper": paper[name], "delta": results[name] - paper[name]}
        for name in paper
    }
```

## Dependencies
```toml
lpips = ">=0.1.4"
scikit-image = ">=0.24"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| ARCTIC GT and HOLD protocol data | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/arctic_bicair/` | official challenge/HOLD scripts |
| HO3D GT frames | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/HO3D_v3/` | official dataset |

## Test Plan
```bash
uv run pytest tests/test_eval_metrics.py tests/test_arctic_export.py -v
```

## References
- Paper: §4.1-4.4, Tables 1-3
- Reference impl: `repositories/GHOST/evaluate.py`, `repositories/GHOST/scene/gaussian_model_mano.py`
- Depends on: PRD-02, PRD-03
- Feeds into: PRD-07

