# PRD-03: Inference Pipeline

> Module: GS-GHOST | Priority: P0  
> Depends on: PRD-01, PRD-02  
> Status: ⬜ Not started

## Objective
A single command ingests one monocular RGB sequence and produces GHOST preprocessing artifacts, object reconstruction, combined reconstruction, and viewer-ready outputs.

## Context (from paper)
The paper runs as a staged offline pipeline that first preprocesses, then aligns, then optimizes object and hand-object Gaussian models.  
**Paper reference**: Fig. 2, §3  
Key line: "The framework operates in three stages."

## Acceptance Criteria
- [ ] `scripts/run_sequence.py` exposes the full paper pipeline for one sequence
- [ ] Supports `--sfm {hloc,vggsfm}`, `--prompt`, `--use-prior`, `--hands {1,2}`
- [ ] Produces `ghost_build`, `output/object`, `output/combined`, and viewer artifacts
- [ ] Emits an artifact manifest and command log for reproducibility
- [ ] Test: `uv run pytest tests/test_pipeline_runner.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `scripts/run_sequence.py` | main orchestrator | Fig. 2 | ~220 |
| `src/anima_gs_ghost/pipeline.py` | stage composition | §3 | ~180 |
| `src/anima_gs_ghost/viewer.py` | animatable hand-avatar export wrapper | Supp. Fig. 13 | ~120 |
| `src/anima_gs_ghost/artifacts.py` | manifest writer | — | ~120 |
| `tests/test_pipeline_runner.py` | dry-run coverage | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- `video_or_frames`: path
- `text_prompt`: optional string
- `sfm_method`: `"hloc"` or `"vggsfm"`

### Outputs
- `artifacts/<seq>/ghost_build/*`
- `artifacts/<seq>/output/object/*`
- `artifacts/<seq>/output/combined/*`
- `artifacts/<seq>/viewer/*`

### Algorithm
```python
# Paper Figure 2 — staged execution
def run_sequence(cfg):
    stage1 = preprocess_sequence(cfg)
    stage2 = align_hands_and_object(stage1, cfg)
    object_ckpt = optimize_object(stage2, cfg)
    combined_ckpt = optimize_hand_object(stage2, object_ckpt, cfg)
    return export_artifacts(stage1, stage2, object_ckpt, combined_ckpt, cfg)
```

## Dependencies
```toml
imageio = ">=2.35"
rich = ">=13.7"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Input sequence frames | variable | `artifacts/<seq>/frames/` | generated from source video |
| Hand Gaussian templates | generated | `artifacts/<seq>/ghost_build/canonical/` | produced during preprocessing |

## Test Plan
```bash
uv run pytest tests/test_pipeline_runner.py -v
```

## References
- Paper: Fig. 2, §3
- Reference impl: `repositories/GHOST/preprocess/run_single_sequence.sh`, `repositories/GHOST/scripts/train_object.bash`, `repositories/GHOST/scripts/train_combined.bash`
- Depends on: PRD-01, PRD-02
- Feeds into: PRD-04, PRD-05, PRD-06

