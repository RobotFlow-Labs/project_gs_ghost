# PRD-07: Production

> Module: GS-GHOST | Priority: P2  
> Depends on: PRD-04, PRD-05, PRD-06  
> Status: ⬜ Not started

## Objective
GS-GHOST ships with reproducibility metadata, preflight checks, failure-mode handling, export tooling, and a production validation checklist.

## Context (from paper)
The paper and supplementary material document failure modes around bad priors, persistent occlusion, and SfM instability. Productionization must surface these instead of hiding them.  
**Paper reference**: §5, Supp. Fig. 12, Table 1 discussion

## Acceptance Criteria
- [ ] Preflight detects wrong paper metadata, missing weights, missing datasets, and unsupported CUDA deps
- [ ] Artifact manifests capture config, commit, timings, sequence id, and benchmark deltas
- [ ] Failure policies cover poor prior retrieval, bad SfM, and `L_bkg,h` persistent-contact failure mode
- [ ] Export tools package viewer assets, reports, and optional ARCTIC submission tensors
- [ ] Test: `uv run pytest tests/test_preflight.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/preflight.py` | env and asset verification | §5, Supp. Fig. 12 | ~160 |
| `src/anima_gs_ghost/reporting.py` | manifest and benchmark delta reports | §4.4, §5 | ~140 |
| `scripts/package_release.py` | bundle artifacts and reports | — | ~120 |
| `tests/test_preflight.py` | preflight checks | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- completed artifact tree
- runtime metadata

### Outputs
- release bundle
- production validation report

### Algorithm
```python
def production_gate(run):
    assert run.assets_ok
    assert run.paper_pdf == "2603.18912_GHOST.pdf"
    return summarize_failures(run)
```

## Dependencies
```toml
packaging = ">=24.1"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| completed reports/artifacts | variable | `artifacts/`, `reports/` | generated |

## Test Plan
```bash
uv run pytest tests/test_preflight.py -v
```

## References
- Paper: §5, Supp. Fig. 12
- Depends on: PRD-04, PRD-05, PRD-06
- Feeds into: release and operator handoff
