# PRD-01: Foundation & Config

> Module: GS-GHOST | Priority: P0  
> Depends on: None  
> Status: ⬜ Not started

## Objective
The repo is normalized to GS-GHOST naming, has typed config and asset validation, and exposes a stable file layout for paper-faithful implementation.

## Context (from paper)
The paper is a three-stage system, so the implementation must start with a stable contract for assets, intermediate artifacts, and runtime configuration.  
**Paper reference**: Section 3 "Method"  
Key line: "Our method reconstructs hand-object interactions from monocular RGB videos."

## Acceptance Criteria
- [ ] `anima-fujin` naming is removed from package metadata and replaced with `anima-gs-ghost`
- [ ] Config models cover preprocessing, alignment, Gaussian optimization, and evaluation
- [ ] Asset and dataset validators fail fast on missing weights or malformed sequence layout
- [ ] Artifact layout mirrors upstream `ghost_build` and `output/*` structure under `artifacts/`
- [ ] Test: `uv run pytest tests/test_config.py tests/test_layout.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/__init__.py` | package root | §3 | ~10 |
| `src/anima_gs_ghost/config.py` | Pydantic settings for all stages | §3 | ~180 |
| `src/anima_gs_ghost/assets.py` | asset registry and validation | §4.1, Supp. B | ~140 |
| `src/anima_gs_ghost/layout.py` | canonical runtime/artifact paths | Fig. 2 | ~140 |
| `configs/default.toml` | corrected GS-GHOST defaults | §3 | ~80 |
| `pyproject.toml` | package rename and deps | — | ~40 |
| `tests/test_config.py` | config coverage | — | ~80 |
| `tests/test_layout.py` | layout validation | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `sequence_root: Path` containing RGB frames or video
- `weights_root: Path`
- `datasets_root: Path`

### Outputs
- `GhostSettings`
- `AssetManifest`
- `SequenceLayout`

### Algorithm
```python
# Paper Section 3 — Method entry contract
from pydantic import BaseModel
from pathlib import Path

class GhostSettings(BaseModel):
    sequence_root: Path
    datasets_root: Path
    weights_root: Path
    sfm_method: str = "vggsfm"
    use_prior: bool = True
```

## Dependencies
```toml
pydantic = ">=2.8"
tomli = ">=2.0"
tyro = ">=0.8"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Correct paper PDF | 12.6 MB | `papers/2603.18912_GHOST.pdf` | DONE |
| HaMeR checkpoint | ~400 MB | `repositories/GHOST/preprocess/` | `gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT` |
| MANO bundle | gated | `repositories/GHOST/preprocess/_DATA/data/mano/` | manual download |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_layout.py -v
```

## References
- Paper: Section 3 "Method"
- Reference impl: `repositories/GHOST/preprocess/run_single_sequence.sh`
- Depends on: None
- Feeds into: PRD-02, PRD-03

