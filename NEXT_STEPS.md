# GS-GHOST — Execution Ledger

Resume rule: Read this file completely before writing code.
This project covers exactly one paper implementation track plus ANIMA wrappers.

## 1. Working Rules
- Work only inside `project_gs_ghost/`
- Prefix every commit with `[GS-GHOST]`
- Use `uv`, never `pip`
- Use Python `3.11`
- Keep macOS bootstrap light, but preserve the Linux CUDA path for later training
- Verify paper and reference repo behavior before implementing paper-facing stages

## 2. Paper
- **Title**: GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting
- **ArXiv**: `2603.18912`
- **Link**: https://arxiv.org/abs/2603.18912
- **Repo**: https://github.com/ATAboukhadra/GHOST
- **Compute**: Mac local for scaffold + Linux CUDA for training
- **Verification status**: PDF ✅ | Repo ✅ | Planning docs ✅ | Implementation in progress ⬜

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: PRD-02 preprocessing + alignment foundations
- **MVP Readiness**: 30%
- **Current active checkpoint**: `PRD-0205` grasp detection and HO alignment losses

### Completed This Session
1. Renamed the stale FUJIN scaffold to GS-GHOST package structure
2. Upgraded project runtime to Python `3.11` with `uv`
3. Added typed config, asset checks, sequence layout helpers, module manifest, and service container scaffold
4. Split Linux CUDA source builds out of base `uv sync` into `scripts/install_cuda_linux.sh`
5. Implemented pure-Python preprocessing wrappers for:
   - SAM2 object mask stage
   - HLoc / VGGSfM SfM stage
   - prompt + prior retrieval contract
6. Implemented prior-mask alignment contract with explicit IoU loss and best-candidate selection
7. Implemented HaMeR jitter rejection + interpolation postprocessing
8. Added foundation, preprocessing, retrieval, and alignment tests

### Verified
- `uv sync --group dev` ✅
- `uv run pytest tests/test_config.py tests/test_layout.py tests/test_preprocess_stages.py tests/test_prior_retrieval.py tests/test_alignment.py -v` ✅
- `uv run ruff check src/ tests/` ✅

## 4. Immediate Next Tasks
1. Implement `tasks/PRD-0205.md`:
   - `src/anima_gs_ghost/alignment/grasp_detection.py`
   - HO alignment loss terms `L_contact`, `L_proj`, `L_temp`
2. Implement `tasks/PRD-0206.md`:
   - object Gaussian optimization shell
   - hand Gaussian optimization shell
3. Start `tasks/PRD-0301.md` once the PRD-02 loss and optimization contracts are in place

## 5. Known Blockers
- True paper execution still needs external assets/checkpoints not yet present:
  - `repositories/GHOST/preprocess/_DATA/data/mano/MANO_RIGHT.pkl`
  - `repositories/GHOST/preprocess/hamer_demo_data.tar.gz`
  - SAM2 / HaMeR / VGGSfM runtime checkpoints and model dependencies
  - Objaverse / OpenShape retrieval assets
- Base macOS scaffold is working; actual reconstruction training remains blocked until Linux CUDA environment + assets are provisioned

## 6. Runtime / Install Notes
- Local mac bootstrap:
  - `uv venv .venv --python /Users/ilessio/.local/bin/python3.11`
  - `uv sync --group dev`
- Future Linux CUDA bootstrap:
  - `./scripts/install_cuda_linux.sh`

## 7. Data Paths
- Shared datasets root: `/Volumes/AIFlowDev/RobotFlowLabs/datasets`
- Models root: `/Volumes/AIFlowDev/RobotFlowLabs/datasets/models`
- Repo mirror root: `/Volumes/AIFlowDev/RobotFlowLabs/repos/wave7`

## 8. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | Codex | Replaced stale FUJIN scaffold with GS-GHOST Python 3.11 foundation |
| 2026-04-03 | Codex | Added uv-safe dependency split for mac bootstrap vs Linux CUDA install |
| 2026-04-03 | Codex | Implemented SAM2/SfM stage wrappers and prior retrieval contract with tests |
| 2026-04-03 | Codex | Implemented prior-mask alignment and HaMeR jitter/interpolation contracts with tests |
