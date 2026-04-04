# GS-GHOST — Execution Ledger

Resume rule: Read this file completely before writing code.
This project covers exactly one paper implementation track plus ANIMA wrappers.

## 1. Working Rules
- Work only inside `project_gs_ghost/`
- Prefix every commit with `[GS-GHOST]`
- Use `uv`, never `pip`
- Use Python `3.11`
- CUDA backend: torch cu128 on L4 GPU
- Shared rasterizer: `/mnt/forge-data/shared_infra/cuda_extensions/gaussian_semantic_rasterization/`
- Hand model: abstracted (MANO/NIMBLE/Handy swappable via HandModel protocol)

## 2. Paper
- **Title**: GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting
- **ArXiv**: `2603.18912`
- **Link**: https://arxiv.org/abs/2603.18912
- **Repo**: https://github.com/ATAboukhadra/GHOST
- **Compute**: GPU server (L4 23GB), CUDA_VISIBLE_DEVICES=1
- **Verification status**: PDF + Repo + Planning + Implementation ✅

## 3. Current Status
- **Date**: 2026-04-04
- **Phase**: Full build complete — all PRDs implemented
- **MVP Readiness**: 65%
- **Current active checkpoint**: All PRDs (01-07) code complete

### Completed This Session (GPU Server Build)
1. Set up .venv with Python 3.11 + torch cu128 on GPU server
2. Updated all configs for GPU server paths (/mnt/forge-data, /mnt/artifacts-datai)
3. **PRD-0205**: Grasp detection (Eq. 2) + HO alignment losses (Eqs. 3-6) + optimizer
4. **PRD-0206**: Gaussian model (shared rasterizer), object GS stage, hand GS rigging (Eqs. 10-12), reconstruction losses (L_rgb, L_bkg_h, L_geo — Eqs. 8-9)
5. **PRD-0301-0304**: Pipeline orchestration (preprocess → alignment → object GS → combined GS → export), sequence CLI, artifact manifest, viewer export
6. **PRD-0401-0404**: Benchmark definitions (9 ARCTIC seqs, HO3D), 2D rendering eval (PSNR/SSIM/LPIPS), ARCTIC export bridge, paper comparison report
7. **PRD-0501-0503**: FastAPI service (/health, /ready, /info, /reconstruct), background job runner, Dockerfile.serve + docker-compose.serve.yml + .env.serve
8. **PRD-0601-0603**: ROS2 contracts (GhostRequest/Status/Result), batch node with dispatch, launch file
9. **PRD-0701-0703**: Preflight checks (paper ID, CUDA, rasterizer, weights), reproducibility packaging, production checklist
10. Hand model abstraction: HandModel protocol + SimpleHandModel for testing
11. Updated anima_module.yaml with full ANIMA infra metadata
12. 74/74 tests passing, ruff lint clean

### Verified
- `uv run pytest tests/ -v` → 74/74 PASS ✅
- `uv run ruff check src/ tests/` → clean ✅
- Shared rasterizer importable from `/mnt/forge-data/shared_infra/cuda_extensions/` ✅
- torch 2.11.0+cu128, CUDA available, 8x L4 GPUs detected ✅

## 4. Immediate Next Tasks
1. **Download/provision hand model weights** — MANO (gated) or obtain NIMBLE pm_dict
2. **Download ARCTIC eval sequences** — needed for benchmark evaluation
3. **Wire actual SAM2/VGGSfM/HaMeR** execution into preprocessing stages (currently MOCK)
4. **CUDA optimization** — implement clone/split in densification, profile and tune
5. **Training run** — once data + models available, run on GPU 1
6. **Export pipeline** — pth → safetensors → ONNX → TRT FP16 + FP32

## 5. Known Blockers
- **Hand model weights**: MANO is gated (Max Planck). Handy/NIMBLE repos are on disk but lack actual model weight files (require registration).
- **ARCTIC dataset**: Not yet on disk. Need download for benchmark evaluation.
- **Preprocessing stages**: SAM2/VGGSfM/HaMeR are wired as MOCK interfaces — need actual model loading to run end-to-end.

## 6. Architecture Summary
```
src/anima_gs_ghost/
├── __init__.py, version.py, config.py, device.py, layout.py, assets.py
├── alignment/
│   ├── prior_alignment.py    — IoU-based prior-mask alignment (Eq. 1)
│   ├── grasp_detection.py    — Motion cosine similarity (Eq. 2)
│   └── ho_alignment.py       — L_contact, L_proj, L_temp (Eqs. 3-6)
├── preprocess/
│   ├── object_masks.py       — SAM2 wrapper
│   ├── sfm.py                — HLoc/VGGSfM wrappers
│   ├── prior_retrieval.py    — OpenShape/text retrieval
│   └── hand_init.py          — HaMeR jitter rejection + interpolation
├── reconstruction/
│   ├── gaussian_model.py     — Core GS model (shared rasterizer)
│   ├── losses.py             — L_rgb, L_bkg_h, L_geo (§3.3.2)
│   ├── object_gs.py          — Object-only stage (30k iters)
│   └── hand_gs.py            — Canonical hand rigging (Eqs. 10-12)
├── hand_model/
│   ├── base.py               — HandModel protocol
│   └── simple.py             — Test-only hand model
├── eval/
│   ├── benchmarks.py         — ARCTIC/HO3D targets
│   ├── rendering.py          — PSNR/SSIM/LPIPS
│   ├── arctic_export.py      — .pt export bridge
│   └── report.py             — Paper comparison
├── api/
│   ├── models.py             — Request/response schemas
│   └── app.py                — FastAPI service
├── ros2/
│   ├── messages.py           — GhostRequest/Status/Result
│   └── node.py               — Batch reconstruction node
├── pipeline.py               — End-to-end orchestrator
├── artifacts.py              — Run directory + manifest
├── viewer.py                 — Viewer asset export
├── preflight.py              — Production checks
├── reporting.py              — Training reports
└── serve.py                  — ANIMA serve entrypoint
```

## 7. Data Paths (GPU Server)
- Models: `/mnt/forge-data/models/`
- Datasets: `/mnt/forge-data/datasets/`
- Shared CUDA: `/mnt/forge-data/shared_infra/cuda_extensions/`
- Checkpoints: `/mnt/artifacts-datai/checkpoints/project_gs_ghost/`
- Logs: `/mnt/artifacts-datai/logs/project_gs_ghost/`
- Exports: `/mnt/artifacts-datai/exports/project_gs_ghost/`

## 8. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | Codex | Replaced stale FUJIN scaffold with GS-GHOST Python 3.11 foundation |
| 2026-04-03 | Codex | Added uv-safe dependency split for mac bootstrap vs Linux CUDA install |
| 2026-04-03 | Codex | Implemented SAM2/SfM stage wrappers and prior retrieval contract with tests |
| 2026-04-03 | Codex | Implemented prior-mask alignment and HaMeR jitter/interpolation contracts with tests |
| 2026-04-04 | Opus | GPU server build: venv + torch cu128, all PRDs 0205-0703, ANIMA infra, 74/74 tests |
