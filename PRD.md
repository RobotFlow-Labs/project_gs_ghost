# GS-GHOST: GHOST — Implementation PRD
## ANIMA Wave-7 Module

**Status:** PRD Suite Generated  
**Version:** 0.2  
**Date:** 2026-04-03  
**Canonical Paper:** GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting  
**Canonical Paper Link:** [arXiv:2603.18912](https://arxiv.org/abs/2603.18912)  
**Reference Repo:** [ATAboukhadra/GHOST](https://github.com/ATAboukhadra/GHOST)  
**Functional Name:** GS-GHOST  
**Stack:** PROMETHEUS  

## Build Plan — Executable PRDs

> Total PRDs: 7 | Tasks: 27 | Status: 0/27 complete

| # | PRD | Title | Priority | Tasks | Status |
|---|-----|-------|----------|-------|--------|
| 1 | [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | P0 | 4 | ⬜ |
| 2 | [PRD-02](prds/PRD-02-core-model.md) | Core Model | P0 | 6 | ⬜ |
| 3 | [PRD-03](prds/PRD-03-inference.md) | Inference Pipeline | P0 | 4 | ⬜ |
| 4 | [PRD-04](prds/PRD-04-evaluation.md) | Evaluation | P1 | 4 | ⬜ |
| 5 | [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | P1 | 3 | ⬜ |
| 6 | [PRD-06](prds/PRD-06-ros2-integration.md) | ROS2 Integration | P1 | 3 | ⬜ |
| 7 | [PRD-07](prds/PRD-07-production.md) | Production | P2 | 3 | ⬜ |

## 1. Executive Summary
GS-GHOST will reproduce the March 19, 2026 GHOST paper as closely as possible inside ANIMA: preprocessing with SAM2 plus SfM, geometric-prior retrieval from Objaverse/OpenShape, grasp-aware hand-object alignment, object-first 2D Gaussian Splatting, then hand-object joint Gaussian optimization with mesh-bound hand Gaussians. The implementation target is faithful offline reconstruction from monocular RGB sequences, not an approximate reinterpretation.

## 2. Paper Verification Status
- [x] Correct paper identified and downloaded as [papers/2603.18912_GHOST.pdf](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/papers/2603.18912_GHOST.pdf)
- [x] Local `2503.14397` PDF identified as incorrect and excluded from planning
- [x] Upstream reference repo inspected under [repositories/GHOST](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/repositories/GHOST)
- [x] Method, ablations, metrics, and supplementary hyperparameters extracted
- [ ] External weights and datasets downloaded
- [ ] Reference demo executed locally
- [ ] ANIMA implementation validated against paper tables
- **Verdict:** READY FOR BUILD

## 3. What We Take From The Paper
- Three-stage pipeline from Fig. 2 and §3: preprocessing, HO alignment, Gaussian reconstruction.
- Geometric-prior retrieval via VLM/OpenShape + IoU-based affine alignment from §3.1.1.
- HaMeR initialization plus supplementary jitter filtering thresholds from §3.1.2 and Supp. A.
- Grasp detection and HO alignment losses `L_contact`, `L_proj`, `L_temp` from §3.2.
- Hand-aware background loss `L_bkg,h` and geometric consistency loss `L_geo` from §3.3.2.
- Canonical-hand Gaussian rigging and per-face affine deformation from §3.3.3.
- Benchmark goals from Tables 1, 2, and 3.

## 4. What We Skip
- Full upstream GUI parity during the first reproduction pass.
- Direct support for all optional upstream sub-environments on day one; `openshape` and `internvl` stay pluggable.
- Real-time inference claims; the paper is still an optimization-heavy offline pipeline.
- Training-time challenge hacks tied to PyTorch 1.9.1 until ARCTIC export is needed.

## 5. What We Adapt
- Rename stale FUJIN scaffold state to `anima_gs_ghost`.
- Replace shell-script glue with typed Python entrypoints and ANIMA-style config objects.
- Keep upstream directory semantics (`ghost_build`, `output/object`, `output/combined`) while nesting them under `artifacts/`.
- Add API, Docker, ROS2, and production validation layers around the paper-faithful core.

## 6. Architecture

### Inputs
- Monocular RGB video or frame directory
- Optional object text prompt
- Optional explicit hand seed pixels and object seed pixels

### Outputs
- Object-only Gaussian point cloud and checkpoints
- Combined hand-object Gaussian checkpoint
- Rendered RGB/RGBA frames and viewer assets
- ARCTIC export tensors and evaluation reports

### Core Components
1. Preprocess masks, SfM, priors, and initial hand trajectories.
2. Align object scale plus hand translations under grasp-aware constraints.
3. Optimize object Gaussians, then jointly optimize deformable hand Gaussians plus object.
4. Evaluate against ARCTIC/HO3D metrics and publish artifacts.

## 7. Implementation Phases

### Phase 1 — Foundation + Faithful Core
- [ ] Normalize package/config names to GS-GHOST
- [ ] Implement preprocessing and HO alignment stages
- [ ] Implement object and joint Gaussian optimization wrappers

### Phase 2 — Reproduction
- [ ] Run ARCTIC eval sequences
- [ ] Reproduce rendering metrics from Table 3
- [ ] Reproduce interaction metrics from Table 2 within tolerance

### Phase 3 — ANIMA Productization
- [ ] Batch API and Docker image
- [ ] ROS2 bridge for recorded sequence ingestion
- [ ] Production preflight, export, and reporting

## 8. Datasets
| Dataset | Split | Notes |
|---------|-------|-------|
| ARCTIC Bi-CAIR | 9 allocentric eval sequences | main 3D benchmark |
| HO3D v3 | selected eval sequences | 2D rendering and object generalization |
| In-the-wild GoPro | Drill + Book | qualitative validation |
| Objaverse / Objaverse-XL | retrieval corpus | geometric prior search |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|------------------|------------------|
| None required | GS-GHOST is self-contained for first reproduction pass |

## 10. Success Criteria
- Match paper pipeline structure and loss inventory with no major omissions.
- Hit ARCTIC `CDh <= 20 cm^2` and rendering `PSNR >= 25`.
- Hit HO3D rendering `LPIPS <= 0.04`.
- Keep a single 300-frame ARCTIC run within `<= 75 min` on RTX A6000-class hardware.

## 11. Risk Assessment
- Wrong paper metadata in project scaffold can corrupt implementation if not corrected early.
- SfM quality is highly sequence-dependent; paper explicitly reports VGGSfM helps ARCTIC more than HO3D.
- `L_geo` only helps when the retrieved prior is good; the supplement documents failure cases when prior alignment is poor.
- `L_bkg,h` fails if the hand never changes contact points, leaving persistent false foreground.
- Upstream stack depends on multiple fragile CUDA/native packages and optional conda envs.

## 12. Immediate Next Step
Start with [tasks/PRD-0101.md](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/tasks/PRD-0101.md), then execute [tasks/INDEX.md](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/tasks/INDEX.md) top to bottom.

