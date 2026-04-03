# PRD-02: Core Model

> Module: GS-GHOST | Priority: P0  
> Depends on: PRD-01  
> Status: ⬜ Not started

## Objective
All paper-defined stages, losses, and geometry transformations are implemented faithfully enough to reproduce the GHOST pipeline end to end.

## Context (from paper)
GHOST is driven by three innovations: prior retrieval plus consistency, grasp-aware alignment, and hand-aware background reasoning.  
**Paper reference**: Sections 3.1, 3.2, 3.3  
Key line: "GHOST represents both hands and objects as dense, view-consistent Gaussian discs."

## Acceptance Criteria
- [ ] SAM2 object masks, hand masks, SfM wrappers, prior retrieval, and prior-mask alignment are implemented
- [ ] HaMeR postprocessing includes supplementary rejection thresholds and interpolation rules
- [ ] Grasp detection and HO alignment reproduce Eqs. (2-6)
- [ ] Object optimization reproduces `L_rgb`, `L_bkg,h`, and `L_geo`
- [ ] Hand rigging reproduces the canonical-face affine deformation from Eqs. (10-12)
- [ ] Test: `uv run pytest tests/test_alignment.py tests/test_losses.py tests/test_rigging.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/preprocess/object_masks.py` | SAM2 wrapper | §3.1 | ~120 |
| `src/anima_gs_ghost/preprocess/sfm.py` | HLoc/VGGSfM wrappers + outputs | §3.1 | ~180 |
| `src/anima_gs_ghost/preprocess/prior_retrieval.py` | InternVL/OpenShape/Objaverse retrieval | §3.1.1 | ~180 |
| `src/anima_gs_ghost/preprocess/hand_init.py` | HaMeR + jitter cleanup | §3.1.2, Supp. A | ~220 |
| `src/anima_gs_ghost/alignment/grasp_detection.py` | motion similarity detection | §3.2.1 | ~100 |
| `src/anima_gs_ghost/alignment/prior_alignment.py` | optimize `(R_p,T_p,S_p)` with IoU | Eq. (1) | ~180 |
| `src/anima_gs_ghost/alignment/ho_alignment.py` | `L_contact`, `L_proj`, `L_temp` | Eq. (3-6) | ~220 |
| `src/anima_gs_ghost/reconstruction/losses.py` | `L_bkg,h`, `L_geo` and helpers | §3.3.2 | ~160 |
| `src/anima_gs_ghost/reconstruction/object_gs.py` | object-only GS stage | §3.3.2 | ~220 |
| `src/anima_gs_ghost/reconstruction/hand_gs.py` | canonical hand Gaussian rigging | §3.3.3 | ~240 |
| `tests/test_alignment.py` | alignment equations | — | ~120 |
| `tests/test_losses.py` | loss units and thresholds | — | ~120 |
| `tests/test_rigging.py` | hand rigging transform invariants | — | ~120 |

## Architecture Detail (from paper)

### Inputs
- `frames`: `Tensor[T, H, W, 3]`
- `M_o`: `BoolTensor[T, H, W]`
- `M_h`: `BoolTensor[T, H_hands, H, W]`
- `P_sfm`: `FloatTensor[N_sfm, 3]`
- `O`: `FloatTensor[N_prior, 3]`
- `V_t^h`: `FloatTensor[T, H_hands, 778, 3]`
- `J_t^h`: `FloatTensor[T, H_hands, 21, 3]`

### Outputs
- `aligned_prior`: `FloatTensor[N_prior, 3]`
- `hand_translations`: `FloatTensor[T, H_hands, 3]`
- `object_scale`: `FloatTensor[1]`
- `object_gaussians`: centers `[G_o, 3]`, scales `[G_o, 2]`, rotations `[G_o, 4]`, opacity `[G_o, 1]`, SH `[G_o, 16, 3]`
- `hand_gaussians`: centers `[G_h, 3]`, scales `[G_h, 2]`, rotations `[G_h, 4]`, opacity `[G_h, 1]`, SH `[G_h, 16, 3]`

### Algorithm
```python
# Paper Sections 3.2 and 3.3 — alignment and Gaussian optimization
class GhostCore:
    def detect_grasp(self, obj_motion_xy, hand_motion_xy) -> bool:
        return cosine_similarity(obj_motion_xy, hand_motion_xy) > 0.5

    def ho_alignment_loss(self, contact, proj, temp):
        return 1e3 * contact + 1e-1 * proj + 10.0 * temp

    def object_loss(self, rgb, bkg_hand_aware, geo):
        return rgb + bkg_hand_aware + 5.0 * geo
```

## Dependencies
```toml
torch = ">=2.0"
torchvision = ">=0.15"
opencv-python = ">=4.10"
numpy = ">=1.26"
scipy = ">=1.13"
trimesh = ">=4.4"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| ARCTIC eval sequences | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/arctic_bicair/` | challenge download |
| HO3D eval data | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/HO3D_v3/` | official dataset |
| Objaverse embeddings | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/objaverse/` | official dataset |

## Test Plan
```bash
uv run pytest tests/test_alignment.py tests/test_losses.py tests/test_rigging.py -v
```

## References
- Paper: §3.1, §3.2, §3.3, Supp. A, Supp. B
- Reference impl: `repositories/GHOST/preprocess/*.py`, `repositories/GHOST/train.py`, `repositories/GHOST/scene/gaussian_model_mano.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04

