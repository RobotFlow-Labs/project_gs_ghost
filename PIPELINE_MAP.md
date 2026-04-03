# GS-GHOST — Pipeline Map

## Paper Pipeline → PRD Mapping

| Paper Component | Paper Section | Reference Repo File | Planned ANIMA File | PRD |
|----------------|---------------|---------------------|--------------------|-----|
| SAM2 object masks `M_o^t` | §3.1 | `repositories/GHOST/preprocess/sam_object.py` | `src/anima_gs_ghost/preprocess/object_masks.py` | PRD-02 |
| SfM intrinsics, poses, sparse cloud `K,(R_c,T_c),P_sfm` | §3.1 | `repositories/GHOST/preprocess/hloc_colmap_sfm.py`, `repositories/GHOST/preprocess/vggsfm_video.py` | `src/anima_gs_ghost/preprocess/sfm.py` | PRD-02 |
| Prior retrieval from text prompt | §3.1.1 | `repositories/GHOST/preprocess/retrieve_prior.py`, `repositories/GHOST/preprocess/internvl.py` | `src/anima_gs_ghost/preprocess/prior_retrieval.py` | PRD-02 |
| Prior-mask affine alignment `(R_p,T_p,S_p)` via IoU | Eq. (1), §3.1.1 | `repositories/GHOST/preprocess/align_prior.py` | `src/anima_gs_ghost/alignment/prior_alignment.py` | PRD-02 |
| HaMeR-based hand initialization and jitter cleanup | §3.1.2, Supp. A | `repositories/GHOST/preprocess/hamer_video.py`, `repositories/GHOST/preprocess/utils/hamer_utils.py` | `src/anima_gs_ghost/preprocess/hand_init.py` | PRD-02 |
| Grasp detection via motion cosine similarity | Eq. (2), §3.2.1 | `repositories/GHOST/preprocess/utils/optim_utils.py` | `src/anima_gs_ghost/alignment/grasp_detection.py` | PRD-02 |
| HO alignment losses `L_contact`, `L_proj`, `L_temp` | Eq. (3-6), §3.2.2 | `repositories/GHOST/preprocess/optim_scale_transl.py` | `src/anima_gs_ghost/alignment/ho_alignment.py` | PRD-02 |
| Object Gaussian optimization with `L_rgb`, `L_bkg,h`, `L_geo` | §3.3.2, Eq. (8-9) | `repositories/GHOST/train.py`, `repositories/GHOST/utils/loss_utils.py` | `src/anima_gs_ghost/reconstruction/object_gs.py`, `src/anima_gs_ghost/reconstruction/losses.py` | PRD-02 |
| Canonical hand Gaussians and affine rigging | §3.3.3, Eq. (10-12) | `repositories/GHOST/preprocess/animate_hand_gaussian.py`, `repositories/GHOST/scene/gaussian_model_mano.py` | `src/anima_gs_ghost/reconstruction/hand_gs.py` | PRD-02 |
| End-to-end single-sequence runner | Fig. 2, §3 | `repositories/GHOST/preprocess/run_single_sequence.sh`, `repositories/GHOST/scripts/train_object.bash`, `repositories/GHOST/scripts/train_combined.bash` | `scripts/run_sequence.py` | PRD-03 |
| 2D rendering evaluation | §4.2, Table 3 | `repositories/GHOST/evaluate.py` | `src/anima_gs_ghost/eval/rendering.py`, `scripts/evaluate_rendering.py` | PRD-04 |
| ARCTIC Bi-CAIR export/eval bridge | §4.2, §4.4 | `repositories/GHOST/scene/gaussian_model_mano.py` | `src/anima_gs_ghost/eval/arctic_export.py` | PRD-04 |
| Interactive hand-avatar viewer | Supp. Fig. 13 | `repositories/GHOST/viewer_mano.py` | `src/anima_gs_ghost/viewer.py` | PRD-03 / PRD-07 |
| Service wrapper for batch inference | motivated by §5 deployment targets | none upstream | `src/anima_gs_ghost/api/app.py` | PRD-05 |
| ROS2 bridge for ANIMA compiler | project adaptation | none upstream | `src/anima_gs_ghost/ros2/node.py` | PRD-06 |

## Data Flow

`video.mp4`
→ sampled RGB frames `I_t`
→ SAM2 object masks + hand masks
→ SfM intrinsics / extrinsics / sparse point cloud
→ text prompt + OpenShape retrieval
→ affine prior alignment `(R_p,T_p,S_p)`
→ HaMeR hands + jitter filtering
→ grasp detection + HO alignment
→ object-only Gaussian optimization
→ hand-object joint Gaussian optimization
→ rendering / viewer / ARCTIC export / evaluation reports

## Runtime Artifact Layout

| Artifact | Upstream Layout | Planned ANIMA Layout |
|---------|-----------------|----------------------|
| preprocessing outputs | `data/<seq>/ghost_build/` | `artifacts/<seq>/ghost_build/` |
| object stage outputs | `data/<seq>/output/object/` | `artifacts/<seq>/output/object/` |
| combined stage outputs | `data/<seq>/output/combined/` | `artifacts/<seq>/output/combined/` |
| render metrics | `metrics_summary_<exp>.json` | `reports/<seq>/metrics_summary_<exp>.json` |
| ARCTIC submission tensor | `<seq>.pt` | `reports/arctic/<seq>.pt` |

