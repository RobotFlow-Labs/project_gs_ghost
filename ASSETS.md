# GS-GHOST — Asset Manifest

## Paper
- Title: GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting
- Canonical arXiv: 2603.18912
- Legacy project reference: 2503.14397 in local scaffolding is incorrect and points to an unrelated paper
- Authors: Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, Didier Stricker

## Status: ALMOST

The correct paper PDF is now stored at [papers/2603.18912_GHOST.pdf](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/papers/2603.18912_GHOST.pdf). The older [papers/2503.14397_GHOST.pdf](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_gs_ghost/papers/2503.14397_GHOST.pdf) is mislabeled and should not be used for planning or implementation.

## Pretrained Weights
| Model | Size | Source | Path | Status |
|-------|------|--------|------|--------|
| HaMeR checkpoint | ~400 MB | `gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT` | `repositories/GHOST/preprocess/hamer_demo_data.tar.gz` | MISSING |
| MANO hand model bundle | gated download | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) | `repositories/GHOST/preprocess/_DATA/data/mano/` | MISSING |
| SAM2.1 Hiera Large | HF cached model | [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large) | local Hugging Face cache | MISSING |
| OpenShape retrieval stack | env + checkpoints | `pip install -e submodules/openshape` in optional `openshape` env | optional external env | MISSING |
| InternVL VLM | model-dependent | see `repositories/GHOST/preprocess/internvl.py` | optional external env | OPTIONAL |
| PyTorch3D | source build | `git+https://github.com/facebookresearch/pytorch3d.git` | active env | MISSING |
| nvdiffrast | source build | `git+https://github.com/NVlabs/nvdiffrast` | active env | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---------|------|-------|--------|------|--------|
| ARCTIC Bi-CAIR allocentric sequences | 9 eval sequences | eval | [ARCTIC leaderboard](https://arctic-leaderboard.is.tuebingen.mpg.de/leaderboard) and HOLD download script | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/arctic_bicair/` | MISSING |
| HO3D v3 | selected eval subset | eval | [HO3D](https://arxiv.org/abs/2004.00818) | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/HO3D_v3/` | MISSING |
| Objaverse / Objaverse-XL retrieval meshes | large-scale asset db | retrieval corpus | [Objaverse](https://objaverse.allenai.org/) | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/objaverse/` | MISSING |
| In-the-wild qualitative sequences | 2 sequences | qualitative | locally captured GoPro videos (`Drill`, `Book`) | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/gs_ghost_in_the_wild/` | MISSING |

## ARCTIC Eval Sequence Targets
- `arctic_s03_box_grab_01_1`
- `arctic_s03_notebook_grab_01_1`
- `arctic_s03_laptop_grab_01_1`
- `arctic_s03_ketchup_grab_01_1`
- `arctic_s03_espressomachine_grab_01_1`
- `arctic_s03_microwave_grab_01_1`
- `arctic_s03_waffleiron_grab_01_1`
- `arctic_s03_mixer_grab_01_1`
- `arctic_s03_capsulemachine_grab_01_1`

## Hyperparameters (from paper + supplementary + reference scripts)
| Param | Value | Paper Section |
|-------|-------|---------------|
| prior_alignment_optimizer | AdamW | Supp. B.1 |
| prior_alignment_lr | `1e-2` | Supp. B.1 |
| prior_alignment_betas | `(0.9, 0.99)` | Supp. B.1 |
| prior_alignment_iters | `1500` | Supp. B.1 |
| ho_alignment_optimizer | Adam | Supp. B.2 |
| ho_alignment_lr | `0.05` | Supp. B.2 |
| ho_alignment_iters | `500` | Supp. B.2 |
| lambda_contact | `1e3` | §3.2.2 |
| lambda_proj | `1e-1` | §3.2.2 |
| lambda_temp | `10` | §3.2.2 |
| object_gs_optimizer | Adam | Supp. B.3 |
| object_gs_iters | `30000` | Supp. B.3 |
| combined_gs_optimizer | Adam | Supp. B.4 |
| combined_gs_iters | `30000` | Supp. B.4 |
| sh_degree | `3` | `repositories/GHOST/scripts/train_object.bash` |
| densify_until_iter | `15000` | `repositories/GHOST/scripts/train_object.bash` |
| lambda_background | `0.3` object stage | `repositories/GHOST/scripts/train_object.bash` |
| gaussians_per_edge | `10` for combined hand stage | `repositories/GHOST/scripts/train_combined.bash` |
| optimize_mano | `true` during combined stage | `repositories/GHOST/scripts/train_combined.bash` |
| transl_lr | `1e-4` | `repositories/GHOST/arguments/__init__.py` |
| lambda_geo | `5.0` default | §3.3.2 and `repositories/GHOST/arguments/__init__.py` |
| tau_out | `0.05` m default, ablated at `0.02` and `0.07` | §3.3.2, Table 1 |
| tau_fill | `0.005` m default, ablated at `1.0/0.1/0.01` mm in Table 1 | §3.3.2, Table 1 |
| tau_sim | `0.5` | §3.2.1 |
| hand_reject_thresholds | `tau_p=1.0`, `tau_o=1.0`, `tau_t=2.0`, `tau_s=4.0`, `tau_c=0.3`, `tau_iou=0.3`, `Amin=0.006*Aimg`, `Amax=0.2*Aimg` | Supp. A |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|-----------|--------|-------------|-----------|
| ARCTIC | `MPJPE_RA,h` | `24.07 mm` | `<= 25 mm` |
| ARCTIC | `MPJPE_RA,r` | `22.71 mm` | `<= 24 mm` |
| ARCTIC | `MPJPE_RA,l` | `25.42 mm` | `<= 26 mm` |
| ARCTIC | `CDICP` | `2.26 cm^2` | `<= 2.5 cm^2` |
| ARCTIC | `CDr` | `13.40 cm^2` | `<= 14.5 cm^2` |
| ARCTIC | `CDl` | `23.41 cm^2` | `<= 25 cm^2` |
| ARCTIC | `CDh` | `18.40 cm^2` | `<= 20 cm^2` |
| ARCTIC | `F10mm` | `60.88%` | `>= 58%` |
| ARCTIC | `F5mm` | `34.67%` | `>= 32%` |
| ARCTIC rendering | `PSNR` | `25.93` | `>= 25.0` |
| ARCTIC rendering | `SSIM` | `0.88` | `>= 0.86` |
| ARCTIC rendering | `LPIPS` | `0.02` | `<= 0.03` |
| HO3D rendering | `PSNR` | `21.37` | `>= 20.5` |
| HO3D rendering | `SSIM` | `0.75` | `>= 0.73` |
| HO3D rendering | `LPIPS` | `0.03` | `<= 0.04` |
| Runtime | end-to-end optimization | `1h` on RTX A6000 for 300 ARCTIC frames | `<= 75 min` on comparable GPU |

