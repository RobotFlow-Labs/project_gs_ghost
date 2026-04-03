# FUJIN — Design & Implementation Checklist

## Paper: GHOST: Hand-Object Reconstruction via 3DGS
## ArXiv: 2503.14397
## Repo: https://github.com/ATAboukhadra/GHOST

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [ ] Paper PDF downloaded to papers/
- [ ] Paper read and annotated
- [ ] Reference repo cloned
- [ ] Reference demo runs successfully
- [ ] Datasets identified and accessibility confirmed
- [ ] CLAUDE.md filled with paper-specific details
- [ ] PRD.md filled with architecture and plan

## Phase 2: Reproduce
- [ ] Core model implemented in src/anima_fujin/
- [ ] Training pipeline (scripts/train.py)
- [ ] Evaluation pipeline (scripts/eval.py)
- [ ] Metrics match paper (within ±5%)
- [ ] Dual-compute verified (MLX + CUDA)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [ ] API endpoints defined
- [ ] Integration test with stack: PROMETHEUS

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
