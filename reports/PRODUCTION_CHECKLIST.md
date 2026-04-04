# GS-GHOST Production Checklist

## Paper Fidelity
- [ ] Correct paper: arXiv 2603.18912 (GHOST)
- [ ] All loss terms implemented: L_rgb, L_bkg_h, L_geo, L_contact, L_proj, L_temp
- [ ] Hyperparameters match paper + supplementary
- [ ] Pipeline structure matches Fig. 2

## Benchmarks
- [ ] ARCTIC Bi-CAIR: CDh <= 20 cm^2
- [ ] ARCTIC rendering: PSNR >= 25.0
- [ ] HO3D rendering: LPIPS <= 0.04
- [ ] Runtime <= 75 min per 300-frame sequence

## Training
- [ ] Checkpoint save/load verified
- [ ] Early stopping functional
- [ ] GPU VRAM 60-80% utilised
- [ ] Training launched with nohup+disown

## Exports
- [ ] pth checkpoint
- [ ] safetensors
- [ ] ONNX
- [ ] TensorRT FP16
- [ ] TensorRT FP32

## ANIMA Infrastructure
- [ ] anima_module.yaml complete
- [ ] Dockerfile.serve builds
- [ ] docker-compose.serve.yml works (profiles: serve, ros2, api, test)
- [ ] /health endpoint responds
- [ ] /ready endpoint responds
- [ ] ROS2 node imports cleanly

## HuggingFace
- [ ] Pushed to ilessio-aiflowlab/project_gs_ghost
- [ ] Model card present
- [ ] Training report attached
- [ ] Config included

## Git
- [ ] All changes committed with [GS-GHOST] prefix
- [ ] Tests pass: uv run pytest tests/ -v
- [ ] Lint clean: uv run ruff check src/ tests/
