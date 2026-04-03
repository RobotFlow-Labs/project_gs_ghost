# GS-GHOST

## Paper
**GHOST: Hand-Object Reconstruction via 3DGS**
arXiv: https://arxiv.org/abs/2503.14397

## Module Identity
- Codename: GS-GHOST
- Domain: 3DGS
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_gs_ghost/
├── pyproject.toml
├── configs/
├── src/anima_gs_ghost/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── CLAUDE.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [GS-GHOST]
