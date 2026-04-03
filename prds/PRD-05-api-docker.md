# PRD-05: API & Docker

> Module: GS-GHOST | Priority: P1  
> Depends on: PRD-03  
> Status: ⬜ Not started

## Objective
GS-GHOST can run as a batch inference service inside a reproducible container with explicit job status and artifact retrieval.

## Context (from paper)
The paper is offline and optimization-heavy, but its target domains include AR/VR, robotics, and embodied AI, so ANIMA needs a service shell around the faithful core.  
**Paper reference**: §5 "Conclusion"  
Key line: "These extensions can enable practical deployment in teleoperation, interactive AR/VR systems, robotics manipulation..."

## Acceptance Criteria
- [ ] FastAPI app exposes `/health`, `/infer/sequence`, `/jobs/{id}`, `/artifacts/{id}`
- [ ] Background worker runs the same pipeline as `scripts/run_sequence.py`
- [ ] Docker image documents CUDA and model-volume requirements
- [ ] Smoke test: container boots and returns healthy status
- [ ] Test: `uv run pytest tests/test_api_smoke.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_gs_ghost/api/app.py` | FastAPI service | §5 adaptation | ~180 |
| `src/anima_gs_ghost/api/models.py` | request/response schemas | — | ~80 |
| `docker/Dockerfile` | service image | — | ~80 |
| `docker/docker-compose.yml` | mounted service runner | — | ~60 |
| `tests/test_api_smoke.py` | health route smoke test | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- uploaded video or mounted frame directory
- optional prompt and SFM mode

### Outputs
- job id
- artifact URLs or file paths

### Algorithm
```python
@app.post("/infer/sequence")
def infer_sequence(req: SequenceRequest) -> JobResponse:
    job_id = queue_sequence(req)
    return JobResponse(job_id=job_id, status="queued")
```

## Dependencies
```toml
fastapi = ">=0.115"
uvicorn = ">=0.30"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| model/data mounts | variable | `/models`, `/datasets`, `/artifacts` | mounted at runtime |

## Test Plan
```bash
uv run pytest tests/test_api_smoke.py -v
```

## References
- Paper: §5
- Depends on: PRD-03
- Feeds into: PRD-06, PRD-07

