"""FastAPI application for GS-GHOST batch reconstruction."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from .models import HealthResponse, InfoResponse, JobStatus, SequenceRequest

try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    FastAPI = None

# Job store (in-memory for single-instance deployment)
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def create_app() -> "FastAPI":
    if FastAPI is None:
        raise ImportError("fastapi not installed — run: uv pip install fastapi uvicorn")

    app = FastAPI(title="GS-GHOST", version="0.2.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        gpu_available = False
        gpu_name = None
        gpu_vram_mb = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_mb = int(torch.cuda.get_device_properties(0).total_memory / 1e6)
        except ImportError:
            pass
        return HealthResponse(
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_mb=gpu_vram_mb,
        )

    @app.get("/ready")
    def ready() -> dict:
        return {"ready": True, "module": "anima-gs-ghost", "version": "0.2.0"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        return InfoResponse()

    @app.post("/reconstruct", response_model=JobStatus)
    def queue_reconstruction(req: SequenceRequest) -> JobStatus:
        job_id = str(uuid.uuid4())[:8]
        job = JobStatus(
            job_id=job_id,
            status="queued",
            sequence=req.sequence_name or Path(req.sequence_path).stem,
        )
        with _lock:
            _jobs[job_id] = job

        # Launch in background thread
        thread = threading.Thread(
            target=_run_job, args=(job_id, req), daemon=True
        )
        thread.start()
        return job

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str) -> JobStatus:
        with _lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            return _jobs[job_id]

    @app.get("/jobs", response_model=list[JobStatus])
    def list_jobs() -> list[JobStatus]:
        with _lock:
            return list(_jobs.values())

    return app


def _run_job(job_id: str, req: SequenceRequest) -> None:
    """Execute reconstruction in background thread."""
    with _lock:
        _jobs[job_id].status = "running"
        _jobs[job_id].stage = "initialising"

    try:
        from ..config import GhostSettings
        from ..pipeline import run_full_pipeline

        cfg = GhostSettings()
        frames_dir = Path(req.sequence_path)
        sequence = req.sequence_name or frames_dir.stem

        with _lock:
            _jobs[job_id].stage = "preprocessing"
            _jobs[job_id].progress = 0.1

        manifest_path = run_full_pipeline(cfg, frames_dir, sequence, req.device)

        with _lock:
            _jobs[job_id].status = "done"
            _jobs[job_id].progress = 1.0
            _jobs[job_id].stage = "complete"
            _jobs[job_id].manifest_path = str(manifest_path)

    except Exception as e:
        with _lock:
            _jobs[job_id].status = "failed"
            _jobs[job_id].error = str(e)


# Module-level app instance
app = create_app() if FastAPI is not None else None
