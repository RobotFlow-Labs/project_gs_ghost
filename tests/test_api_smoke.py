"""Smoke tests for API and service — PRD-05."""

import pytest


class TestAPIModels:
    def test_health_response(self):
        from anima_gs_ghost.api.models import HealthResponse
        h = HealthResponse()
        assert h.status == "ok"
        assert h.module == "anima-gs-ghost"

    def test_sequence_request(self):
        from anima_gs_ghost.api.models import SequenceRequest
        req = SequenceRequest(sequence_path="/data/my_seq")
        assert req.sfm_method == "vggsfm"
        assert req.device == "cuda:1"

    def test_job_status(self):
        from anima_gs_ghost.api.models import JobStatus
        job = JobStatus(job_id="abc123")
        assert job.status == "queued"
        assert job.progress == 0.0


class TestAPIApp:
    def test_app_creation(self):
        try:
            from anima_gs_ghost.api.app import create_app
            app = create_app()
            assert app is not None
            assert app.title == "GS-GHOST"
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_health_endpoint(self):
        try:
            from fastapi.testclient import TestClient
            from anima_gs_ghost.api.app import create_app
            client = TestClient(create_app())
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["module"] == "anima-gs-ghost"
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_ready_endpoint(self):
        try:
            from fastapi.testclient import TestClient
            from anima_gs_ghost.api.app import create_app
            client = TestClient(create_app())
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json()["ready"] is True
        except ImportError:
            pytest.skip("fastapi not installed")


class TestServeModule:
    def test_serve_module_importable(self):
        # Just verify it imports without error
        from anima_gs_ghost import serve  # noqa: F401
