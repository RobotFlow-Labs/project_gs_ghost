"""Tests for ROS2 contracts and node — PRD-06."""


class TestContracts:
    def test_request_has_sequence_path(self):
        from anima_gs_ghost.ros2.messages import GhostRequest
        req = GhostRequest(sequence_path="/data/seq1")
        assert req.sequence_path == "/data/seq1"
        assert req.sfm_method == "vggsfm"

    def test_status_default_unknown(self):
        from anima_gs_ghost.ros2.messages import GhostStatus
        s = GhostStatus()
        assert s.status == "unknown"

    def test_result_has_metrics(self):
        from anima_gs_ghost.ros2.messages import GhostResult
        r = GhostResult(job_id="abc", metrics={"PSNR": 25.0})
        assert r.metrics["PSNR"] == 25.0


class TestNode:
    def test_node_creation(self):
        from anima_gs_ghost.ros2.node import GhostBatchNode
        node = GhostBatchNode()
        assert node is not None

    def test_dispatch_returns_status(self):
        from anima_gs_ghost.ros2.messages import GhostRequest
        from anima_gs_ghost.ros2.node import GhostBatchNode
        node = GhostBatchNode()
        req = GhostRequest(sequence_path="/tmp/test", sequence_name="test")
        status = node.handle_reconstruct(req)
        # Status may be queued or running (race with background thread)
        assert status.status in ("queued", "running")
        assert status.job_id != ""

    def test_get_status(self):
        from anima_gs_ghost.ros2.messages import GhostRequest
        from anima_gs_ghost.ros2.node import GhostBatchNode
        node = GhostBatchNode()
        req = GhostRequest(sequence_path="/tmp/test")
        status = node.handle_reconstruct(req)
        retrieved = node.get_status(status.job_id)
        assert retrieved is not None


class TestLaunch:
    def test_launch_file_exists(self):
        from pathlib import Path
        launch_file = Path(__file__).resolve().parents[1] / "launch" / "gs_ghost.launch.py"
        assert launch_file.exists()
