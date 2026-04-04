"""ROS2 node for GS-GHOST batch reconstruction.

Subscribes to reconstruction requests and publishes status/results.
Falls back to a standalone loop if rclpy is not available.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from .messages import GhostRequest, GhostStatus

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node as RclpyNode
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    RclpyNode = object


class GhostBatchNode(RclpyNode if HAS_ROS2 else object):
    """ROS2 node for dispatching GHOST reconstruction jobs.

    If rclpy is not available, runs as a standalone Python service.
    """

    def __init__(self, node_name: str = "gs_ghost_node") -> None:
        if HAS_ROS2:
            super().__init__(node_name)
            logger.info("ROS2 node '%s' initialised", node_name)
        else:
            logger.info("ROS2 not available — running in standalone mode")

        self._jobs: dict[str, GhostStatus] = {}
        self._lock = threading.Lock()

    def handle_reconstruct(self, req: GhostRequest) -> GhostStatus:
        """Handle a reconstruction request.

        Dispatches the pipeline in a background thread and returns
        initial status immediately.
        """
        job_id = str(uuid.uuid4())[:8]
        status = GhostStatus(
            job_id=job_id,
            sequence=req.sequence_name or Path(req.sequence_path).stem,
            status="queued",
        )
        with self._lock:
            self._jobs[job_id] = status

        thread = threading.Thread(
            target=self._run_reconstruction,
            args=(job_id, req),
            daemon=True,
        )
        thread.start()
        return status

    def _run_reconstruction(self, job_id: str, req: GhostRequest) -> None:
        """Execute reconstruction pipeline in background."""
        with self._lock:
            self._jobs[job_id].status = "running"

        try:
            from ..config import GhostSettings
            from ..pipeline import run_full_pipeline

            cfg = GhostSettings()
            manifest = run_full_pipeline(
                cfg,
                Path(req.sequence_path),
                req.sequence_name or Path(req.sequence_path).stem,
                req.device,
            )
            with self._lock:
                self._jobs[job_id].status = "done"
                self._jobs[job_id].manifest_path = str(manifest)
                self._jobs[job_id].progress = 1.0
        except Exception as e:
            with self._lock:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].error = str(e)

    def get_status(self, job_id: str) -> GhostStatus | None:
        with self._lock:
            return self._jobs.get(job_id)


def main() -> None:
    """Entrypoint for the ROS2 node."""
    if HAS_ROS2:
        rclpy.init()
        node = GhostBatchNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        logger.warning("ROS2 not available. Node running in standalone mode.")
        node = GhostBatchNode()
        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
