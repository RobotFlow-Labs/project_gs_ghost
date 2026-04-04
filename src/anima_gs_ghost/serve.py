"""ANIMA serve entrypoint — FastAPI + optional ROS2 node."""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Start the GS-GHOST service."""
    try:
        import uvicorn
        from .api.app import app

        if app is None:
            raise ImportError("FastAPI not available")

        logger.info("Starting GS-GHOST service on port 8080")
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    except ImportError:
        logger.warning("FastAPI/uvicorn not installed — running health stub")
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = json.dumps(
                    {"status": "stub", "module": "anima-gs-ghost", "version": "0.2.0"}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        HTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()


if __name__ == "__main__":
    main()
