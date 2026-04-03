"""Placeholder service entrypoint for ANIMA infra checks."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps({"status": "stub", "service": "anima-gs-ghost"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    HTTPServer(("0.0.0.0", 8000), _Handler).serve_forever()


if __name__ == "__main__":
    main()

