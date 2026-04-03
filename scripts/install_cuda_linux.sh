#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CUDA install path is Linux-only. Run the standard macOS bootstrap locally." >&2
  exit 1
fi

PYTHON_BIN="${1:-python3.11}"

echo "[GS-GHOST] creating uv environment with ${PYTHON_BIN}"
uv venv .venv --python "${PYTHON_BIN}"

echo "[GS-GHOST] syncing base project + paper + serve extras"
uv sync --group dev --extra paper --extra serve --extra cuda

echo "[GS-GHOST] installing paper-faithful CUDA source builds"
uv pip install --python .venv/bin/python --no-build-isolation -r requirements/cuda-linux.txt

echo "[GS-GHOST] CUDA environment ready"
