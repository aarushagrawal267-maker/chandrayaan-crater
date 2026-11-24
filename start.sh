#!/usr/bin/env bash
# One-command setup + launch (macOS / Linux). Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import torch, cv2, fastapi, multipart" 2>/dev/null; then
  echo "Installing dependencies (first run only, ~250 MB)..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi
PORT=${PORT:-8000}
echo "Open http://localhost:${PORT} in your browser (Ctrl+C to stop)."
exec python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
