#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://127.0.0.1:1234/v1}"
LMSTUDIO_PORT="${LMSTUDIO_PORT:-1234}"

if ! curl -fsS --max-time 3 "$LMSTUDIO_BASE_URL/models" >/dev/null 2>&1; then
  "$LMS_BIN" server start --port "$LMSTUDIO_PORT"
fi

exec "$ROOT_DIR/.venv/bin/python" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
