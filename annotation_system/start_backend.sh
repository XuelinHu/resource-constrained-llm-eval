#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"

cd "$ROOT_DIR/annotation_system/backend"
exec "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT:-8000}"
