#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_MODE="${FRONTEND_MODE:-frp}"

cd "$ROOT_DIR/annotation_system/frontend"
exec npm run "dev:${FRONTEND_MODE}" -- --host 0.0.0.0 --port "${FRONTEND_PORT:-4005}"
