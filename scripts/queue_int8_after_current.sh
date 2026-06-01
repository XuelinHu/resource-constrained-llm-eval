#!/usr/bin/env bash
set -euo pipefail

# Queue an int8 paper experiment chain after the currently running int4 chain.
# This intentionally waits instead of running concurrently, preserving GPU
# latency/throughput measurements for the active experiment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${1:-}"
if [[ -z "${CURRENT_PID}" ]]; then
  CURRENT_PID="$(pgrep -f 'bash scripts/run_paper_experiment_chain.sh' | head -n 1 || true)"
fi
if [[ -z "${CURRENT_PID}" ]]; then
  echo "No current paper experiment chain PID found." >&2
  exit 1
fi

cd "${ROOT_DIR}"

run_id="$(date +%Y%m%d_%H%M%S)"
log="logs/paper_experiment_chain_int8_queued_${run_id}.log"
pidfile="logs/paper_experiment_chain_int8_queued_${run_id}.pid"

{
  echo "[$(date '+%F %T')] Waiting for current chain PID ${CURRENT_PID}"
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 300
  done
  echo "[$(date '+%F %T')] Current chain finished; starting int8 chain"
  exec env \
    PYTHONUNBUFFERED=1 \
    PRECISION=int8 \
    BASELINE_GROUP=paper_baseline_int8 \
    BASELINE_LABEL=public_domain_regqa \
    ADAPTER_GROUP=paper_adapter_domainqa_int8 \
    ADAPTER_LABEL=domainqa_adapter_public_domain_regqa \
    bash scripts/run_paper_experiment_chain.sh
} > "${log}" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${pidfile}"
printf 'queued_pid=%s\nlog=%s\npidfile=%s\nwaiting_for=%s\n' "${pid}" "${log}" "${pidfile}" "${CURRENT_PID}"
