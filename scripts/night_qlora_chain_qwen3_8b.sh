#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${1:-2294652}"
CURRENT_MODEL="${CURRENT_MODEL:-qwen2_5_7b_instruct}"
NEXT_MODEL="${NEXT_MODEL:-qwen3_8b}"
DATASET="${DATASET:-domain_qa}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/single_gpu_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
CURRENT_LOG="${CURRENT_LOG:-logs/qlora_qwen2_5_7b_instruct_full_20260531_182209.log}"

cd "${ROOT_DIR}"
mkdir -p logs

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

process_is_current_training() {
  local cmd
  cmd="$(ps -p "${CURRENT_PID}" -o cmd= 2>/dev/null || true)"
  [[ "${cmd}" == *"run-qlora"* && "${cmd}" == *"--model ${CURRENT_MODEL}"* ]]
}

log "Watching current QLoRA run: pid=${CURRENT_PID}, model=${CURRENT_MODEL}"
while process_is_current_training; do
  sleep 120
done

log "Current training process is no longer running; waiting for final files to settle."
sleep 20

log "Running sanity check for ${CURRENT_MODEL}."
"${PYTHON_BIN}" - "${ROOT_DIR}" "${CURRENT_MODEL}" "${CURRENT_LOG}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
model = sys.argv[2]
log_path = root / sys.argv[3]
run_dir = root / "results" / "single_gpu_3090" / "qlora" / model
adapter = run_dir / "adapter" / "adapter_model.safetensors"
train_metrics_path = run_dir / "train_metrics.json"
eval_metrics_path = run_dir / "eval_metrics.json"

missing = [str(p) for p in (adapter, train_metrics_path, eval_metrics_path) if not p.exists()]
if missing:
    raise SystemExit("missing required output files: " + ", ".join(missing))

train_metrics = json.loads(train_metrics_path.read_text())
eval_metrics = json.loads(eval_metrics_path.read_text())
train_loss = float(train_metrics.get("train_loss", float("nan")))
eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
eval_accuracy = eval_metrics.get("eval_accuracy")
eval_accuracy = float(eval_accuracy) if eval_accuracy is not None else None

if not math.isfinite(train_loss) or not math.isfinite(eval_loss):
    raise SystemExit(f"non-finite loss: train_loss={train_loss}, eval_loss={eval_loss}")
if train_loss >= 1.0:
    raise SystemExit(f"train_loss too high for a completed domain run: {train_loss}")
if eval_loss >= 1.0:
    raise SystemExit(f"eval_loss too high for a completed domain run: {eval_loss}")
if eval_accuracy is not None and eval_accuracy < 0.75:
    raise SystemExit(f"eval_accuracy too low for a completed domain run: {eval_accuracy}")

if log_path.exists():
    text = log_path.read_text(errors="ignore")
    bad = re.search(r"Traceback|CUDA out of memory|RuntimeError|ValueError|\bnan\b", text, re.IGNORECASE)
    if bad:
        raise SystemExit(f"current log contains error marker: {bad.group(0)}")

print(
    "sanity ok: "
    f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, "
    f"eval_accuracy={eval_accuracy if eval_accuracy is not None else 'n/a'}"
)
PY

if ps -eo cmd | grep -F "run-qlora" | grep -v grep >/dev/null; then
  log "Another run-qlora process is active; not starting ${NEXT_MODEL}."
  exit 3
fi

ts="$(date +%Y%m%d_%H%M%S)"
next_dir="results/single_gpu_3090/qlora/${NEXT_MODEL}"
next_log="logs/qlora_${NEXT_MODEL}_full_${ts}.log"

if [[ -d "${next_dir}" ]]; then
  backup_dir="${next_dir}_backup_${ts}"
  log "Existing ${NEXT_MODEL} output directory found; moving it to ${backup_dir}."
  mv "${next_dir}" "${backup_dir}"
fi

log "Starting next full QLoRA run: model=${NEXT_MODEL}, dataset=${DATASET}, log=${next_log}"
{
  echo "[$(timestamp)] starting full qlora: ${NEXT_MODEL} ${DATASET}"
  echo "estimated runtime: about 7-8 hours on RTX 3090 based on qwen2.5-7b throughput and qwen3-8b int4 latency"
} > "${next_log}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"${PYTHON_BIN}" -u -m src.rc_llm_eval.cli run-qlora \
  --experiment "${EXPERIMENT}" \
  --model "${NEXT_MODEL}" \
  --dataset "${DATASET}" >> "${next_log}" 2>&1 &
next_pid=$!
log "Started ${NEXT_MODEL}: pid=${next_pid}, log=${next_log}"
wait "${next_pid}"
status=$?
log "${NEXT_MODEL} finished with status=${status}"
exit "${status}"
