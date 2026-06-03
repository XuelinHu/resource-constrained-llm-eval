#!/usr/bin/env bash
set -euo pipefail

# Queue precision follow-up experiments after the active int4 paper chain.
# The follow-up matrix is intentionally smaller than the full int4 baseline:
# - int8: representative subset for precision/efficiency comparison
# - bf16: feasibility boundary subset on a 24 GB RTX 3090

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-configs/experiments/single_gpu_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
CURRENT_PID="${1:-${CURRENT_PID:-}}"

if [[ "${DETACHED:-0}" != "1" ]]; then
  if [[ -z "${CURRENT_PID}" ]]; then
    CURRENT_PID="$(pgrep -f 'bash scripts/run_paper_experiment_chain.sh' | head -n 1 || true)"
  fi

  run_id="$(date +%Y%m%d_%H%M%S)"
  log="logs/precision_followups_queued_${run_id}.log"
  pidfile="logs/precision_followups_queued_${run_id}.pid"

  cd "${ROOT_DIR}"
  setsid env DETACHED=1 CURRENT_PID="${CURRENT_PID}" EXPERIMENT="${EXPERIMENT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${BASH_SOURCE[0]}" "${CURRENT_PID}" > "${log}" 2>&1 < /dev/null &

  pid=$!
  printf '%s\n' "${pid}" > "${pidfile}"
  printf 'queued_pid=%s\nlog=%s\npidfile=%s\nwaiting_for=%s\n' "${pid}" "${log}" "${pidfile}" "${CURRENT_PID:-none}"
  exit 0
fi

cd "${ROOT_DIR}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1

notify() {
  local title="$1"
  local status="$2"
  local detail="${3:-}"
  "${PYTHON_BIN}" - "$title" "$status" "$detail" <<'PY'
from datetime import datetime
import sys

from src.rc_llm_eval.utils.notifications import build_markdown_message, send_dingtalk_notification

title, status, detail = sys.argv[1:4]
fields = [
    ("状态", status),
    ("时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
]
if detail:
    fields.append(("说明", detail))
send_dingtalk_notification(build_markdown_message(title, fields), err=status != "完成")
PY
}

output_root="$("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print(configs["experiment"]["experiment"]["output_root"])
PY
)"

run_eval_one() {
  local precision="$1"
  local output_group="$2"
  local label="$3"
  local model="$4"
  local file_stem="${model}_${precision}_${label}"
  local summary_path="${output_root}/${output_group}/${model}/${file_stem}_summary.json"
  local lm_eval_path="${output_root}/${output_group}/${model}/${file_stem}_lm_eval.json"

  if [[ "${FORCE:-0}" != "1" && -f "${summary_path}" && -f "${lm_eval_path}" ]]; then
    echo "[$(date '+%F %T')] Skip completed ${precision}: ${model}"
    return 0
  fi

  echo "[$(date '+%F %T')] Run ${precision}: ${model}"
  if "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli run-eval \
      --experiment "${EXPERIMENT}" \
      --model "${model}" \
      --precision "${precision}" \
      --output-group "${output_group}" \
      --label "${label}"; then
    notify "论文后续精度实验完成" "完成" "${precision} ${model}"
    return 0
  fi

  local rc=$?
  echo "[$(date '+%F %T')] Failed ${precision}: ${model} rc=${rc}"
  notify "论文后续精度实验失败" "失败" "${precision} ${model} rc=${rc}"
  return 0
}

summarize_group() {
  local output_group="$1"
  if "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli summarize-results \
      --experiment "${EXPERIMENT}" \
      --output-group "${output_group}"; then
    notify "论文后续精度汇总完成" "完成" "${output_group}"
  else
    notify "论文后续精度汇总失败" "失败" "${output_group}"
  fi
}

INT8_MODELS=(
  qwen3_4b
  qwen2_5_7b_instruct
  glm_4_9b_chat_hf
)

BF16_MODELS=(
  phi_3_mini_4k_instruct
  qwen3_4b
  gemma_3_4b
  qwen2_5_7b_instruct
)

echo "[$(date '+%F %T')] Precision follow-up queue start"
echo "Experiment: ${EXPERIMENT}"
echo "Output root: ${output_root}"
echo "Waiting for current chain PID: ${CURRENT_PID:-none}"
notify "论文后续精度实验队列启动" "开始" "wait=${CURRENT_PID:-none}; int8=${INT8_MODELS[*]}; bf16=${BF16_MODELS[*]}"

if [[ -n "${CURRENT_PID:-}" ]]; then
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 300
  done
fi

echo "[$(date '+%F %T')] Current chain finished; starting int8 representative subset"
for model in "${INT8_MODELS[@]}"; do
  run_eval_one "int8" "paper_baseline_int8_subset" "public_domain_regqa" "${model}"
done
summarize_group "paper_baseline_int8_subset"

echo "[$(date '+%F %T')] Starting bf16 feasibility boundary subset"
for model in "${BF16_MODELS[@]}"; do
  run_eval_one "bf16" "paper_baseline_bf16_boundary" "public_domain_regqa" "${model}"
done
summarize_group "paper_baseline_bf16_boundary"

notify "论文后续精度实验队列完成" "完成" "int8 subset and bf16 boundary finished"
echo "[$(date '+%F %T')] Precision follow-up queue complete"
