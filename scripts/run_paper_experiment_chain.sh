#!/usr/bin/env bash
set -euo pipefail

# Paper experiment chain for the current manuscript plan.
# It runs one model at a time on the single RTX 3090, writes resumable outputs,
# and relies on the existing run-eval DingTalk notifications plus stage notices.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
PRECISION="${PRECISION:-int4}"
BASELINE_GROUP="${BASELINE_GROUP:-paper_baseline_int4}"
BASELINE_LABEL="${BASELINE_LABEL:-public_domain_regqa}"
ADAPTER_GROUP="${ADAPTER_GROUP:-paper_adapter_domainqa_int4}"
ADAPTER_LABEL="${ADAPTER_LABEL:-domainqa_adapter_public_domain_regqa}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

cd "${ROOT_DIR}"

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

run_or_notify_failure() {
  local stage="$1"
  shift
  if "$@"; then
    return 0
  fi
  notify "论文实验链阶段失败" "失败" "${stage}"
  return 1
}

output_root="$("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print(configs["experiment"]["experiment"]["output_root"])
PY
)"

mapfile -t MODELS < <("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print("\n".join(configs["experiment"]["baseline"]["models"]))
PY
)

ADAPTER_MODELS=(
  qwen2_5_7b_instruct
  qwen3_8b
)

echo "[$(date '+%F %T')] Paper experiment chain start"
echo "Experiment: ${EXPERIMENT}"
echo "Precision: ${PRECISION}"
echo "Baseline group: ${BASELINE_GROUP}"
echo "Adapter group: ${ADAPTER_GROUP}"
echo "Output root: ${output_root}"

notify "论文实验链启动" "开始" "baseline=${BASELINE_GROUP}, adapter=${ADAPTER_GROUP}, precision=${PRECISION}"

run_or_notify_failure "readiness_check" \
  "${PYTHON_BIN}" scripts/check_experiment_readiness.py --experiment "${EXPERIMENT}"

echo "[$(date '+%F %T')] Stage 1: ${PRECISION} baseline over public + domain + regqa tasks"
for model in "${MODELS[@]}"; do
  summary_path="${output_root}/${BASELINE_GROUP}/${model}/${model}_${PRECISION}_${BASELINE_LABEL}_summary.json"
  if [[ "${FORCE:-0}" != "1" && -f "${summary_path}" ]]; then
    echo "[$(date '+%F %T')] Skip completed baseline: ${model}"
    continue
  fi
  echo "[$(date '+%F %T')] Baseline ${model} ${PRECISION}"
  run_or_notify_failure "baseline_${model}_${PRECISION}" \
    "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli run-eval \
      --experiment "${EXPERIMENT}" \
      --model "${model}" \
      --precision "${PRECISION}" \
      --output-group "${BASELINE_GROUP}" \
      --label "${BASELINE_LABEL}"
done

run_or_notify_failure "summarize_${BASELINE_GROUP}" \
  "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli summarize-results \
    --experiment "${EXPERIMENT}" \
    --output-group "${BASELINE_GROUP}"

notify "论文实验链阶段完成" "完成" "baseline ${PRECISION} finished: ${BASELINE_GROUP}"

echo "[$(date '+%F %T')] Stage 2: existing domain_qa adapters evaluated on public + domain + regqa tasks"
for model in "${ADAPTER_MODELS[@]}"; do
  adapter_dir="${output_root}/qlora/${model}/adapter"
  summary_path="${output_root}/${ADAPTER_GROUP}/${model}/${model}_${PRECISION}_${ADAPTER_LABEL}_summary.json"
  if [[ ! -d "${adapter_dir}" ]]; then
    echo "[$(date '+%F %T')] Missing adapter, skip: ${model} ${adapter_dir}" >&2
    notify "论文实验链适配器跳过" "失败" "missing adapter for ${model}: ${adapter_dir}"
    continue
  fi
  if [[ "${FORCE:-0}" != "1" && -f "${summary_path}" ]]; then
    echo "[$(date '+%F %T')] Skip completed adapter eval: ${model}"
    continue
  fi
  echo "[$(date '+%F %T')] Adapter eval ${model} ${PRECISION}"
  run_or_notify_failure "adapter_${model}_${PRECISION}" \
    "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli run-eval \
      --experiment "${EXPERIMENT}" \
      --model "${model}" \
      --precision "${PRECISION}" \
      --peft-adapter "${adapter_dir}" \
      --output-group "${ADAPTER_GROUP}" \
      --label "${ADAPTER_LABEL}"
done

run_or_notify_failure "summarize_${ADAPTER_GROUP}" \
  "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli summarize-results \
    --experiment "${EXPERIMENT}" \
    --output-group "${ADAPTER_GROUP}"

notify "论文实验链完成" "完成" "baseline and existing adapter evaluation finished"
echo "[$(date '+%F %T')] Paper experiment chain complete"
