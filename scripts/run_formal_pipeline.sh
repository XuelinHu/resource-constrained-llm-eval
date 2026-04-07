#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
DATASET="${2:-domain_qa}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

cd "${ROOT_DIR}"

RUN_ROOT="$("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print(configs["experiment"]["experiment"]["output_root"])
PY
)"

FAIL_LOG="${RUN_ROOT}/formal_failures.log"
SKIP_LOG="${RUN_ROOT}/formal_skips.log"
mkdir -p "${RUN_ROOT}"
: > "${FAIL_LOG}"
: > "${SKIP_LOG}"

MODELS=(
  qwen3_0_6b
  qwen3_1_7b
  qwen3_4b
  qwen3_8b
  qwen2_5_7b_instruct
  deepseek_r1_distill_qwen_7b
  gemma_3_4b
)

PRECISIONS=(
  bf16
  int8
  int4
)

echo "[$(date '+%F %T')] Formal pipeline start"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset: ${DATASET}"
echo "Python: ${PYTHON_BIN}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "Run root: ${RUN_ROOT}"

log_failure() {
  local stage="$1"
  local model="$2"
  local detail="$3"
  printf '[%s] %s\t%s\t%s\n' "$(date '+%F %T')" "${stage}" "${model}" "${detail}" | tee -a "${FAIL_LOG}" >&2
}

run_or_log() {
  local stage="$1"
  local model="$2"
  shift 2
  if "$@"; then
    return 0
  fi
  log_failure "${stage}" "${model}" "command_failed"
  return 1
}

required_free_mib() {
  local stage="$1"
  local model="$2"
  local precision="${3:-bf16}"
  "${PYTHON_BIN}" - "${EXPERIMENT}" "${stage}" "${model}" "${precision}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import math
import sys

configs = load_all_configs(sys.argv[1])
stage = sys.argv[2]
model = sys.argv[3]
precision = sys.argv[4]
params_b = float(configs["models"][model]["params_b"])

if stage == "qlora_train":
    estimate = 5000 + params_b * 1200
elif precision == "bf16":
    estimate = 2500 + params_b * 2200
elif precision == "int8":
    estimate = 1800 + params_b * 1200
else:
    estimate = 1200 + params_b * 700

estimate = min(int(math.ceil(estimate)), 22000)
print(estimate)
PY
}

wait_for_gpu_budget() {
  local stage="$1"
  local model="$2"
  local precision="${3:-bf16}"
  local required
  required="$(required_free_mib "${stage}" "${model}" "${precision}")"
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ -n "${free_mib}" ]] && (( free_mib >= required )); then
      echo "[$(date '+%F %T')] GPU ready for ${stage} ${model} ${precision}: free ${free_mib} MiB, need ${required} MiB"
      break
    fi
    echo "[$(date '+%F %T')] Waiting for GPU for ${stage} ${model} ${precision}: free ${free_mib:-unknown} MiB, need ${required} MiB"
    sleep 60
  done
}

echo "[$(date '+%F %T')] Step 1/5: prefetch baseline models"
if ! "${PYTHON_BIN}" scripts/prefetch_models.py --experiment "${EXPERIMENT}"; then
  log_failure "prefetch" "global" "prefetch_script_reported_failures"
fi

AVAILABLE_MODELS=()
echo "[$(date '+%F %T')] Checking local model accessibility"
for model in "${MODELS[@]}"; do
  if "${PYTHON_BIN}" - "${EXPERIMENT}" "${model}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
from transformers import AutoConfig
import sys

configs = load_all_configs(sys.argv[1])
model_key = sys.argv[2]
hf_id = configs["models"][model_key]["hf_id"]
AutoConfig.from_pretrained(hf_id, trust_remote_code=True, local_files_only=True)
print(model_key)
PY
  then
    AVAILABLE_MODELS+=("${model}")
  else
    echo "${model}" | tee -a "${SKIP_LOG}"
    log_failure "availability_check" "${model}" "local_model_files_unavailable"
  fi
done

echo "[$(date '+%F %T')] Accessible models: ${AVAILABLE_MODELS[*]:-none}"
if [[ "${#AVAILABLE_MODELS[@]}" -eq 0 ]]; then
  echo "No locally accessible models available. Stopping." >&2
  exit 1
fi

echo "[$(date '+%F %T')] Step 2/5: baseline precision sweep"
for precision in "${PRECISIONS[@]}"; do
  for model in "${AVAILABLE_MODELS[@]}"; do
    wait_for_gpu_budget "baseline" "${model}" "${precision}"
    echo "[$(date '+%F %T')] Baseline ${model} ${precision}"
    run_or_log "baseline_${precision}" "${model}" \
      "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-eval \
      --experiment "${EXPERIMENT}" \
      --model "${model}" \
      --precision "${precision}" \
      --output-group baseline || true
  done
done
"${PYTHON_BIN}" -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}" --output-group baseline

echo "[$(date '+%F %T')] Step 3/5: qlora training"
for model in qwen3_4b qwen2_5_7b_instruct gemma_3_4b; do
  if printf '%s\n' "${AVAILABLE_MODELS[@]}" | grep -qx "${model}"; then
    wait_for_gpu_budget "qlora_train" "${model}" "int4"
    echo "[$(date '+%F %T')] QLoRA ${model}"
    run_or_log "qlora_train" "${model}" \
      "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-qlora --experiment "${EXPERIMENT}" --model "${model}" --dataset "${DATASET}" || true
  else
    log_failure "qlora_train" "${model}" "skipped_model_unavailable"
  fi
done

echo "[$(date '+%F %T')] Step 4/5: qlora adapted evaluation"
for model in qwen3_4b qwen2_5_7b_instruct gemma_3_4b; do
  if [[ -d "${RUN_ROOT}/qlora/${model}/adapter" ]]; then
    wait_for_gpu_budget "qlora_eval" "${model}" "int4"
    echo "[$(date '+%F %T')] QLoRA eval ${model}"
    run_or_log "qlora_eval" "${model}" \
      "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-eval \
      --experiment "${EXPERIMENT}" \
      --model "${model}" \
      --precision int4 \
      --peft-adapter "${RUN_ROOT}/qlora/${model}/adapter" \
      --output-group qlora_eval \
      --label "${DATASET}_adapter" || true
  else
    log_failure "qlora_eval" "${model}" "missing_adapter_directory"
  fi
done
"${PYTHON_BIN}" -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}" --output-group qlora_eval || true

echo "[$(date '+%F %T')] Step 5/5: export paper tables"
run_or_log "reporting" "global" "${PYTHON_BIN}" -m src.rc_llm_eval.cli export-paper-tables --experiment "${EXPERIMENT}" || true

echo "[$(date '+%F %T')] Formal pipeline complete"
