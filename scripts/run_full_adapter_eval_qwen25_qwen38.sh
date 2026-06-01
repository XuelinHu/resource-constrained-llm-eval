#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/single_gpu_3090.yaml}"
OUTPUT_GROUP="${OUTPUT_GROUP:-qlora_eval_full}"
LABEL="${LABEL:-domain_qa_full_adapter}"
PRECISION="${PRECISION:-int4}"

cd "${ROOT_DIR}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

run_adapter_eval() {
  local model="$1"
  local adapter="$2"

  if [[ ! -f "${adapter}/adapter_model.safetensors" ]]; then
    log "missing adapter weights: ${adapter}/adapter_model.safetensors"
    return 2
  fi

  log "starting formal adapter eval: model=${model}, precision=${PRECISION}, output_group=${OUTPUT_GROUP}, label=${LABEL}"
  "${PYTHON_BIN}" -u -m src.rc_llm_eval.cli run-eval \
    --experiment "${EXPERIMENT}" \
    --model "${model}" \
    --precision "${PRECISION}" \
    --peft-adapter "${adapter}" \
    --output-group "${OUTPUT_GROUP}" \
    --label "${LABEL}"
  log "finished formal adapter eval: model=${model}"
}

log "formal adapter eval chain started"
run_adapter_eval "qwen2_5_7b_instruct" "results/single_gpu_3090/qlora/qwen2_5_7b_instruct/adapter"
run_adapter_eval "qwen3_8b" "results/single_gpu_3090/qlora/qwen3_8b/adapter"

log "summarizing results: output_group=${OUTPUT_GROUP}"
"${PYTHON_BIN}" -u -m src.rc_llm_eval.cli summarize-results \
  --experiment "${EXPERIMENT}" \
  --output-group "${OUTPUT_GROUP}"
log "formal adapter eval chain completed"
