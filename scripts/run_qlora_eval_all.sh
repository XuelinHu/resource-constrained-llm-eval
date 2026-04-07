#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
DATASET="${2:-domain_qa}"

cd "${ROOT_DIR}"

OUTPUT_ROOT="$(python - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print(configs["experiment"]["experiment"]["output_root"])
PY
)"

MODELS=(
  qwen3_4b
  qwen2_5_7b_instruct
  gemma_3_4b
)

for model in "${MODELS[@]}"; do
  adapter_dir="${OUTPUT_ROOT}/qlora/${model}/adapter"
  if [[ ! -d "${adapter_dir}" ]]; then
    echo "Missing adapter for ${model}: ${adapter_dir}" >&2
    exit 1
  fi

  python -m src.rc_llm_eval.cli run-eval \
    --experiment "${EXPERIMENT}" \
    --model "${model}" \
    --precision int4 \
    --peft-adapter "${adapter_dir}" \
    --output-group qlora_eval \
    --label "${DATASET}_adapter"
done

python -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}" --output-group qlora_eval
