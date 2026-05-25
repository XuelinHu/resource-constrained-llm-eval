#!/usr/bin/env bash
set -euo pipefail

# Foreground-only full smoke test over the configured smoke model pool.
# This script is intentionally not launched automatically.
EXPERIMENT="${1:-configs/experiments/smoke_single_gpu_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

"${PYTHON_BIN}" scripts/build_domain_smoke_dataset.py
"${PYTHON_BIN}" scripts/check_experiment_readiness.py --experiment "${EXPERIMENT}"

mapfile -t MODELS < <("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print("\n".join(configs["experiment"]["baseline"]["models"]))
PY
)

for model in "${MODELS[@]}"; do
  summary_path="results/smoke_single_gpu_3090/smoke_baseline/${model}/${model}_int4_domain_qa_smoke_summary.json"
  if [[ "${SMOKE_FORCE:-0}" != "1" && -f "${summary_path}" ]]; then
    echo "Skipping completed smoke model: ${model}"
    continue
  fi
  if [[ "${SMOKE_FORCE:-0}" == "1" ]]; then
    echo "Re-running smoke model: ${model}"
  fi
  "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-eval \
    --experiment "${EXPERIMENT}" \
    --model "${model}" \
    --precision int4 \
    --output-group smoke_baseline \
    --label domain_qa_smoke
done

"${PYTHON_BIN}" -m src.rc_llm_eval.cli summarize-results \
  --experiment "${EXPERIMENT}" \
  --output-group smoke_baseline
