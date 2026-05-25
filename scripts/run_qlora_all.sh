#!/usr/bin/env bash
set -euo pipefail

# 依次训练预选模型，避免单卡并发导致资源冲突。
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
DATASET="${2:-domain_qa}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

mapfile -t MODELS < <("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print("\n".join(configs["experiment"]["qlora"]["candidate_models"]))
PY
)

for model in "${MODELS[@]}"; do
  "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-qlora --experiment "${EXPERIMENT}" --model "${model}" --dataset "${DATASET}"
done
