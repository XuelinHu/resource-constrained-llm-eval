#!/usr/bin/env bash
set -euo pipefail

# 按预设模型列表顺序执行完整 baseline sweep。
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

mapfile -t MODELS < <("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print("\n".join(configs["experiment"]["baseline"]["models"]))
PY
)

for model in "${MODELS[@]}"; do
  # 每轮只跑一个模型，便于在单卡环境中顺序回收显存。
  "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-eval --experiment "${EXPERIMENT}" --model "${model}"
done

# 所有模型完成后统一汇总结果。
"${PYTHON_BIN}" -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}"
