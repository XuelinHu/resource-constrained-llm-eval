#!/usr/bin/env bash
set -euo pipefail

# Pilot 只跑一个轻量模型，用于先验证环境、流程和结果落盘。
EXPERIMENT="${1:-configs/experiments/pilot_single_gpu_3090.yaml}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_XET

MODELS=(
  qwen3_0_6b
)

echo "Running pilot baseline with experiment: ${EXPERIMENT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"

for model in "${MODELS[@]}"; do
  echo "==> Pilot run for ${model}"
  python -m src.rc_llm_eval.cli run-eval --experiment "${EXPERIMENT}" --model "${model}"
done

# Pilot 完成后也输出聚合表，方便与正式流程保持一致。
python -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}"

echo "Pilot baseline complete."
