#!/usr/bin/env bash
set -euo pipefail

# 按预设模型列表顺序执行完整 baseline sweep。
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"

MODELS=(
  qwen3_0_6b
  qwen3_1_7b
  qwen3_4b
  qwen3_8b
  qwen2_5_7b_instruct
  deepseek_r1_distill_qwen_7b
  gemma_3_4b
)

for model in "${MODELS[@]}"; do
  # 每轮只跑一个模型，便于在单卡环境中顺序回收显存。
  python -m src.rc_llm_eval.cli run-eval --experiment "${EXPERIMENT}" --model "${model}"
done

# 所有模型完成后统一汇总结果。
python -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}"
