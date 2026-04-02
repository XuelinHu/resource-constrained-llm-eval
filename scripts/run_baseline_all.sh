#!/usr/bin/env bash
set -euo pipefail

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
  python -m src.rc_llm_eval.cli run-eval --experiment "${EXPERIMENT}" --model "${model}"
done

python -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}"
