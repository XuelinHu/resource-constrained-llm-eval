#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
DATASET="${2:-domain_qa}"

MODELS=(
  qwen3_4b
  qwen2_5_7b_instruct
  gemma_3_4b
)

for model in "${MODELS[@]}"; do
  python -m src.rc_llm_eval.cli run-qlora --experiment "${EXPERIMENT}" --model "${model}" --dataset "${DATASET}"
done
