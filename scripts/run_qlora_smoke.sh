#!/usr/bin/env bash
set -euo pipefail

# Foreground-only QLoRA smoke run. This uses the pilot config and one model so
# the user can verify the training path before launching formal experiments.
EXPERIMENT="${1:-configs/experiments/pilot_single_gpu_3090.yaml}"
MODEL="${2:-qwen3_4b}"
DATASET="${3:-domain_qa}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

"${PYTHON_BIN}" -m src.rc_llm_eval.cli run-qlora \
  --experiment "${EXPERIMENT}" \
  --model "${MODEL}" \
  --dataset "${DATASET}"
