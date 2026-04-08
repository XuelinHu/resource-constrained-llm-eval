#!/usr/bin/env bash
set -euo pipefail

# 调用 Python 预下载脚本，提前把 baseline 模型缓存到本地。
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python scripts/prefetch_models.py --experiment "${EXPERIMENT}"
