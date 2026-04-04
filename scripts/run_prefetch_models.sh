#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python scripts/prefetch_models.py --experiment "${EXPERIMENT}"
