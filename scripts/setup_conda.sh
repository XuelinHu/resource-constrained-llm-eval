#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-rc-llm-eval}"

conda env create -f environment.yml -n "${ENV_NAME}" || conda env update -f environment.yml -n "${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "Environment ready: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
