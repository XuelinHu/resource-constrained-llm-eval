#!/usr/bin/env bash
set -euo pipefail

# 如果环境不存在则创建，存在则按 environment.yml 增量更新。
ENV_NAME="${1:-rc-llm-eval}"

conda env create -f environment.yml -n "${ENV_NAME}" || conda env update -f environment.yml -n "${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

# 输出后续激活提示，便于人工执行下一步。
echo "Environment ready: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
