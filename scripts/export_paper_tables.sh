#!/usr/bin/env bash
set -euo pipefail

# 汇总 baseline 结果后，导出论文使用的 LaTeX 表格文件。
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"

python -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}"
python -m src.rc_llm_eval.cli export-paper-tables --experiment "${EXPERIMENT}"
