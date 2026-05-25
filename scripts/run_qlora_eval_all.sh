#!/usr/bin/env bash
set -euo pipefail

# 对已经训练好的 QLoRA 适配器执行统一的 int4 评测。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-configs/experiments/single_gpu_3090.yaml}"
DATASET="${2:-domain_qa}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

cd "${ROOT_DIR}"

OUTPUT_ROOT="$("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print(configs["experiment"]["experiment"]["output_root"])
PY
)"

mapfile -t MODELS < <("${PYTHON_BIN}" - "${EXPERIMENT}" <<'PY'
from src.rc_llm_eval.utils.config import load_all_configs
import sys
configs = load_all_configs(sys.argv[1])
print("\n".join(configs["experiment"]["qlora"]["candidate_models"]))
PY
)

for model in "${MODELS[@]}"; do
  adapter_dir="${OUTPUT_ROOT}/qlora/${model}/adapter"
  if [[ ! -d "${adapter_dir}" ]]; then
    # 缺适配器时直接失败，避免产出不完整的对比结果。
    echo "Missing adapter for ${model}: ${adapter_dir}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" -m src.rc_llm_eval.cli run-eval \
    --experiment "${EXPERIMENT}" \
    --model "${model}" \
    --precision int4 \
    --peft-adapter "${adapter_dir}" \
    --output-group qlora_eval \
    --label "${DATASET}_adapter"
done

# 统一输出 qlora_eval 聚合结果，供表格导出使用。
"${PYTHON_BIN}" -m src.rc_llm_eval.cli summarize-results --experiment "${EXPERIMENT}" --output-group qlora_eval
