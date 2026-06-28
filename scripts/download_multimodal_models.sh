#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/multimodal}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"

models=(
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "OpenGVLab/InternVL3_5-8B-HF"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "microsoft/Florence-2-large"
  "microsoft/Florence-2-base"
)

mkdir -p "$MODEL_DIR"

for model in "${models[@]}"; do
  local_dir="$MODEL_DIR/${model#*/}"
  mkdir -p "$local_dir"
  echo "Downloading $model -> $local_dir"
  hf download "$model" --local-dir "$local_dir" --max-workers "$HF_MAX_WORKERS"
done
