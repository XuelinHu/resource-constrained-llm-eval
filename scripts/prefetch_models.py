from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from huggingface_hub import snapshot_download

from src.rc_llm_eval.utils.config import load_all_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prefetch baseline models into the local Hugging Face cache.")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/single_gpu_3090.yaml",
        help="Experiment config used to choose the model list.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts for each model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configs = load_all_configs(args.experiment)
    baseline_model_keys = configs["experiment"]["baseline"]["models"]

    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}")
    print(f"Preparing {len(baseline_model_keys)} baseline models")

    failed_models: list[str] = []

    for index, model_key in enumerate(baseline_model_keys, start=1):
        model_cfg = configs["models"][model_key]
        repo_id = model_cfg["hf_id"]
        print(f"[{index}/{len(baseline_model_keys)}] Prefetching {model_key} -> {repo_id}")
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    resume_download=True,
                    allow_patterns=[
                        "*.json",
                        "*.jsonl",
                        "*.txt",
                        "*.model",
                        "*.py",
                        "*.tiktoken",
                        "*.safetensors",
                        "*.bin",
                        "*.md",
                    ],
                )
                print(f"[{index}/{len(baseline_model_keys)}] Ready: {model_key}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[{index}/{len(baseline_model_keys)}] Attempt {attempt}/{args.retries} failed for {model_key}: {exc}")
                if attempt == args.retries:
                    failed_models.append(model_key)
                else:
                    time.sleep(min(30, 5 * attempt))

    if failed_models:
        print("Failed models:")
        for model_key in failed_models:
            print(f"- {model_key}")
        return 1

    print("All baseline models are cached locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
