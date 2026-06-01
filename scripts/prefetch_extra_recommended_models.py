"""Prefetch extra recommended open LLMs into a local Hugging Face cache.

The main experiment config intentionally keeps a smaller baseline set. This
helper is for ad-hoc caching of larger recommended models without changing the
formal experiment registry.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


MODELS: dict[str, str] = {
    "phi_4_mini_instruct": "microsoft/Phi-4-mini-instruct",
    "gemma_3_4b_it": "google/gemma-3-4b-it",
    "starcoder2_7b": "bigcode/starcoder2-7b",
    "gemma_3_12b_it": "google/gemma-3-12b-it",
    "phi_4": "microsoft/phi-4",
    "mistral_nemo_12b_instruct": "mistralai/Mistral-Nemo-Instruct-2407",
    "starcoder2_15b": "bigcode/starcoder2-15b",
}

ALLOW_PATTERNS = [
    "*.json",
    "*.jsonl",
    "*.txt",
    "*.model",
    "*.py",
    "*.tiktoken",
    "*.safetensors",
    "*.bin",
    "*.md",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prefetch extra recommended models.")
    parser.add_argument(
        "--cache-dir",
        default="/ds1/workspace/ai/hf_cache/hub",
        help="Hugging Face hub cache directory.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per model.")
    parser.add_argument("--max-workers", type=int, default=4, help="hf download worker count.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Specific model key to download. Can be passed multiple times.",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip local config/tokenizer verification.")
    return parser


def download_model(key: str, repo_id: str, cache_dir: Path, retries: int, max_workers: int) -> bool:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    cmd_base = [
        "hf",
        "download",
        repo_id,
        "--cache-dir",
        str(cache_dir),
        "--max-workers",
        str(max_workers),
        "--quiet",
    ]
    for pattern in ALLOW_PATTERNS:
        cmd_base.extend(["--include", pattern])

    for attempt in range(1, retries + 1):
        print(f"[{key}] download attempt {attempt}/{retries}: {repo_id}", flush=True)
        started = time.time()
        proc = subprocess.run(cmd_base, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        elapsed = time.time() - started
        if proc.returncode == 0:
            target = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            print(f"[{key}] download ok in {elapsed:.1f}s {target}", flush=True)
            return True

        print(f"[{key}] download failed in {elapsed:.1f}s rc={proc.returncode}", flush=True)
        if proc.stderr.strip():
            print(proc.stderr.strip()[-2000:], flush=True)
        if attempt < retries:
            time.sleep(min(60, 10 * attempt))
    return False


def verify_model(key: str, repo_id: str, cache_dir: Path) -> bool:
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(repo_id, local_files_only=True, trust_remote_code=True)
        AutoTokenizer.from_pretrained(repo_id, local_files_only=True, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{key}] verify failed: {type(exc).__name__}: {exc}", flush=True)
        return False

    print(f"[{key}] verify ok", flush=True)
    return True


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    requested_keys = args.models or list(MODELS)
    unknown = [key for key in requested_keys if key not in MODELS]
    if unknown:
        print("Unknown model key(s):", ", ".join(unknown), file=sys.stderr)
        return 2

    print(f"cache_dir={cache_dir}", flush=True)
    failed: list[str] = []
    for index, key in enumerate(requested_keys, start=1):
        repo_id = MODELS[key]
        print(f"[{index}/{len(requested_keys)}] start {key}", flush=True)
        if not download_model(key, repo_id, cache_dir, args.retries, args.max_workers):
            failed.append(key)
            continue
        if not args.skip_verify and not verify_model(key, repo_id, cache_dir):
            failed.append(key)

    if failed:
        print("Failed model(s):", flush=True)
        for key in failed:
            print(f"- {key}: {MODELS[key]}", flush=True)
        return 1

    print("All requested extra models are cached and verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
