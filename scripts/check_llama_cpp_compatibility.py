"""Verify and smoke-test a GGUF model registered in models.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_CLI = Path("/ds2/workspace/ai/tools/llama.cpp/build/bin/llama-cli")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip().isdigit()]
    return sum(values) if values else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen3_8_27b_gguf")
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="请用两句话说明列车制动系统日常检查的目的，并避免给出未经证据支持的具体数值。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/model_compatibility/qwen3_8_27b_gguf.json",
    )
    args = parser.parse_args()

    models = yaml.safe_load((ROOT / "configs/models/models.yaml").read_text(encoding="utf-8"))["models"]
    model = models[args.model_key]
    model_path = Path(model["local_path"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not args.llama_cli.is_file():
        raise FileNotFoundError(args.llama_cli)

    actual_size = model_path.stat().st_size
    expected_size = int(model["file_size_bytes"])
    if actual_size != expected_size:
        raise ValueError(f"size mismatch: expected {expected_size}, got {actual_size}")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != model["sha256"]:
        raise ValueError(f"SHA-256 mismatch: expected {model['sha256']}, got {actual_sha256}")

    command = [
        str(args.llama_cli),
        "--model", str(model_path),
        "--gpu-layers", "all",
        "--ctx-size", str(args.ctx_size),
        "--predict", str(args.predict),
        "--temp", "0",
        "--flash-attn", "on",
        "--no-display-prompt",
        "--color", "off",
        "--single-turn",
        "--prompt", args.prompt,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_gpu_memory_mib = 0
    stop_monitor = threading.Event()

    def monitor() -> None:
        nonlocal peak_gpu_memory_mib
        while not stop_monitor.wait(0.25):
            current = gpu_memory_mib()
            if current is not None:
                peak_gpu_memory_mib = max(peak_gpu_memory_mib, current)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    stdout, stderr = process.communicate()
    elapsed_seconds = time.perf_counter() - started
    stop_monitor.set()
    monitor_thread.join()

    result = {
        "model_key": args.model_key,
        "official_model": model["hf_id"],
        "quantized_source": model["quantized_hf_id"],
        "model_path": str(model_path),
        "quantization": model["quantization"],
        "file_size_bytes": actual_size,
        "sha256": actual_sha256,
        "llama_cli": str(args.llama_cli),
        "ctx_size": args.ctx_size,
        "predicted_tokens_limit": args.predict,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "return_code": process.returncode,
        "prompt": args.prompt,
        "response": stdout.strip(),
        "stderr_tail": stderr.splitlines()[-80:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "model_key", "model_path", "elapsed_seconds", "peak_gpu_memory_mib", "return_code", "response"
    )}, ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
