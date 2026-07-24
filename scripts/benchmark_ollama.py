"""Benchmark the Ollama reference generator with the frozen efficiency prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def gpu_memory_mib() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return float(output.strip().splitlines()[0])


def monitor_memory(stop: threading.Event, samples: list[float]) -> None:
    while not stop.wait(0.05):
        try:
            samples.append(gpu_memory_mib())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass


def generate(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    request = Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    chunks = []
    final = {}
    with urlopen(request, timeout=600) as response:
        for line in response:
            event = json.loads(line)
            content = event.get("message", {}).get("content", "")
            if content and first_token is None:
                first_token = time.perf_counter()
            chunks.append(content)
            if event.get("done"):
                final = event
    finished = time.perf_counter()
    tokens = int(final.get("eval_count", 0))
    generation_s = float(final.get("eval_duration", 0)) / 1e9
    return {
        "output": "".join(chunks),
        "first_token_latency_s": (first_token or finished) - started,
        "end_to_end_latency_s": finished - started,
        "generation_latency_s": generation_s,
        "new_tokens": tokens,
        "tokens_per_second": tokens / max(generation_s, 1e-12),
        "ollama_load_duration_s": float(final.get("load_duration", 0)) / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--prompts", type=Path, default=Path("data/efficiency/prompts.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/ijwis_single_gpu_3090/efficiency/qwen3_14b_ollama.json"))
    args = parser.parse_args()
    prompts = [json.loads(line) for line in args.prompts.open(encoding="utf-8") if line.strip()]
    generate(args.url, args.model, prompts[0]["prompt"], 16)
    static_memory = gpu_memory_mib()
    memory_samples = [static_memory]
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_memory, args=(stop, memory_samples), daemon=True)
    monitor.start()
    rows = []
    failures = 0
    for repeat in range(args.repeats):
        for index, prompt in enumerate(prompts, 1):
            try:
                result = generate(args.url, args.model, prompt["prompt"], args.max_tokens)
                rows.append(
                    {
                        "prompt_id": prompt["id"], "workload": prompt["workload"],
                        "repeat": repeat + 1, "prompt": prompt["prompt"], **result,
                    }
                )
            except Exception as exc:
                failures += 1
                rows.append({"prompt_id": prompt["id"], "workload": prompt["workload"], "repeat": repeat + 1, "error": str(exc)})
            if index % 10 == 0:
                print(f"repeat={repeat + 1} prompts={index}/{len(prompts)}", flush=True)
    stop.set()
    monitor.join(timeout=1)
    successful = [row for row in rows if "error" not in row]
    payload = {
        "model": args.model, "backend": "ollama", "num_unique_prompts": len(prompts),
        "repeats": args.repeats, "num_measurements": len(successful), "failures": failures,
        "mean_first_token_latency_s": statistics.mean(row["first_token_latency_s"] for row in successful),
        "mean_latency_s": statistics.mean(row["generation_latency_s"] for row in successful),
        "std_latency_s": statistics.stdev(row["generation_latency_s"] for row in successful),
        "mean_end_to_end_latency_s": statistics.mean(row["end_to_end_latency_s"] for row in successful),
        "mean_tokens_per_second": statistics.mean(row["tokens_per_second"] for row in successful),
        "static_gpu_memory_gb": static_memory / 1024,
        "peak_gpu_memory_gb": max(memory_samples) / 1024,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
