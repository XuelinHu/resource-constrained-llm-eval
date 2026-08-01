"""Run a reproducible concurrency test against the deployed RAG Web API."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import delete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import RagSession


RESULT_ROOT = ROOT / "results/ijwis_single_gpu_3090"
OUTPUT_JSON = RESULT_ROOT / "analysis/web_load_test.json"
OUTPUT_CSV = ROOT / "paper/ijwis/tables/table13_web_load_test.csv"
OUTPUT_TEX = ROOT / "paper/ijwis/tables/table13_web_load_test.tex"
FIGURE_PNG = ROOT / "paper/ijwis/figures/web_load_test.png"
FIGURE_PDF = ROOT / "paper/ijwis/figures/web_load_test.pdf"

QUESTIONS = (
    "What is the purpose of railway interlocking equipment?",
    "How should a railway worker respond when a signal aspect is unclear?",
    "Explain the function of a track circuit in Chinese and English.",
    "铁路线路设备检查时应重点关注哪些安全风险？",
    "什么是铁路信号联锁，为什么它对行车安全重要？",
    "请用中英双语解释轨道电路的基本作用。",
)


def gpu_memory_mib() -> float:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return float("nan")


async def sample_gpu(stop: asyncio.Event, samples: list[float]) -> None:
    while not stop.is_set():
        samples.append(await asyncio.to_thread(gpu_memory_mib))
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


async def create_session(client: httpx.AsyncClient, run_tag: str, index: int) -> int:
    response = await client.post(
        "/api/rag/sessions",
        json={"title": f"[IJWIS-LOADTEST:{run_tag}] {index}"},
    )
    response.raise_for_status()
    return int(response.json()["id"])


async def issue_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    session_id: int,
    question: str,
    retrieval_mode: str,
    generate: bool,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/api/rag/ask",
                json={
                    "question": question,
                    "top_k": 3,
                    "generate": generate,
                    "session_id": session_id,
                    "retrieval_mode": retrieval_mode,
                    "approved_only": True,
                    "synthesize_audio": False,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            return {
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "retrieval_ms": float(payload["retrieval_ms"]),
                "generation_ms": float(payload["generation_ms"]),
                "sources": len(payload.get("sources") or []),
            }
        except Exception as error:
            return {
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}",
            }


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(values, level)) if values else float("nan")


async def run_setting(
    client: httpx.AsyncClient,
    run_tag: str,
    retrieval_mode: str,
    concurrency: int,
    requests: int,
    generate: bool,
    session_ids: list[int],
) -> tuple[dict, list[dict]]:
    setting_sessions = [
        await create_session(client, run_tag, len(session_ids) + index)
        for index in range(requests)
    ]
    session_ids.extend(setting_sessions)
    semaphore = asyncio.Semaphore(concurrency)
    gpu_samples: list[float] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu(stop, gpu_samples))
    started = time.perf_counter()
    rows = await asyncio.gather(
        *(
            issue_request(
                client,
                semaphore,
                setting_sessions[index],
                QUESTIONS[index % len(QUESTIONS)],
                retrieval_mode,
                generate,
            )
            for index in range(requests)
        )
    )
    wall_s = time.perf_counter() - started
    stop.set()
    await sampler
    successful = [row for row in rows if row["ok"]]
    latencies = [row["elapsed_ms"] for row in successful]
    summary = {
        "retrieval_mode": retrieval_mode,
        "concurrency": concurrency,
        "requests": requests,
        "successful": len(successful),
        "failures": requests - len(successful),
        "throughput_rps": len(successful) / wall_s,
        "end_to_end_p50_ms": percentile(latencies, 50),
        "end_to_end_p95_ms": percentile(latencies, 95),
        "mean_retrieval_ms": float(np.mean([row["retrieval_ms"] for row in successful]))
        if successful
        else float("nan"),
        "mean_generation_ms": float(np.mean([row["generation_ms"] for row in successful]))
        if successful
        else float("nan"),
        "peak_gpu_memory_mib": float(np.nanmax(gpu_samples)) if gpu_samples else float("nan"),
    }
    return summary, rows


def cleanup_sessions(session_ids: list[int]) -> None:
    if not session_ids:
        return
    with SessionLocal() as db:
        db.execute(delete(RagSession).where(RagSession.id.in_(session_ids)))
        db.commit()


def write_outputs(payload: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(payload["summaries"])
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_TEX.write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.2f}", escape=True),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    colors = {"bm25": "#0072B2", "hybrid": "#D55E00"}
    for mode, group in frame.groupby("retrieval_mode"):
        group = group.sort_values("concurrency")
        axes[0].plot(
            group["concurrency"],
            group["end_to_end_p95_ms"] / 1000,
            marker="o",
            color=colors[mode],
            label=mode.upper(),
        )
        axes[1].plot(
            group["concurrency"],
            group["throughput_rps"],
            marker="o",
            color=colors[mode],
            label=mode.upper(),
        )
    axes[0].set_ylabel("End-to-end P95 (s)")
    axes[1].set_ylabel("Throughput (requests/s)")
    for axis in axes:
        axis.set_xlabel("Concurrent requests")
        axis.set_xticks(sorted(frame["concurrency"].unique()))
        axis.grid(color="#d9d9d9", linewidth=0.6)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


async def async_main(args: argparse.Namespace) -> None:
    run_tag = uuid.uuid4().hex[:10]
    session_ids: list[int] = []
    summaries: list[dict] = []
    details: list[dict] = []
    warmups: list[dict] = []
    try:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
            health = await client.get("/api/health")
            health.raise_for_status()
            for retrieval_mode in ("bm25", "hybrid"):
                warmup_session = await create_session(client, run_tag, len(session_ids))
                session_ids.append(warmup_session)
                warmup = await issue_request(
                    client,
                    asyncio.Semaphore(1),
                    warmup_session,
                    QUESTIONS[0],
                    retrieval_mode,
                    args.generate,
                )
                warmups.append({"retrieval_mode": retrieval_mode, **warmup})
                for concurrency in args.concurrency:
                    print(f"mode={retrieval_mode} concurrency={concurrency}", flush=True)
                    summary, rows = await run_setting(
                        client,
                        run_tag,
                        retrieval_mode,
                        concurrency,
                        args.requests,
                        args.generate,
                        session_ids,
                    )
                    summaries.append(summary)
                    details.append({"setting": summary, "requests": rows})
    finally:
        cleanup_sessions(session_ids)

    payload = {
        "base_url": args.base_url,
        "run_tag": run_tag,
        "generate": args.generate,
        "approved_only": True,
        "top_k": 3,
        "questions": QUESTIONS,
        "warmups": warmups,
        "summaries": summaries,
        "details": details,
        "measurement_note": (
            "Client latency covers the non-streaming API path with TTS disabled. "
            "The formal concurrency run isolates retrieval unless --generate is supplied."
        ),
    }
    write_outputs(payload)
    print(pd.DataFrame(summaries).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
