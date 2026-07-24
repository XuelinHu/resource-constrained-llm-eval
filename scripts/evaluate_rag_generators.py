"""Freeze RAG contexts and evaluate multiple local generators on identical inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from annotation_system.backend.app.config import settings
from annotation_system.backend.app.evaluate_qa import (
    citation_coverage,
    containment,
    f1_score,
    hallucination_proxy,
)
from annotation_system.backend.app.evaluate_retrieval import evidence_hit, load_cases, search
from rc_llm_eval.utils.config import load_all_configs
from rc_llm_eval.utils.modeling import clear_cuda, get_inference_device, load_model_and_tokenizer


STRATEGIES = ("no_retrieval", "bm25_rag", "hybrid_rag_approved")


def build_messages(language: str, question: str, sources: list[dict]) -> list[dict]:
    if language == "en":
        system = (
            "You are a bilingual assistant for international railway vocational education. "
            "Answer concisely in English. Use only supplied evidence when evidence is present, "
            "and cite it with labels such as [Evidence1]. State uncertainty instead of inventing facts."
        )
        label = "Evidence"
    else:
        system = (
            "你是面向国际铁路职业教育的双语问答助手。请用中文简洁回答。提供证据时只能依据证据，"
            "并使用[证据1]这样的编号引用；证据不足时说明不确定，不得编造。"
        )
        label = "证据"
    if not sources:
        user = question
    else:
        context = "\n\n".join(
            f"[{label}{index}] Source: {source['source_document']}\n{source['evidence']}"
            for index, source in enumerate(sources, 1)
        )
        user = f"Question: {question}\n\nEvidence:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prepare_cache(output: Path, top_k: int) -> None:
    rows = []
    cases = load_cases(language="both", test_set="railway_bilingual_400")
    for strategy in STRATEGIES:
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            if strategy == "no_retrieval":
                sources = []
            elif strategy == "bm25_rag":
                sources = search("bm25", case.question, top_k=top_k, approved_only=False)
            else:
                sources = search("hybrid", case.question, top_k=top_k, approved_only=True)
            retrieval_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "strategy": strategy,
                    "item_id": case.item_id,
                    "question": case.question,
                    "reference_answer": case.answer,
                    "task_type": case.task_type,
                    "source_document": case.source_document,
                    "language": case.language,
                    "top_k": top_k,
                    "retrieval_ms": retrieval_ms,
                    "evidence_hit": any(evidence_hit(case, source) for source in sources),
                    "sources": sources,
                    "messages": build_messages(case.language, case.question, sources),
                }
            )
            if index % 100 == 0:
                print(f"prepared strategy={strategy} cases={index}/{len(cases)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"top_k": top_k, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={output} rows={len(rows)}")


def render_prompt(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_hf(rows: list[dict], configs: dict, model_key: str, adapter: str | None, batch_size: int) -> list[dict]:
    model_cfg = configs["models"][model_key]
    clear_cuda()
    model, tokenizer = load_model_and_tokenizer(model_cfg, "int4", model_cfg.get("default_dtype", "bfloat16"), adapter)
    device = get_inference_device(model)
    outputs = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        prompts = [render_prompt(tokenizer, row["messages"]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(device)
        input_width = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        for index, row in enumerate(batch):
            answer = tokenizer.decode(generated[index, input_width:], skip_special_tokens=True).strip()
            outputs.append(score_row(row, answer, elapsed_ms / len(batch)))
        print(f"generated={min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)
    del model
    clear_cuda()
    return outputs


def call_ollama(messages: list[dict], model: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {"temperature": 0.0, "num_predict": 96},
    }
    request = Request(
        f"{settings.ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=settings.rag_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("message", {}).get("content", "").strip(), (time.perf_counter() - started) * 1000


def score_row(row: dict, answer: str, generation_ms: float) -> dict:
    reference = row["reference_answer"]
    return {
        **{key: value for key, value in row.items() if key != "messages"},
        "answer": answer,
        "answer_f1": f1_score(answer, reference),
        "reference_containment": containment(answer, reference),
        "citation_coverage": citation_coverage(answer),
        "hallucination_proxy": hallucination_proxy(answer),
        "answer_length_ratio": len(answer) / max(len(reference), 1),
        "generation_ms": generation_ms,
        "end_to_end_ms": row["retrieval_ms"] + generation_ms,
    }


def summarize(rows: list[dict], generator: str) -> list[dict]:
    metrics = (
        "answer_f1", "reference_containment", "citation_coverage", "evidence_hit",
        "hallucination_proxy", "retrieval_ms", "generation_ms", "end_to_end_ms",
    )
    summaries = []
    for strategy in STRATEGIES:
        for language in ("zh", "en"):
            selected = [row for row in rows if row["strategy"] == strategy and row["language"] == language]
            summary = {"generator": generator, "strategy": strategy, "language": language, "cases": len(selected)}
            summary.update({metric: statistics.mean(float(row[metric]) for row in selected) for metric in metrics})
            summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("results/ijwis_single_gpu_3090/rag/rag_contexts_top3.json"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=("hf", "ollama"), default="hf")
    parser.add_argument("--model-key")
    parser.add_argument("--adapter")
    parser.add_argument("--ollama-model", default=settings.rag_model)
    parser.add_argument("--label", default="generator")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare_cache(args.cache, args.top_k)
        return
    cached = json.loads(args.cache.read_text(encoding="utf-8"))["rows"]
    if args.backend == "hf":
        if not args.model_key:
            parser.error("--model-key is required for the hf backend")
        rows = generate_hf(cached, load_all_configs("configs/experiments/ijwis_single_gpu_3090.yaml"), args.model_key, args.adapter, args.batch_size)
    else:
        rows = []
        for index, row in enumerate(cached, 1):
            answer, generation_ms = call_ollama(row["messages"], args.ollama_model)
            rows.append(score_row(row, answer, generation_ms))
            if index % 50 == 0:
                print(f"generated={index}/{len(cached)}", flush=True)
    output = args.output or Path(f"results/ijwis_single_gpu_3090/rag/{args.label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summaries": summarize(rows, args.label), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
