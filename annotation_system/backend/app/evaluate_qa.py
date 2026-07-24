from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .config import settings
from .evaluate_retrieval import RetrievalCase, load_cases, search


HAN_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokens(text: str) -> list[str]:
    normalized = (text or "").lower()
    han = HAN_RE.findall(normalized)
    words = WORD_RE.findall(normalized)
    return han + words


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def containment(prediction: str, reference: str) -> float:
    prediction = " ".join((prediction or "").split())
    reference = " ".join((reference or "").split())
    if not prediction or not reference:
        return 0.0
    if reference in prediction:
        return 1.0
    return f1_score(prediction, reference)


def citation_coverage(answer: str) -> float:
    return 1.0 if re.search(r"\[(?:证据|Evidence)\s*\d+\]", answer or "", re.IGNORECASE) else 0.0


def hallucination_proxy(answer: str) -> float:
    normalized = (answer or "").lower()
    unsupported = "现有语料无法支持" in normalized or "no relevant evidence" in normalized
    has_citation = citation_coverage(answer) > 0
    return 0.0 if unsupported or has_citation else 1.0


def call_ollama(messages: list[dict], *, num_predict: int = 360) -> tuple[str, float]:
    payload = {
        "model": settings.rag_model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {"temperature": 0.1, "num_predict": num_predict},
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
    elapsed_ms = (time.perf_counter() - started) * 1000
    return body.get("message", {}).get("content", "").strip(), elapsed_ms


def generate_no_retrieval(case: RetrievalCase) -> tuple[str, list[dict], float, float]:
    system_prompt = (
        "You are a bilingual assistant for international railway vocational education. "
        "Answer directly in English and state uncertainty when needed."
        if case.language == "en"
        else "你是面向国际铁路职业教育的双语问答助手。请直接用中文回答问题；不知道时说明不确定。"
    )
    answer, generation_ms = call_ollama(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": case.question},
        ]
    )
    return answer, [], 0.0, generation_ms


def generate_with_retrieval(case: RetrievalCase, *, mode: str, top_k: int, approved_only: bool, generate: bool) -> tuple[str, list[dict], float, float]:
    started = time.perf_counter()
    sources = search(mode, case.question, top_k=top_k, approved_only=approved_only)
    retrieval_ms = (time.perf_counter() - started) * 1000
    if not sources:
        answer = (
            "No relevant evidence was found in the available corpus."
            if case.language == "en"
            else "现有语料中未检索到与该问题相关的证据。"
        )
        return answer, [], retrieval_ms, 0.0
    if not generate:
        return sources[0]["evidence"], sources, retrieval_ms, 0.0

    evidence_label = "Evidence" if case.language == "en" else "证据"
    context = "\n\n".join(
        f"[{evidence_label}{index}] Source: {source['source_document']}\n{source['evidence']}"
        for index, source in enumerate(sources, 1)
    )
    system_prompt = (
        "You are a bilingual assistant for international railway vocational education. "
        "Answer in English using only the supplied evidence. Do not add unsupported facts. "
        "Keep the answer concise and cite relevant sentences with labels such as [Evidence1]."
        if case.language == "en"
        else (
            "你是面向国际铁路职业教育的双语问答助手。只能依据提供的证据回答，"
            "不得补充证据中没有的事实。回答应简洁，并在相关句子后使用[证据1]这样的编号。"
        )
    )
    answer, generation_ms = call_ollama(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": f"Question: {case.question}\n\nEvidence:\n{context}"},
        ]
    )
    return answer, sources, retrieval_ms, generation_ms


def evaluate_case(case: RetrievalCase, *, strategy: str, top_k: int) -> dict:
    if strategy == "no_retrieval":
        answer, sources, retrieval_ms, generation_ms = generate_no_retrieval(case)
    elif strategy == "retrieval_only":
        answer, sources, retrieval_ms, generation_ms = generate_with_retrieval(
            case, mode="hybrid", top_k=top_k, approved_only=True, generate=False
        )
    elif strategy == "bm25_rag":
        answer, sources, retrieval_ms, generation_ms = generate_with_retrieval(
            case, mode="bm25", top_k=top_k, approved_only=False, generate=True
        )
    elif strategy == "vector_rag":
        answer, sources, retrieval_ms, generation_ms = generate_with_retrieval(
            case, mode="vector", top_k=top_k, approved_only=False, generate=True
        )
    elif strategy == "hybrid_rag":
        answer, sources, retrieval_ms, generation_ms = generate_with_retrieval(
            case, mode="hybrid", top_k=top_k, approved_only=False, generate=True
        )
    elif strategy == "hybrid_rag_approved":
        answer, sources, retrieval_ms, generation_ms = generate_with_retrieval(
            case, mode="hybrid", top_k=top_k, approved_only=True, generate=True
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return {
        "strategy": strategy,
        "item_id": case.item_id,
        "question": case.question,
        "reference_answer": case.answer,
        "answer": answer,
        "task_type": case.task_type,
        "source_document": case.source_document,
        "language": case.language,
        "answer_f1": f1_score(answer, case.answer),
        "reference_containment": containment(answer, case.answer),
        "citation_coverage": citation_coverage(answer),
        "hallucination_proxy": hallucination_proxy(answer),
        "answer_length_ratio": len(answer) / max(1, len(case.answer)),
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "end_to_end_ms": retrieval_ms + generation_ms,
        "sources": sources,
    }


def summarize(rows: list[dict], strategy: str, language: str) -> dict:
    subset = [row for row in rows if row["strategy"] == strategy and row["language"] == language]
    metrics = [
        "answer_f1",
        "reference_containment",
        "citation_coverage",
        "hallucination_proxy",
        "answer_length_ratio",
        "retrieval_ms",
        "generation_ms",
        "end_to_end_ms",
    ]
    summary = {"strategy": strategy, "language": language, "cases": len(subset)}
    for metric in metrics:
        summary[metric] = sum(float(row[metric]) for row in subset) / len(subset) if subset else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer generation on held-out railway QA cases.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both")
    parser.add_argument("--test-set", default="railway_bilingual_400")
    parser.add_argument("--strategies", nargs="*", default=["retrieval_only", "bm25_rag", "vector_rag", "hybrid_rag", "hybrid_rag_approved"])
    parser.add_argument("--include-no-retrieval", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/exports/qa_eval.json"))
    args = parser.parse_args()

    strategies = list(args.strategies)
    if args.include_no_retrieval and "no_retrieval" not in strategies:
        strategies.insert(0, "no_retrieval")
    cases = load_cases(args.limit, args.language, args.test_set)
    rows = []
    for strategy in strategies:
        for index, case in enumerate(cases, 1):
            print(f"evaluating strategy={strategy} case={index}/{len(cases)} item_id={case.item_id}", flush=True)
            rows.append(evaluate_case(case, strategy=strategy, top_k=args.top_k))

    languages = sorted({case.language for case in cases})
    summaries = [summarize(rows, strategy, language) for strategy in strategies for language in languages]
    payload = {"summaries": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
