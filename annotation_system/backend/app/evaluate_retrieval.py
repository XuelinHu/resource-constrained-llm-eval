from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import select

from .database import SessionLocal
from .models import CorpusItem
from .rag import hybrid_search, retriever, vector_search


@dataclass
class RetrievalCase:
    item_id: int
    question: str
    answer: str
    evidence: str
    task_type: str
    source_document: str
    language: str


def load_cases(
    limit: int | None = None,
    language: str = "zh",
    test_set: str = "railway_bilingual_400",
) -> list[RetrievalCase]:
    with SessionLocal() as db:
        query = (
            select(CorpusItem)
            .where(
                CorpusItem.review_status == "approved",
                CorpusItem.metadata_json["rag_test_set"].as_string() == test_set,
                CorpusItem.question != "",
                CorpusItem.evidence != "",
            )
            .order_by(CorpusItem.id)
        )
        items = list(db.scalars(query))
    cases = []
    for item in items:
        languages = ("zh", "en") if language == "both" else (language,)
        for selected_language in languages:
            question = item.question if selected_language == "zh" else item.question_en
            answer = item.answer if selected_language == "zh" else item.answer_en
            if not question or not answer:
                continue
            cases.append(
                RetrievalCase(
                    item_id=item.id,
                    question=question,
                    answer=answer,
                    evidence=item.evidence,
                    task_type=item.task_type,
                    source_document=item.source_document,
                    language=selected_language,
                )
            )
    if limit:
        cases = cases[:limit]
    return cases


def evidence_hit(case: RetrievalCase, result: dict) -> bool:
    result_evidence = " ".join((result.get("evidence") or "").split())
    gold_evidence = " ".join(case.evidence.split())
    if result.get("item_id") == case.item_id:
        return True
    if gold_evidence and (gold_evidence in result_evidence or result_evidence in gold_evidence):
        return True
    return bool(case.source_document and result.get("source_document") == case.source_document and case.answer in result_evidence)


def search(mode: str, question: str, top_k: int, approved_only: bool) -> list[dict]:
    if mode == "bm25":
        return retriever.search(question, top_k=top_k, approved_only=approved_only)
    if mode == "vector":
        return vector_search(question, top_k=top_k, approved_only=approved_only)
    if mode == "hybrid":
        return hybrid_search(question, top_k=top_k, approved_only=approved_only)
    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_mode(cases: list[RetrievalCase], *, mode: str, top_k: int, approved_only: bool) -> dict:
    ranks: list[int | None] = []
    latencies: list[float] = []
    examples = []
    for case in cases:
        started = time.perf_counter()
        results = search(mode, case.question, top_k=top_k, approved_only=approved_only)
        latencies.append((time.perf_counter() - started) * 1000)
        rank = None
        for index, result in enumerate(results, 1):
            if evidence_hit(case, result):
                rank = index
                break
        ranks.append(rank)
        if len(examples) < 5:
            examples.append(
                {
                    "question": case.question,
                    "task_type": case.task_type,
                    "rank": rank,
                    "top_results": [
                        {
                            "item_id": result.get("item_id"),
                            "score": result.get("score"),
                            "task_type": result.get("task_type"),
                            "source_document": result.get("source_document"),
                            "evidence": (result.get("evidence") or "")[:160],
                        }
                        for result in results[:3]
                    ],
                }
            )

    total = len(cases)
    metrics = {
        "mode": mode,
        "approved_only": approved_only,
        "language": cases[0].language if cases and len({case.language for case in cases}) == 1 else "both",
        "cases": total,
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / total if total else 0.0,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / total if total else 0.0,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / total if total else 0.0,
        f"recall_at_{top_k}": sum(rank is not None and rank <= top_k for rank in ranks) / total if total else 0.0,
        "mrr": sum(1 / rank for rank in ranks if rank) / total if total else 0.0,
        "mean_latency_ms": sum(latencies) / total if total else 0.0,
        "examples": examples,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate railway retrieval modes on held-out test cases.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both")
    parser.add_argument("--test-set", default="railway_bilingual_400")
    parser.add_argument("--output", type=Path, default=Path("data/exports/retrieval_eval.json"))
    args = parser.parse_args()

    cases = load_cases(args.limit, args.language, args.test_set)
    modes = [
        ("bm25", False),
        ("vector", False),
        ("hybrid", False),
        ("hybrid", True),
    ]
    languages = sorted({case.language for case in cases})
    results = [
        evaluate_mode(
            [case for case in cases if case.language == language],
            mode=mode,
            top_k=args.top_k,
            approved_only=approved_only,
        )
        for mode, approved_only in modes
        for language in languages
    ]
    payload = {
        "cases": [asdict(case) for case in cases],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in result.items() if k != "examples"} for result in results], ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
