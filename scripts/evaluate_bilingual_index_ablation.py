"""Evaluate source, monolingual, and bilingual railway retrieval indexes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.evaluate_retrieval import RetrievalCase, evidence_hit, load_cases
from annotation_system.backend.app.models import CorpusItem
from annotation_system.backend.app.rag import tokenize


RESULT_ROOT = ROOT / "results/ijwis_single_gpu_3090"
CACHE_DIR = RESULT_ROOT / "ablation_cache"
OUTPUT_JSON = RESULT_ROOT / "analysis/bilingual_index_ablation.json"
OUTPUT_CSV = ROOT / "paper/ijwis/tables/table10_bilingual_index_ablation.csv"
OUTPUT_TEX = ROOT / "paper/ijwis/tables/table10_bilingual_index_ablation.tex"
FIGURE_PNG = ROOT / "paper/ijwis/figures/bilingual_index_ablation.png"
FIGURE_PDF = ROOT / "paper/ijwis/figures/bilingual_index_ablation.pdf"

VARIANTS = ("source_only", "zh_only", "en_only", "bilingual")
VARIANT_LABELS = {
    "source_only": "Source evidence",
    "zh_only": "Chinese fields",
    "en_only": "English fields",
    "bilingual": "Bilingual fields",
}


def clean(*values: str) -> str:
    return " ".join(" ".join(value.split()) for value in values if value)


def variant_text(item: CorpusItem, variant: str, evidence: str) -> str:
    metadata = clean(item.domain_category, item.knowledge_category, item.chapter)
    if variant == "source_only":
        return clean(evidence, item.source_text, metadata)
    if variant == "zh_only":
        return clean(item.question, item.answer, evidence, item.source_text, metadata)
    if variant == "en_only":
        return clean(item.question_en, item.answer_en, metadata)
    if variant == "bilingual":
        return clean(
            item.question,
            item.question_en,
            item.answer,
            item.answer_en,
            evidence,
            item.source_text,
            metadata,
        )
    raise ValueError(variant)


def load_corpus() -> tuple[list[dict], dict[str, list[str]]]:
    rows: list[dict] = []
    texts = {variant: [] for variant in VARIANTS}
    seen_evidence: set[str] = set()
    with SessionLocal() as db:
        items = db.scalars(
            select(CorpusItem)
            .where(
                CorpusItem.source_type != "generated_eval_review",
                CorpusItem.review_status == "approved",
            )
            .order_by(CorpusItem.id)
        )
        for item in items:
            metadata = item.metadata_json or {}
            if metadata.get("split") == "test" or metadata.get("rag_test_set"):
                continue
            evidence = clean(item.evidence or item.source_text or item.answer)
            if len(evidence) < 4:
                continue
            evidence_key = f"{item.source_document}\x1f{item.page_number}\x1f{evidence}"
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            rows.append(
                {
                    "item_id": item.id,
                    "evidence": evidence[:2400],
                    "source_document": item.source_document,
                    "source_type": item.source_type,
                    "task_type": item.task_type,
                    "domain_category": item.domain_category,
                    "chapter": item.chapter,
                    "page_number": item.page_number,
                    "review_status": item.review_status,
                }
            )
            for variant in VARIANTS:
                texts[variant].append(variant_text(item, variant, evidence))
    return rows, texts


class Bm25Index:
    def __init__(self, texts: list[str]) -> None:
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        total_length = 0
        for index, text in enumerate(texts):
            frequencies = Counter(tokenize(text))
            self.term_frequencies.append(frequencies)
            total_length += sum(frequencies.values())
            for token in frequencies:
                self.document_frequencies[token] += 1
                self.postings[token].append(index)
        self.average_length = total_length / max(len(texts), 1)
        self.total_documents = len(texts)

    def search(self, query: str, top_k: int) -> list[int]:
        query_tokens = Counter(tokenize(query))
        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(self.postings.get(token, ()))
        scores: list[tuple[float, int]] = []
        for index in candidates:
            frequencies = self.term_frequencies[index]
            document_length = sum(frequencies.values())
            score = 0.0
            for token, query_frequency in query_tokens.items():
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies[token]
                inverse_document_frequency = math.log(
                    1 + (self.total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * document_length / max(1.0, self.average_length)
                )
                score += inverse_document_frequency * frequency * 2.5 / denominator
                score *= 1 + min(query_frequency, 3) * 0.05
            if score > 0:
                scores.append((score, index))
        return [index for _score, index in sorted(scores, reverse=True)[:top_k]]


def load_or_encode(
    model: SentenceTransformer,
    variant: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"bge_m3_{variant}_fp16.npy"
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == (len(texts), 1024):
            print(f"loaded_cache={cache_path} shape={cached.shape}", flush=True)
            return cached
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float16)
    np.save(cache_path, vectors)
    return np.load(cache_path, mmap_mode="r")


def dense_rankings(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[list[list[int]], float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = torch.from_numpy(np.asarray(corpus_embeddings)).to(device=device, dtype=torch.float16)
    rankings: list[list[int]] = []
    started = time.perf_counter()
    for offset in range(0, len(query_embeddings), 64):
        queries = torch.from_numpy(query_embeddings[offset : offset + 64]).to(device=device, dtype=torch.float16)
        scores = queries @ corpus.T
        indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()
        rankings.extend(indices)
    elapsed_ms = (time.perf_counter() - started) * 1000 / max(len(query_embeddings), 1)
    del corpus
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rankings, elapsed_ms


def rrf(bm25: list[int], dense: list[int], top_k: int, rank_constant: int = 60) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in (bm25, dense):
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (rank_constant + rank)
    return [index for index, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def rank_for_case(case: RetrievalCase, ranking: list[int], corpus_rows: list[dict]) -> int | None:
    for rank, index in enumerate(ranking, 1):
        if evidence_hit(case, corpus_rows[index]):
            return rank
    return None


def summarize(ranks: list[int | None], language: str, variant: str, method: str, latency_ms: float) -> dict:
    total = len(ranks)
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "method": method,
        "language": language,
        "cases": total,
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / total,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / total,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        "mrr": sum(1 / rank for rank in ranks if rank) / total,
        "mean_search_ms": latency_ms,
    }


def write_outputs(payload: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(payload["summaries"])
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_TEX.write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True),
        encoding="utf-8",
    )

    hybrid = frame[frame["method"] == "hybrid"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    colors = {"zh": "#0072B2", "en": "#D55E00"}
    x = np.arange(len(VARIANTS))
    for axis, metric, title in zip(axes, ("recall_at_1", "recall_at_5"), ("Recall@1", "Recall@5"), strict=True):
        for offset, language in ((-0.18, "zh"), (0.18, "en")):
            values = [
                hybrid[(hybrid["variant"] == variant) & (hybrid["language"] == language)][metric].iloc[0]
                for variant in VARIANTS
            ]
            axis.bar(x + offset, values, width=0.34, color=colors[language], label=language.upper())
        axis.set_title(title)
        axis.set_xticks(x, ["Source", "ZH", "EN", "Bilingual"], rotation=20)
        axis.set_ylim(0, 0.85)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Retrieval recall")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()

    corpus_rows, texts_by_variant = load_corpus()
    cases = load_cases(language="both", test_set="railway_bilingual_400")
    model = SentenceTransformer(args.model, local_files_only=not args.allow_model_download)
    model.max_seq_length = args.max_seq_length
    query_started = time.perf_counter()
    query_embeddings = model.encode(
        [case.question for case in cases],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float16)
    query_encode_ms = (time.perf_counter() - query_started) * 1000 / len(cases)

    summaries: list[dict] = []
    case_results: list[dict] = []
    for variant in VARIANTS:
        print(f"variant={variant} corpus={len(corpus_rows)}", flush=True)
        bm25_started = time.perf_counter()
        bm25_index = Bm25Index(texts_by_variant[variant])
        bm25_build_s = time.perf_counter() - bm25_started
        bm25_rankings: list[list[int]] = []
        bm25_latencies: list[float] = []
        for case in cases:
            started = time.perf_counter()
            bm25_rankings.append(bm25_index.search(case.question, args.candidate_k))
            bm25_latencies.append((time.perf_counter() - started) * 1000)

        embeddings = load_or_encode(model, variant, texts_by_variant[variant], args.batch_size)
        dense, dense_search_ms = dense_rankings(embeddings, query_embeddings, args.candidate_k)
        hybrid = [rrf(bm25, vector, args.candidate_k) for bm25, vector in zip(bm25_rankings, dense, strict=True)]

        for language in ("zh", "en"):
            selected = [index for index, case in enumerate(cases) if case.language == language]
            for method, rankings, latency in (
                ("bm25", bm25_rankings, float(np.mean(bm25_latencies))),
                ("vector", dense, query_encode_ms + dense_search_ms),
                ("hybrid", hybrid, float(np.mean(bm25_latencies)) + query_encode_ms + dense_search_ms),
            ):
                ranks = [rank_for_case(cases[index], rankings[index][:5], corpus_rows) for index in selected]
                summaries.append(summarize(ranks, language, variant, method, latency))
                for case_index, rank in zip(selected, ranks, strict=True):
                    case_results.append(
                        {
                            "item_id": cases[case_index].item_id,
                            "language": language,
                            "variant": variant,
                            "method": method,
                            "rank": rank,
                        }
                    )
        print(f"variant={variant} bm25_build_s={bm25_build_s:.2f}", flush=True)
        del bm25_index, embeddings

    payload = {
        "model": args.model,
        "candidate_documents": len(corpus_rows),
        "test_cases": len(cases),
        "candidate_k": args.candidate_k,
        "max_seq_length": args.max_seq_length,
        "variants": VARIANT_LABELS,
        "summaries": summaries,
        "case_results": case_results,
        "cases": [asdict(case) for case in cases],
    }
    write_outputs(payload)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"wrote={OUTPUT_JSON}")


if __name__ == "__main__":
    main()
