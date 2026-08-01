"""Evaluate claim-level evidence support and citation validity for RAG outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "results/ijwis_single_gpu_3090/rag"
OUTPUT_JSON = ROOT / "results/ijwis_single_gpu_3090/analysis/rag_evidence_support.json"
OUTPUT_CSV = ROOT / "paper/ijwis/tables/table12_rag_faithfulness.csv"
OUTPUT_TEX = ROOT / "paper/ijwis/tables/table12_rag_faithfulness.tex"
FIGURE_PNG = ROOT / "paper/ijwis/figures/rag_faithfulness.png"
FIGURE_PDF = ROOT / "paper/ijwis/figures/rag_faithfulness.pdf"
MODEL_FILES = {
    "qwen2_5_original": "qwen2_5_original.json",
    "qwen2_5_qlora": "qwen2_5_qlora.json",
    "glm_4_original": "glm_4_original.json",
    "glm_4_qlora": "glm_4_qlora.json",
    "qwen3_14b_reference": "qwen3_14b_reference.json",
}
STRATEGIES = {"bm25_rag", "hybrid_rag_approved"}
CITATION_RE = re.compile(
    r"\[(?:证据|Evidence)\s*\d+"
    r"(?:(?:\s*[,，、]\s*)(?:(?:证据|Evidence)\s*)?\d+)*\]",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=[.!?])\s+|[；;\n]+")
NO_EVIDENCE_MARKERS = (
    "现有语料中未检索到",
    "现有语料无法支持",
    "no relevant evidence",
    "insufficient evidence",
)


def claims(answer: str) -> list[dict]:
    rows = []
    for raw_sentence in SENTENCE_SPLIT_RE.split(answer or ""):
        sentence = " ".join(raw_sentence.split()).strip(" -")
        if len(sentence) < 3:
            continue
        citation_indices = [
            int(value)
            for citation_group in CITATION_RE.findall(sentence)
            for value in re.findall(r"\d+", citation_group)
        ]
        clean_claim = CITATION_RE.sub("", sentence).strip()
        if len(clean_claim) < 3:
            continue
        rows.append({"claim": clean_claim, "citation_indices": citation_indices})
    return rows


def evidence_text(sources: list[dict], indices: list[int] | None = None) -> str:
    if indices is None:
        selected = sources
    else:
        selected = [sources[index - 1] for index in indices if 1 <= index <= len(sources)]
    return "\n".join(" ".join((source.get("evidence") or "").split()) for source in selected if source.get("evidence"))


def is_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in NO_EVIDENCE_MARKERS)


def entailment_id(model) -> int:
    for key, value in model.config.label2id.items():
        if key.lower() == "entailment":
            return int(value)
    for value, label in model.config.id2label.items():
        if str(label).lower() == "entailment":
            return int(value)
    raise ValueError(f"Entailment label not found: {model.config.id2label}")


def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    if not pairs:
        return []
    device = next(model.parameters()).device
    target_id = entailment_id(model)
    scores: list[float] = []
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset : offset + batch_size]
        encoded = tokenizer(
            [premise for premise, _claim in batch],
            [claim for _premise, claim in batch],
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)[:, target_id]
        scores.extend(probabilities.cpu().tolist())
    return scores


def score_pairs_embedding(
    model: SentenceTransformer,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> list[float]:
    if not pairs:
        return []
    unique_texts = list(dict.fromkeys(text for pair in pairs for text in pair))
    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embedding_by_text = dict(zip(unique_texts, embeddings, strict=True))
    return [
        float(np.dot(embedding_by_text[premise], embedding_by_text[claim]))
        for premise, claim in pairs
    ]


def bootstrap_interval(values: list[float], seed: int = 42, repetitions: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict], generator: str, strategy: str, language: str) -> dict:
    subset = [
        row for row in rows
        if row["generator"] == generator and row["strategy"] == strategy and row["language"] == language
    ]
    claims_total = sum(row["claim_count"] for row in subset)
    supported_total = sum(row["supported_claims"] for row in subset)
    cited_total = sum(row["cited_claims"] for row in subset)
    valid_cited_total = sum(row["valid_cited_claims"] for row in subset)
    supported_cited_total = sum(row["supported_cited_claims"] for row in subset)
    supported_and_cited = sum(row["supported_and_cited_claims"] for row in subset)
    score_values = [row["faithfulness_score"] for row in subset if row["claim_count"]]
    fully_supported = [float(row["fully_supported"]) for row in subset if row["claim_count"]]
    faith_low, faith_high = bootstrap_interval(score_values)
    full_low, full_high = bootstrap_interval(fully_supported)
    return {
        "generator": generator,
        "strategy": strategy,
        "language": language,
        "answers": len(subset),
        "claims": claims_total,
        "faithfulness_score": float(np.mean(score_values)) if score_values else 0.0,
        "faithfulness_ci_low": faith_low,
        "faithfulness_ci_high": faith_high,
        "supported_claim_rate": supported_total / claims_total if claims_total else 0.0,
        "unsupported_claim_rate": 1 - supported_total / claims_total if claims_total else 0.0,
        "fully_supported_answer_rate": float(np.mean(fully_supported)) if fully_supported else 0.0,
        "fully_supported_ci_low": full_low,
        "fully_supported_ci_high": full_high,
        "citation_precision": supported_cited_total / valid_cited_total if valid_cited_total else 0.0,
        "citation_recall": supported_and_cited / supported_total if supported_total else 0.0,
        "valid_citation_rate": valid_cited_total / cited_total if cited_total else 0.0,
        "abstention_rate": sum(row["abstention"] for row in subset) / len(subset) if subset else 0.0,
    }


def plot_summary(frame: pd.DataFrame) -> None:
    hybrid = frame[frame["strategy"] == "hybrid_rag_approved"].copy()
    generators = list(MODEL_FILES)
    labels = ["Qwen", "Qwen QLoRA", "GLM", "GLM QLoRA", "Qwen3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    colors = {"zh": "#0072B2", "en": "#D55E00"}
    x = np.arange(len(generators))
    for axis, metric, title in zip(
        axes,
        ("supported_claim_rate", "citation_recall"),
        ("Evidence-supported claims", "Supported claims with valid citations"),
        strict=True,
    ):
        for offset, language in ((-0.18, "zh"), (0.18, "en")):
            values = [
                hybrid[(hybrid["generator"] == generator) & (hybrid["language"] == language)][metric].iloc[0]
                for generator in generators
            ]
            axis.bar(x + offset, values, width=0.34, color=colors[language], label=language.upper())
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Rate")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("embedding", "nli"), default="embedding")
    parser.add_argument("--model", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit-per-group", type=int, default=None)
    args = parser.parse_args()

    answer_rows: list[dict] = []
    pair_records: list[dict] = []
    for generator, filename in MODEL_FILES.items():
        payload = json.loads((RAG_DIR / filename).read_text(encoding="utf-8"))
        group_counts: dict[tuple[str, str], int] = {}
        for row in payload["rows"]:
            if row["strategy"] not in STRATEGIES:
                continue
            group = (row["strategy"], row["language"])
            if args.limit_per_group is not None and group_counts.get(group, 0) >= args.limit_per_group:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1
            parsed_claims = claims(row.get("answer") or "")
            answer_index = len(answer_rows)
            answer_rows.append(
                {
                    "generator": generator,
                    "strategy": row["strategy"],
                    "language": row["language"],
                    "item_id": row["item_id"],
                    "answer": row.get("answer") or "",
                    "sources": row.get("sources") or [],
                    "evidence_hit": bool(row.get("evidence_hit")),
                    "abstention": is_abstention(row.get("answer") or ""),
                    "claim_count": len(parsed_claims),
                    "supported_claims": 0,
                    "cited_claims": 0,
                    "valid_cited_claims": 0,
                    "supported_cited_claims": 0,
                    "supported_and_cited_claims": 0,
                    "faithfulness_score": 0.0,
                    "fully_supported": False,
                    "claims": parsed_claims,
                }
            )
            all_evidence = evidence_text(row.get("sources") or [])
            for claim_index, claim_row in enumerate(parsed_claims):
                citation_indices = claim_row["citation_indices"]
                valid_indices = [index for index in citation_indices if 1 <= index <= len(row.get("sources") or [])]
                pair_records.append(
                    {
                        "answer_index": answer_index,
                        "claim_index": claim_index,
                        "all_pair": (all_evidence, claim_row["claim"]),
                        "cited_pair": (
                            evidence_text(row.get("sources") or [], valid_indices),
                            claim_row["claim"],
                        ) if valid_indices else None,
                        "has_citation": bool(citation_indices),
                        "has_valid_citation": bool(valid_indices),
                    }
                )

    if args.backend == "embedding":
        args.model = args.model or "BAAI/bge-m3"
        args.threshold = 0.45 if args.threshold is None else args.threshold
        print(
            f"loading_embedding_model={args.model} answers={len(answer_rows)} claims={len(pair_records)}",
            flush=True,
        )
        model = SentenceTransformer(args.model, local_files_only=not args.allow_model_download)
        model.max_seq_length = args.max_seq_length
        score_function = lambda pairs: score_pairs_embedding(model, pairs, args.batch_size)
    else:
        args.model = args.model or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        args.threshold = 0.5 if args.threshold is None else args.threshold
        print(f"loading_nli_model={args.model} answers={len(answer_rows)} claims={len(pair_records)}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=not args.allow_model_download,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            local_files_only=not args.allow_model_download,
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        print(f"label_mapping={model.config.id2label}", flush=True)
        score_function = lambda pairs: score_pairs(model, tokenizer, pairs, args.batch_size)

    all_scores = score_function([record["all_pair"] for record in pair_records])
    cited_records = [record for record in pair_records if record["cited_pair"] is not None]
    cited_scores = score_function([record["cited_pair"] for record in cited_records])
    cited_score_by_pair = {id(record): score for record, score in zip(cited_records, cited_scores, strict=True)}

    answer_claim_scores: dict[int, list[float]] = {}
    for record, score in zip(pair_records, all_scores, strict=True):
        answer = answer_rows[record["answer_index"]]
        supported = score >= args.threshold
        answer_claim_scores.setdefault(record["answer_index"], []).append(score)
        answer["supported_claims"] += int(supported)
        answer["cited_claims"] += int(record["has_citation"])
        answer["valid_cited_claims"] += int(record["has_valid_citation"])
        if supported and record["has_valid_citation"]:
            answer["supported_and_cited_claims"] += 1
        cited_score = cited_score_by_pair.get(id(record))
        if cited_score is not None and cited_score >= args.threshold:
            answer["supported_cited_claims"] += 1

    for answer_index, answer in enumerate(answer_rows):
        scores = answer_claim_scores.get(answer_index, [])
        answer["faithfulness_score"] = float(np.mean(scores)) if scores else 0.0
        answer["fully_supported"] = bool(scores) and answer["supported_claims"] == len(scores)
        answer.pop("sources", None)
        answer.pop("claims", None)

    summaries = [
        summarize(answer_rows, generator, strategy, language)
        for generator in MODEL_FILES
        for strategy in sorted(STRATEGIES)
        for language in ("zh", "en")
    ]
    payload = {
        "scoring_backend": args.backend,
        "metric_model": args.model,
        "support_threshold": args.threshold,
        "max_sequence_length": args.max_seq_length,
        "interpretation_boundary": (
            "The automated score provides a semantic claim-evidence support estimate. "
            "It is not a substitute for expert factual or pedagogical judgement."
        ),
        "summaries": summaries,
        "answers": answer_rows,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(summaries)
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_TEX.write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True),
        encoding="utf-8",
    )
    plot_summary(frame)
    print(frame.to_string(index=False))
    print(f"wrote={OUTPUT_JSON}")


if __name__ == "__main__":
    main()
