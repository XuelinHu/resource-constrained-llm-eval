"""Create traceable IJWIS metrics, statistics, error analysis, tables and figures."""

from __future__ import annotations

import json
import hashlib
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/ijwis_single_gpu_3090"
ANALYSIS = RESULTS / "analysis"
FIGURES = ROOT / "paper/ijwis/figures"
TABLES = ROOT / "paper/ijwis/tables"
MODELS = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B",
    "glm_4_9b_chat_hf": "GLM-4-9B",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_ci(values: list[float], seed: int = 42, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def generation_files() -> list[tuple[str, str, Path]]:
    found = []
    for condition, group in (("original", "baseline"), ("qlora", "qlora_eval")):
        for model_key in MODELS:
            folder = RESULTS / group / model_key
            for path in folder.glob("*_generations.json") if folder.exists() else []:
                if "efficiency" not in path.name:
                    found.append((model_key, condition, path))
    return found


def aggregate_generation_metrics() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    records = []
    samples_by_condition = {}
    for model_key, condition, path in generation_files():
        task = "bilingual_approved_qa" if "bilingual_approved_qa" in path.name else "domain_qa"
        samples = load_json(path).get("samples", [])
        frame = pd.DataFrame(samples)
        if frame.empty:
            continue
        frame["model_key"] = model_key
        frame["condition"] = condition
        frame["task"] = task
        samples_by_condition[(model_key, condition, task)] = frame
        for language in sorted(frame["language"].dropna().unique()):
            selected = frame[frame["language"] == language]
            for metric in ("exact_match", "char_f1", "token_f1", "reference_contained", "length_ratio"):
                values = [float(item.get(metric, 0.0)) for item in selected["metrics"]]
                low, high = bootstrap_ci(values)
                records.append(
                    {
                        "model": MODELS[model_key], "model_key": model_key, "condition": condition,
                        "task": task, "language": language, "metric": metric,
                        "score": np.mean(values), "ci95_low": low, "ci95_high": high,
                        "n": len(values), "source_file": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(records), samples_by_condition


def paired_statistics(samples: dict) -> pd.DataFrame:
    rows = []
    for model_key in MODELS:
        for task in ("bilingual_approved_qa", "domain_qa"):
            original = samples.get((model_key, "original", task))
            adapted = samples.get((model_key, "qlora", task))
            if original is None or adapted is None:
                continue
            keys = [key for key in ("pair_id", "language", "category", "prompt") if key in original.columns]
            left = original.copy()
            right = adapted.copy()
            left["sample_key"] = left[keys].fillna("").astype(str).agg(lambda row: "|".join(row.tolist()), axis=1)
            right["sample_key"] = right[keys].fillna("").astype(str).agg(lambda row: "|".join(row.tolist()), axis=1)
            merged = left.merge(right, on="sample_key", suffixes=("_original", "_qlora"))
            for language in ("zh", "en"):
                subset = merged[merged["language_original"] == language]
                if subset.empty:
                    continue
                for metric in ("char_f1", "reference_contained"):
                    before = np.array([float(value.get(metric, 0.0)) for value in subset["metrics_original"]])
                    after = np.array([float(value.get(metric, 0.0)) for value in subset["metrics_qlora"]])
                    differences = after - before
                    try:
                        p_value = float(wilcoxon(differences).pvalue)
                    except ValueError:
                        p_value = 1.0
                    rows.append(
                        {
                            "comparison": f"{MODELS[model_key]} QLoRA vs original", "task": task,
                            "language": language, "metric": metric, "n": len(differences),
                            "original_mean": before.mean(), "qlora_mean": after.mean(),
                            "absolute_gain": differences.mean(),
                            "relative_gain_pct": 100 * differences.mean() / max(abs(before.mean()), 1e-12),
                            "cohen_dz": differences.mean() / max(differences.std(ddof=1), 1e-12),
                            "p_value": p_value,
                        }
                    )
    for task in ("bilingual_approved_qa", "domain_qa"):
        for condition in ("original", "qlora"):
            qwen = samples.get(("qwen2_5_7b_instruct", condition, task))
            glm = samples.get(("glm_4_9b_chat_hf", condition, task))
            if qwen is None or glm is None:
                continue
            keys = [key for key in ("pair_id", "language", "category", "prompt") if key in qwen.columns]
            left = qwen.copy()
            right = glm.copy()
            left["sample_key"] = left[keys].fillna("").astype(str).agg("|".join, axis=1)
            right["sample_key"] = right[keys].fillna("").astype(str).agg("|".join, axis=1)
            merged = left.merge(right, on="sample_key", suffixes=("_qwen", "_glm"))
            for language in ("zh", "en"):
                subset = merged[merged["language_qwen"] == language]
                if subset.empty:
                    continue
                before = np.array([float(value.get("char_f1", 0.0)) for value in subset.metrics_glm])
                after = np.array([float(value.get("char_f1", 0.0)) for value in subset.metrics_qwen])
                rows.append(comparison_row(
                    f"Qwen2.5-7B vs GLM-4-9B: {condition}", task, language,
                    "char_f1", before, after,
                ))
        for model_key in MODELS:
            for condition in ("original", "qlora"):
                frame = samples.get((model_key, condition, task))
                if frame is None or "pair_id" not in frame:
                    continue
                paired = frame.copy()
                paired["score"] = paired.metrics.map(lambda value: float(value.get("char_f1", 0.0)))
                pivot = paired.pivot_table(index="pair_id", columns="language", values="score", aggfunc="mean").dropna()
                if not {"zh", "en"}.issubset(pivot.columns):
                    continue
                rows.append(comparison_row(
                    f"English vs Chinese: {MODELS[model_key]} {condition}", task, "en-vs-zh",
                    "char_f1", pivot.zh.to_numpy(), pivot.en.to_numpy(),
                ))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        order = frame["p_value"].sort_values().index.tolist()
        adjusted = pd.Series(index=frame.index, dtype=float)
        running = 0.0
        count = len(order)
        for rank, index in enumerate(order):
            running = max(running, min(1.0, frame.loc[index, "p_value"] * (count - rank)))
            adjusted.loc[index] = running
        frame["p_holm"] = adjusted
    return frame


def comparison_row(label: str, task: str, language: str, metric: str,
                   before: np.ndarray, after: np.ndarray) -> dict:
    differences = after - before
    try:
        p_value = float(wilcoxon(differences).pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "comparison": label, "task": task, "language": language, "metric": metric,
        "n": len(differences), "original_mean": before.mean(), "qlora_mean": after.mean(),
        "absolute_gain": differences.mean(),
        "relative_gain_pct": 100 * differences.mean() / max(abs(before.mean()), 1e-12),
        "cohen_dz": differences.mean() / max(differences.std(ddof=1), 1e-12),
        "p_value": p_value,
    }


def error_analysis(samples: dict) -> pd.DataFrame:
    rows = []
    for (model_key, condition, task), frame in samples.items():
        counts = Counter()
        examples = {}
        for _, sample in frame.iterrows():
            metrics = sample["metrics"]
            prediction = str(sample.get("prediction", ""))
            reference = str(sample.get("reference", ""))
            labels = []
            if float(metrics.get("char_f1", 0.0)) < 0.2:
                labels.append("low_answer_overlap")
            if float(metrics.get("length_ratio", 0.0)) > 4.0:
                labels.append("overlong_answer")
            if not prediction.strip():
                labels.append("empty_answer")
            if sample.get("category", "").startswith("terminology_") and not metrics.get("reference_contained"):
                labels.append("terminology_error")
            if sample.get("language") == "en" and any("\u4e00" <= char <= "\u9fff" for char in prediction):
                labels.append("wrong_language")
            for label in labels:
                counts[label] += 1
                examples.setdefault(label, {"prompt": sample.get("prompt"), "prediction": prediction, "reference": reference})
        for label, count in counts.items():
            rows.append(
                {
                    "model": MODELS[model_key], "condition": condition, "task": task,
                    "error_type": label, "count": count, "proportion": count / len(frame),
                    "representative_example": json.dumps(examples[label], ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def retrieval_table() -> pd.DataFrame:
    files = sorted((ROOT / "data/exports").glob("retrieval_eval_railway_bilingual_400*.json"))
    if not files:
        return pd.DataFrame()
    payload = load_json(files[-1])
    rows = []
    for row in payload["results"]:
        clean = {key: value for key, value in row.items() if key != "examples"}
        clean["source_file"] = str(files[-1].relative_to(ROOT))
        rows.append(clean)
    return pd.DataFrame(rows)


def retrieval_ablation_table() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "data/exports").glob("retrieval_eval_top*.json")):
        top_k = int(path.stem.rsplit("top", 1)[1])
        for row in load_json(path).get("results", []):
            rows.append(
                {
                    **{key: value for key, value in row.items() if key != "examples"},
                    "top_k": top_k, "source_file": str(path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def rag_table() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "rag").glob("*.json")) if (RESULTS / "rag").exists() else []:
        payload = load_json(path)
        samples = pd.DataFrame(payload.get("rows", []))
        for row in payload.get("summaries", []):
            selected = samples[(samples.strategy == row["strategy"]) & (samples.language == row["language"])]
            low, high = bootstrap_ci(selected.answer_f1.astype(float).tolist())
            rows.append({**row, "answer_f1_ci95_low": low, "answer_f1_ci95_high": high,
                         "source_file": str(path.relative_to(ROOT))})
    return pd.DataFrame(rows)


def rag_paired_statistics() -> pd.DataFrame:
    folder = RESULTS / "rag"
    payloads = {}
    for path in folder.glob("*.json") if folder.exists() else []:
        payload = load_json(path)
        if payload.get("rows") and payload.get("summaries"):
            payloads[path.stem] = pd.DataFrame(payload["rows"])
    rows = []

    def compare(label: str, left: pd.DataFrame, right: pd.DataFrame, language: str) -> None:
        left = left[left.language == language][["item_id", "language", "answer_f1"]]
        right = right[right.language == language][["item_id", "language", "answer_f1"]]
        merged = left.merge(right, on=["item_id", "language"], suffixes=("_left", "_right"))
        before = merged["answer_f1_left"].to_numpy(dtype=float)
        after = merged["answer_f1_right"].to_numpy(dtype=float)
        differences = after - before
        try:
            p_value = float(wilcoxon(differences).pvalue)
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "comparison": label, "task": "rag_answer_generation", "language": language,
                "metric": "answer_f1", "n": len(differences), "original_mean": before.mean(),
                "qlora_mean": after.mean(), "absolute_gain": differences.mean(),
                "relative_gain_pct": 100 * differences.mean() / max(abs(before.mean()), 1e-12),
                "cohen_dz": differences.mean() / max(differences.std(ddof=1), 1e-12), "p_value": p_value,
            }
        )

    for generator, frame in payloads.items():
        for language in ("zh", "en"):
            compare(
                f"{generator}: approved hybrid vs no retrieval",
                frame[frame.strategy == "no_retrieval"], frame[frame.strategy == "hybrid_rag_approved"], language,
            )
    for base, adapted in (("qwen2_5_original", "qwen2_5_qlora"), ("glm_4_original", "glm_4_qlora")):
        if base not in payloads or adapted not in payloads:
            continue
        for strategy in STRATEGIES_FOR_ANALYSIS:
            for language in ("zh", "en"):
                compare(
                    f"{adapted} vs {base}: {strategy}",
                    payloads[base][payloads[base].strategy == strategy],
                    payloads[adapted][payloads[adapted].strategy == strategy],
                    language,
                )
    for generator, frame in payloads.items():
        for strategy in STRATEGIES_FOR_ANALYSIS:
            zh = frame[(frame.strategy == strategy) & (frame.language == "zh")][["item_id", "answer_f1"]]
            en = frame[(frame.strategy == strategy) & (frame.language == "en")][["item_id", "answer_f1"]]
            merged = zh.merge(en, on="item_id", suffixes=("_zh", "_en"))
            if not merged.empty:
                rows.append(comparison_row(
                    f"English vs Chinese: {generator} {strategy}", "rag_answer_generation",
                    "en-vs-zh", "answer_f1", merged.answer_f1_zh.to_numpy(), merged.answer_f1_en.to_numpy(),
                ))
    for condition in ("original", "qlora"):
        qwen_key = f"qwen2_5_{condition}"
        glm_key = f"glm_4_{condition}"
        if qwen_key not in payloads or glm_key not in payloads:
            continue
        for strategy in STRATEGIES_FOR_ANALYSIS:
            for language in ("zh", "en"):
                qwen = payloads[qwen_key]
                glm = payloads[glm_key]
                left = glm[(glm.strategy == strategy) & (glm.language == language)][["item_id", "answer_f1"]]
                right = qwen[(qwen.strategy == strategy) & (qwen.language == language)][["item_id", "answer_f1"]]
                merged = left.merge(right, on="item_id", suffixes=("_glm", "_qwen"))
                rows.append(comparison_row(
                    f"Qwen2.5 vs GLM-4: {condition} {strategy}", "rag_answer_generation",
                    language, "answer_f1", merged.answer_f1_glm.to_numpy(), merged.answer_f1_qwen.to_numpy(),
                ))
    return pd.DataFrame(rows)


def representative_cases(samples: dict) -> pd.DataFrame:
    rows = []
    for (model_key, condition, task), frame in samples.items():
        ranked = frame.copy()
        ranked["char_f1"] = ranked.metrics.map(lambda value: float(value.get("char_f1", 0.0)))
        for outcome, selected in (("success", ranked.nlargest(2, "char_f1")),
                                  ("failure", ranked.nsmallest(2, "char_f1"))):
            for _, sample in selected.iterrows():
                rows.append({
                    "model": MODELS[model_key], "condition": condition, "task": task,
                    "outcome": outcome, "language": sample.get("language"),
                    "category": sample.get("category"), "char_f1": sample.char_f1,
                    "prompt": sample.get("prompt"), "prediction": sample.get("prediction"),
                    "reference": sample.get("reference"),
                })
    return pd.DataFrame(rows)


STRATEGIES_FOR_ANALYSIS = ("no_retrieval", "bm25_rag", "hybrid_rag_approved")


def translation_table() -> pd.DataFrame:
    rows = []
    for group, condition in (("baseline", "original"), ("qlora_eval", "qlora")):
        for model_key in MODELS:
            folder = RESULTS / group / model_key
            if not folder.exists():
                continue
            for path in folder.glob("*_domain_qa_generations.json"):
                samples = load_json(path).get("samples", [])
                groups = {
                    ("zh_to_en", "terminology"): {"terminology_zh_to_en"},
                    ("zh_to_en", "sentence"): {"zh_to_en_translation"},
                    ("en_to_zh", "terminology"): {"terminology_en_to_zh"},
                    ("en_to_zh", "sentence"): {"en_to_zh_translation"},
                }
                for (direction, subtask), categories in groups.items():
                    selected = [sample for sample in samples if sample.get("category") in categories]
                    if not selected:
                        continue
                    predictions = [sample["prediction"] for sample in selected]
                    references = [sample["reference"] for sample in selected]
                    bleu = BLEU(tokenize="13a" if direction == "zh_to_en" else "zh", effective_order=True)
                    chrf = CHRF(word_order=2)
                    for metric, score, signature in (
                        ("corpus_bleu", bleu.corpus_score(predictions, [references]).score, str(bleu.get_signature())),
                        ("chrf_pp", chrf.corpus_score(predictions, [references]).score, str(chrf.get_signature())),
                    ):
                        rows.append(
                            {
                                "model": MODELS[model_key], "model_key": model_key, "condition": condition,
                                "direction": direction, "subtask": subtask, "metric": metric, "score": score,
                                "n": len(selected), "signature": signature, "source_file": str(path.relative_to(ROOT)),
                            }
                        )
                    if subtask == "terminology":
                        rows.append(
                            {
                                "model": MODELS[model_key], "model_key": model_key, "condition": condition,
                                "direction": direction, "subtask": subtask, "metric": "terminology_success_rate",
                                "score": np.mean([float(sample["metrics"]["reference_contained"]) for sample in selected]),
                                "n": len(selected), "signature": None, "source_file": str(path.relative_to(ROOT)),
                            }
                        )
            for path in folder.glob("*_comet.json"):
                for row in load_json(path).get("summaries", []):
                    rows.append(
                        {
                            "model": MODELS[model_key], "model_key": model_key, "condition": condition,
                            "direction": row["direction"], "subtask": row.get("subtask", "combined"),
                            "metric": "COMET", "score": row["mean"],
                            "n": row["num_examples"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
                            "signature": row["model"], "source_file": str(path.relative_to(ROOT)),
                        }
                    )
    return pd.DataFrame(rows)


def efficiency_table() -> pd.DataFrame:
    rows = []
    for path in RESULTS.glob("**/*_efficiency.json"):
        payload = load_json(path)
        payload["condition"] = "qlora" if "qlora_eval" in path.parts else "original"
        payload["source_file"] = str(path.relative_to(ROOT))
        rows.append(payload)
    ollama_path = RESULTS / "efficiency/qwen3_14b_ollama.json"
    if ollama_path.exists():
        payload = load_json(ollama_path)
        rows.append(
            {
                **{key: value for key, value in payload.items() if key != "samples"},
                "condition": "reference",
                "peak_memory_reserved_gb": payload.get("peak_gpu_memory_gb"),
                "source_file": str(ollama_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def general_capability_table() -> pd.DataFrame:
    rows = []
    for group, condition in (("baseline", "original"), ("qlora_eval", "qlora")):
        for model_key in MODELS:
            folder = RESULTS / group / model_key
            for path in folder.glob("*_lm_eval.json") if folder.exists() else []:
                payload = load_json(path)
                for benchmark in ("ceval-valid", "mmlu"):
                    result = payload.get("results", {}).get(benchmark, {})
                    rows.append(
                        {
                            "model": MODELS[model_key], "condition": condition,
                            "benchmark": benchmark, "accuracy": result.get("acc,none"),
                            "standard_error": result.get("acc_stderr,none"),
                            "limit_per_task": payload.get("config", {}).get("limit"),
                            "source_file": str(path.relative_to(ROOT)),
                        }
                    )
    return pd.DataFrame(rows)


def rag_error_analysis() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "rag").glob("*.json")):
        payload = load_json(path)
        if not payload.get("summaries"):
            continue
        samples = pd.DataFrame(payload.get("rows", []))
        if samples.empty:
            continue
        for strategy in STRATEGIES_FOR_ANALYSIS:
            for language in ("zh", "en"):
                selected = samples[(samples.strategy == strategy) & (samples.language == language)]
                if selected.empty:
                    continue
                evidence_hit = selected.get("evidence_hit", pd.Series(False, index=selected.index)).astype(bool)
                answer_f1 = selected.get("answer_f1", pd.Series(0.0, index=selected.index)).astype(float)
                length_ratio = selected.get("answer_length_ratio", pd.Series(0.0, index=selected.index)).astype(float)
                citation = selected.get("citation_coverage", pd.Series(0.0, index=selected.index)).astype(float)
                flags = {
                    "retrieval_miss": (~evidence_hit) if strategy != "no_retrieval" else None,
                    "evidence_hit_but_low_answer_overlap": (
                        evidence_hit & (answer_f1 < 0.2)
                    ) if strategy != "no_retrieval" else None,
                    "overlong_answer": length_ratio > 4.0,
                    "citation_format_missing": citation == 0.0,
                }
                for error_type, mask in flags.items():
                    if mask is None:
                        continue
                    flagged = selected[mask]
                    example = None
                    if not flagged.empty:
                        item = flagged.iloc[0]
                        example = {
                            "item_id": int(item.get("item_id")), "question": item.get("question"),
                            "answer": item.get("answer"), "reference_answer": item.get("reference_answer"),
                        }
                    rows.append(
                        {
                            "generator": path.stem, "strategy": strategy, "language": language,
                            "error_type": error_type, "count": len(flagged),
                            "proportion": len(flagged) / len(selected),
                            "representative_example": json.dumps(example, ensure_ascii=False),
                            "source_file": str(path.relative_to(ROOT)),
                        }
                    )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 300, "axes.linewidth": 0.9,
    })


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves() -> None:
    colors = {"Qwen2.5-7B": "#0072B2", "GLM-4-9B": "#D55E00"}
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for model_key, label in MODELS.items():
        states = sorted((RESULTS / "qlora" / model_key / "checkpoint").glob("checkpoint-*/trainer_state.json"))
        if not states:
            continue
        history = load_json(states[-1]).get("log_history", [])
        train = [(row["step"], row["loss"]) for row in history if "loss" in row]
        if train:
            ax.plot(*zip(*train), label=f"{label} train", color=colors[label], linewidth=1.8)
        evaluation = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
        if evaluation:
            ax.scatter(*zip(*evaluation), label=f"{label} validation", color=colors[label], marker="D", s=28)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Completion loss")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "training_validation_loss")


def plot_errors(errors: pd.DataFrame) -> None:
    if errors.empty:
        return
    grouped = errors.groupby("error_type", as_index=False)["proportion"].mean().sort_values("proportion")
    grouped["label"] = grouped["error_type"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.barh(grouped["label"], 100 * grouped["proportion"], color="#009E73")
    ax.set_xlabel("Mean proportion of flagged outputs (%)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    save_figure(fig, "error_type_distribution")


def plot_pareto(metrics: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    if metrics.empty or efficiency.empty:
        return
    quality = metrics[(metrics.task == "bilingual_approved_qa") & (metrics.metric == "char_f1")].groupby(
        ["model_key", "condition"], as_index=False
    )["score"].mean()
    efficiency = efficiency.copy()
    efficiency["model_key"] = efficiency["model"]
    merged = quality.merge(efficiency, on=["model_key", "condition"])
    if merged.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55), sharey=True)
    markers = {"original": "o", "qlora": "s"}
    colors = {"qwen2_5_7b_instruct": "#0072B2", "glm_4_9b_chat_hf": "#D55E00"}
    panels = (("mean_latency_s", "Mean generation latency (s)"),
              ("peak_memory_reserved_gb", "Peak reserved GPU memory (GB)"))
    for ax, (column, xlabel) in zip(axes, panels):
        for _, row in merged.iterrows():
            ax.scatter(row[column], row["score"], marker=markers[row["condition"]],
                       color=colors[row["model_key"]], s=50)
            horizontal = -4 if row[column] == merged[column].max() else 4
            alignment = "right" if horizontal < 0 else "left"
            condition_label = "QLoRA" if row["condition"] == "qlora" else row["condition"]
            ax.annotate(f"{MODELS[row['model_key']]} {condition_label}",
                        (row[column], row["score"]), xytext=(horizontal, 4),
                        textcoords="offset points", fontsize=7.5, ha=alignment)
        ax.margins(x=0.08, y=0.1)
        ax.set_xlabel(xlabel)
        ax.grid(color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Bilingual QA Char F1")
    save_figure(fig, "quality_latency_pareto")


def plot_top_k(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        return
    selected = ablation[(ablation["mode"] == "hybrid") & (ablation["approved_only"] == True)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    styles = {"zh": ("#0072B2", "o", "-"), "en": ("#D55E00", "s", "--")}
    for language, (color, marker, line) in styles.items():
        values = selected[selected["language"] == language].sort_values("top_k")
        recall = [row[f"recall_at_{int(row['top_k'])}"] for _, row in values.iterrows()]
        axes[0].plot(values["top_k"], recall, color=color, marker=marker, linestyle=line, linewidth=1.8, label=language.upper())
        axes[1].plot(values["top_k"], values["mean_latency_ms"], color=color, marker=marker, linestyle=line, linewidth=1.8, label=language.upper())
    axes[0].set_ylabel("Recall@k")
    axes[1].set_ylabel("Mean retrieval latency (ms)")
    for axis in axes:
        axis.set_xlabel("Top-k")
        axis.set_xticks(sorted(selected["top_k"].unique()))
        axis.grid(color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False)
    save_figure(fig, "top_k_quality_latency")


def plot_translation_before_after(translation: pd.DataFrame) -> None:
    selected = translation[translation.metric == "COMET"].copy()
    if selected.empty:
        return
    selected["group"] = selected["direction"].str.replace("_", "-") + " / " + selected["subtask"]
    groups = ["zh-to-en / terminology", "zh-to-en / sentence", "en-to-zh / terminology", "en-to-zh / sentence"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), sharey=True)
    colors = {"original": "#0072B2", "qlora": "#D55E00"}
    for ax, model in zip(axes, ("Qwen2.5-7B", "GLM-4-9B")):
        frame = selected[selected.model == model]
        x = np.arange(len(groups))
        width = 0.36
        for offset, condition in ((-width / 2, "original"), (width / 2, "qlora")):
            values = [frame[(frame.group == group) & (frame.condition == condition)].score.iloc[0] for group in groups]
            ax.bar(x + offset, values, width, label=condition.capitalize(), color=colors[condition])
        ax.set_title(model)
        ax.set_xticks(x, ["ZH-EN\nterm", "ZH-EN\nsentence", "EN-ZH\nterm", "EN-ZH\nsentence"])
        ax.set_ylim(0.25, 0.8)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("COMET score")
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, "translation_before_after")


def write_table(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        return
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(frame.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


def main() -> None:
    for path in (ANALYSIS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)
    metrics, samples = aggregate_generation_metrics()
    stats = pd.concat([paired_statistics(samples), rag_paired_statistics()], ignore_index=True)
    if not stats.empty:
        order = stats["p_value"].sort_values().index.tolist()
        adjusted = pd.Series(index=stats.index, dtype=float)
        running = 0.0
        count = len(order)
        for rank, index in enumerate(order):
            running = max(running, min(1.0, stats.loc[index, "p_value"] * (count - rank)))
            adjusted.loc[index] = running
        stats["p_holm"] = adjusted
    errors = error_analysis(samples)
    retrieval = retrieval_table()
    ablation = retrieval_ablation_table()
    rag = rag_table()
    translation = translation_table()
    efficiency = efficiency_table()
    general = general_capability_table()
    rag_errors = rag_error_analysis()
    cases = representative_cases(samples)
    for frame, name in (
        (metrics, "model_task_metrics"), (stats, "paired_statistics"), (errors, "error_analysis"),
        (retrieval, "retrieval_results"), (ablation, "retrieval_ablation"),
        (rag, "rag_results"), (translation, "translation_results"),
        (efficiency, "efficiency_results"), (general, "general_capability"),
        (rag_errors, "rag_error_analysis"), (cases, "representative_cases"),
    ):
        frame.to_csv(ANALYSIS / f"{name}.csv", index=False)
    write_table(retrieval, "table3_retrieval")
    write_table(metrics[metrics.metric.isin(["char_f1", "reference_contained"])], "table4_qa_before_after")
    write_table(stats, "table4_statistical_comparisons")
    write_table(translation, "table5_directional_translation")
    write_table(rag, "table6_rag_generators")
    write_table(ablation, "table7_retrieval_ablation")
    efficiency_columns = [
        "model", "condition", "num_measurements", "mean_first_token_latency_s",
        "mean_latency_s", "std_latency_s", "mean_tokens_per_second",
        "static_memory_reserved_gb", "peak_memory_reserved_gb", "static_gpu_memory_gb",
        "peak_gpu_memory_gb", "failures", "source_file",
    ]
    write_table(efficiency[[column for column in efficiency_columns if column in efficiency]], "table8_efficiency")
    write_table(general, "table9_general_capability")
    configure_plots()
    plot_training_curves()
    plot_errors(errors)
    plot_pareto(metrics, efficiency)
    plot_top_k(ablation)
    plot_translation_before_after(translation)
    generated = sorted(path for folder in (ANALYSIS, FIGURES, TABLES) for path in folder.glob("*"))
    source_frames = (metrics, stats, retrieval, ablation, rag, translation, efficiency, general, rag_errors)
    sources = sorted({str(value) for frame in source_frames if "source_file" in frame
                      for value in frame.source_file.dropna().unique()})
    manifest = {
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in generated if path.name != "asset_manifest.json"
        ],
        "source_files": sources,
        "source_generation_files": [str(path.relative_to(ROOT)) for _, _, path in generation_files()],
    }
    (ANALYSIS / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": len(metrics), "statistics": len(stats), "errors": len(errors), "rag": len(rag)}, indent=2))


if __name__ == "__main__":
    main()
