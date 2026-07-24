from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import CorpusItem


SPLITS = ("train", "valid", "test")


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def declared_split(item: CorpusItem) -> str:
    return str((item.metadata_json or {}).get("split", ""))


def select_balanced_items(items: list[CorpusItem], max_pairs_per_task: int, seed: int) -> list[CorpusItem]:
    """Cap dominant tasks while always retaining the pre-declared RAG test items."""
    by_task: dict[str, list[CorpusItem]] = defaultdict(list)
    for item in items:
        by_task[item.task_type].append(item)

    selected: list[CorpusItem] = []
    for task_type, task_items in sorted(by_task.items()):
        fixed_test = [item for item in task_items if declared_split(item) == "test"]
        candidates = [item for item in task_items if declared_split(item) != "test"]
        random.Random(stable_seed(seed, "select", task_type)).shuffle(candidates)
        limit = max(max_pairs_per_task, len(fixed_test))
        selected.extend(fixed_test)
        selected.extend(candidates[: max(0, limit - len(fixed_test))])
    return selected


def stratified_split(items: list[CorpusItem], seed: int) -> dict[str, list[CorpusItem]]:
    """Split knowledge pairs 80/10/10 within task and source strata."""
    strata: dict[tuple[str, str], list[CorpusItem]] = defaultdict(list)
    for item in items:
        strata[(item.task_type, item.source_document)].append(item)

    output: dict[str, list[CorpusItem]] = {split: [] for split in SPLITS}
    for (task_type, source_document), stratum in sorted(strata.items()):
        fixed_test = [item for item in stratum if declared_split(item) == "test"]
        candidates = [item for item in stratum if declared_split(item) != "test"]
        random.Random(stable_seed(seed, "split", task_type, source_document)).shuffle(candidates)

        total = len(stratum)
        target_test = max(round(total * 0.1), len(fixed_test))
        target_valid = round(total * 0.1)
        extra_test = max(0, target_test - len(fixed_test))
        test_items = fixed_test + candidates[:extra_test]
        remaining = candidates[extra_test:]
        valid_items = remaining[: min(target_valid, len(remaining))]
        train_items = remaining[len(valid_items) :]

        output["train"].extend(train_items)
        output["valid"].extend(valid_items)
        output["test"].extend(test_items)
    return output


def make_record(item: CorpusItem, language: str, split: str) -> dict:
    question = item.question if language == "zh" else item.question_en
    answer = item.answer if language == "zh" else item.answer_en
    question_label = "问题" if language == "zh" else "Question"
    answer_label = "回答" if language == "zh" else "Answer"
    prefix = f"{question_label}: {question}\n{answer_label}:"
    return {
        "id": f"{item.external_id}:{language}",
        "pair_id": item.external_id,
        "split": split,
        "language": language,
        "task_type": item.task_type,
        "source_document": item.source_document,
        "prompt": question,
        "answer": answer,
        "prompt_text": prefix,
        "text": f"{prefix} {answer}",
        "review_status": item.review_status,
    }


def pairwise_overlaps(output: dict[str, list[dict]]) -> dict[str, int]:
    pairs = {split: {row["pair_id"] for row in rows} for split, rows in output.items()}
    return {
        "train_valid": len(pairs["train"] & pairs["valid"]),
        "train_test": len(pairs["train"] & pairs["test"]),
        "valid_test": len(pairs["valid"] & pairs["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an approved, group-stratified bilingual QLoRA corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/qlora_bilingual_approved"))
    parser.add_argument("--rag-test-output", type=Path, default=Path("data/rag_eval/regulation_test_120.jsonl"))
    parser.add_argument("--max-pairs-per-task", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with SessionLocal() as db:
        approved = list(
            db.scalars(
                select(CorpusItem)
                .where(CorpusItem.review_status == "approved")
                .order_by(CorpusItem.id)
            )
        )

    eligible = [item for item in approved if item.question and item.answer and item.question_en and item.answer_en]
    selected = select_balanced_items(eligible, args.max_pairs_per_task, args.seed)
    split_items = stratified_split(selected, args.seed)

    output: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for split, items in split_items.items():
        for item in items:
            output[split].extend(make_record(item, language, split) for language in ("zh", "en"))
        random.Random(stable_seed(args.seed, "output", split)).shuffle(output[split])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in output.items():
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    fixed_rag_items = [item for item in eligible if declared_split(item) == "test"]
    rag_records = [
        make_record(item, language, "rag_regulation_test")
        for item in fixed_rag_items
        for language in ("zh", "en")
    ]
    args.rag_test_output.parent.mkdir(parents=True, exist_ok=True)
    args.rag_test_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rag_records), encoding="utf-8"
    )

    total_records = sum(len(records) for records in output.values())
    stats = {
        "seed": args.seed,
        "split_method": "pair-grouped, task-and-source-stratified 80/10/10; declared test items forced to test",
        "max_pairs_per_task": args.max_pairs_per_task,
        "source_approved_items": len(approved),
        "eligible_complete_bilingual_pairs": len(eligible),
        "selected_pairs": len(selected),
        "excluded": {
            "incomplete_bilingual_pairs": len(approved) - len(eligible),
            "task_cap_pairs": len(eligible) - len(selected),
        },
        "fixed_rag_regulation_test_pairs": len(fixed_rag_items),
        "counts": {
            split: {
                "records": len(records),
                "record_ratio": len(records) / total_records if total_records else 0.0,
                "pairs": len({row["pair_id"] for row in records}),
                "by_language": dict(Counter(row["language"] for row in records)),
                "by_task": dict(Counter(row["task_type"] for row in records)),
                "source_documents": len({row["source_document"] for row in records}),
            }
            for split, records in output.items()
        },
        "pair_overlap": pairwise_overlaps(output),
        "rag_regulation_test_records": len(rag_records),
    }
    (args.output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
