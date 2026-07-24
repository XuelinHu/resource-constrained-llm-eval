from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import CorpusItem, KnowledgeChunkEmbedding


def category(task_type: str) -> str:
    if task_type.startswith("terminology_"):
        return "terminology"
    if task_type.startswith("regulation_"):
        return "regulation"
    return "textbook"


def stable_order(item: CorpusItem, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{item.external_id}".encode("utf-8")).hexdigest()


def round_robin_sample(items: list[CorpusItem], count: int, seed: int) -> list[CorpusItem]:
    groups: dict[str, deque[CorpusItem]] = {}
    by_task: dict[str, list[CorpusItem]] = defaultdict(list)
    for item in items:
        by_task[item.task_type].append(item)
    for task_type, rows in by_task.items():
        groups[task_type] = deque(sorted(rows, key=lambda item: stable_order(item, seed)))

    selected: list[CorpusItem] = []
    task_types = sorted(groups)
    while len(selected) < count and task_types:
        remaining = []
        for task_type in task_types:
            if groups[task_type] and len(selected) < count:
                selected.append(groups[task_type].popleft())
            if groups[task_type]:
                remaining.append(task_type)
        task_types = remaining
    return selected


def allocate_targets(available: Counter[str], requested: dict[str, int], total: int) -> dict[str, int]:
    targets = {name: min(requested.get(name, 0), available[name]) for name in available}
    deficit = total - sum(targets.values())
    for name in ("terminology", "regulation", "textbook"):
        addition = min(deficit, available[name] - targets.get(name, 0))
        targets[name] = targets.get(name, 0) + addition
        deficit -= addition
    if deficit:
        raise ValueError(f"Only {total - deficit} eligible pairs are available; requested {total}.")
    return targets


def export_record(item: CorpusItem, language: str, test_set: str) -> dict:
    return {
        "item_id": item.id,
        "pair_id": item.external_id,
        "test_set": test_set,
        "language": language,
        "task_type": item.task_type,
        "domain_category": item.domain_category,
        "source_document": item.source_document,
        "page_number": item.page_number,
        "question": item.question if language == "zh" else item.question_en,
        "answer": item.answer if language == "zh" else item.answer_en,
        "evidence": item.evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a cross-source bilingual RAG test set.")
    parser.add_argument("--qlora-test", type=Path, default=Path("data/qlora_bilingual_approved/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/rag_eval/railway_bilingual_400.jsonl"))
    parser.add_argument("--statistics", type=Path, default=Path("data/rag_eval/railway_bilingual_400_statistics.json"))
    parser.add_argument("--test-set", default="railway_bilingual_400")
    parser.add_argument("--total-pairs", type=int, default=400)
    parser.add_argument("--terminology", type=int, default=100)
    parser.add_argument("--regulation", type=int, default=150)
    parser.add_argument("--textbook", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate_ids = {
        json.loads(line)["pair_id"]
        for line in args.qlora_test.open(encoding="utf-8")
        if line.strip()
    }
    with SessionLocal() as db:
        candidates = list(
            db.scalars(
                select(CorpusItem)
                .where(
                    CorpusItem.external_id.in_(candidate_ids),
                    CorpusItem.review_status == "approved",
                    CorpusItem.question != "",
                    CorpusItem.answer != "",
                    CorpusItem.question_en != "",
                    CorpusItem.answer_en != "",
                    CorpusItem.evidence != "",
                )
                .order_by(CorpusItem.id)
            )
        )

        available = Counter(category(item.task_type) for item in candidates)
        targets = allocate_targets(
            available,
            {"terminology": args.terminology, "regulation": args.regulation, "textbook": args.textbook},
            args.total_pairs,
        )
        fixed = [item for item in candidates if (item.metadata_json or {}).get("split") == "test"]
        fixed_ids = {item.id for item in fixed}
        selected = list(fixed)
        for name, target in targets.items():
            fixed_category = [item for item in fixed if category(item.task_type) == name]
            pool = [item for item in candidates if category(item.task_type) == name and item.id not in fixed_ids]
            needed = target - len(fixed_category)
            if needed < 0:
                raise ValueError(f"Fixed test items exceed the {name} target: {len(fixed_category)} > {target}")
            selected.extend(round_robin_sample(pool, needed, args.seed))

        selected_ids = {item.id for item in selected}
        if len(selected_ids) != args.total_pairs:
            raise ValueError(f"Selected {len(selected_ids)} unique pairs instead of {args.total_pairs}.")

        previously_tagged = list(
            db.scalars(select(CorpusItem).where(CorpusItem.metadata_json["rag_test_set"].as_string() == args.test_set))
        )
        for item in previously_tagged:
            metadata = dict(item.metadata_json or {})
            metadata.pop("rag_test_set", None)
            item.metadata_json = metadata
        for item in selected:
            metadata = dict(item.metadata_json or {})
            metadata["rag_test_set"] = args.test_set
            item.metadata_json = metadata

        deleted_embeddings = db.execute(
            delete(KnowledgeChunkEmbedding).where(KnowledgeChunkEmbedding.item_id.in_(selected_ids))
        ).rowcount
        db.commit()

    selected.sort(key=lambda item: item.id)
    records = [export_record(item, language, args.test_set) for item in selected for language in ("zh", "en")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    stats = {
        "test_set": args.test_set,
        "seed": args.seed,
        "pairs": len(selected),
        "records": len(records),
        "requested_category_targets": {
            "terminology": args.terminology,
            "regulation": args.regulation,
            "textbook": args.textbook,
        },
        "effective_category_targets": targets,
        "available_test_pairs": dict(available),
        "fixed_regulation_pairs": len(fixed),
        "by_category": dict(Counter(category(item.task_type) for item in selected)),
        "by_task": dict(Counter(item.task_type for item in selected)),
        "by_source": dict(Counter(item.source_document for item in selected)),
        "deleted_embeddings": deleted_embeddings,
    }
    args.statistics.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
