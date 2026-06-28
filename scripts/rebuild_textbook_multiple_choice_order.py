"""Reorder all textbook multiple-choice options and balance answer labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "annotation_system" / "backend"))

from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "data" / "textbook_multiple_choice_v2"
OLD_METHOD = "expanded_current_textbook_multiple_choice"
NEW_METHOD = "rebuilt_textbook_multiple_choice_v2"
OPTION_RE = re.compile(r"^([ABCD])\.\s*(.+)$")
ANSWER_RE = re.compile(r"^([ABCD])\.\s*(.+)$", re.S)
LABELS = "ABCD"


def parse_item(item: CorpusItem) -> tuple[str, list[str], str]:
    stem_lines: list[str] = []
    options: dict[str, str] = {}
    for raw_line in (item.question or "").splitlines():
        line = raw_line.strip()
        option_match = OPTION_RE.match(line)
        if option_match:
            options[option_match.group(1)] = option_match.group(2).strip()
        elif not options:
            stem_lines.append(raw_line.rstrip())
        else:
            raise ValueError(f"item {item.id}: unexpected text after options: {line}")
    answer_match = ANSWER_RE.match((item.answer or "").strip())
    if len(options) != 4 or not answer_match:
        raise ValueError(f"item {item.id}: invalid question or answer format")
    correct_text = answer_match.group(2).strip()
    if correct_text != options.get(answer_match.group(1)):
        raise ValueError(f"item {item.id}: answer text does not match labeled option")
    return "\n".join(stem_lines).strip(), [options[label] for label in LABELS], correct_text


def rebuild(rows: list[CorpusItem], seed: int) -> list[dict]:
    target_labels = list(LABELS) * (len(rows) // 4)
    target_labels.extend(LABELS[: len(rows) % 4])
    random.Random(seed).shuffle(target_labels)

    output: list[dict] = []
    for item, correct_label in zip(rows, target_labels, strict=True):
        stem, options, correct_text = parse_item(item)
        distractors = [option for option in options if option != correct_text]
        random.Random(f"{seed}:{item.external_id}").shuffle(distractors)
        ordered = list(distractors)
        ordered.insert(LABELS.index(correct_label), correct_text)
        question = stem + "\n" + "\n".join(
            f"{label}. {option}" for label, option in zip(LABELS, ordered, strict=True)
        )
        answer = f"{correct_label}. {correct_text}"
        output.append(
            {
                "id": item.id,
                "external_id": item.external_id,
                "question": question,
                "answer": answer,
                "correct_label": correct_label,
                "correct_text": correct_text,
                "options": dict(zip(LABELS, ordered, strict=True)),
                "source_document": item.source_document,
                "page_number": item.page_number,
            }
        )
    return output


def write_outputs(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "correct_label", "question", "answer", "source_document", "page_number"])
        for row in rows:
            writer.writerow(
                [row["id"], row["correct_label"], row["question"], row["answer"], row["source_document"], row["page_number"]]
            )
    summary = {
        "generation_method": NEW_METHOD,
        "total": len(rows),
        "seed": 20260624,
        "answer_label_distribution": dict(sorted(Counter(row["correct_label"] for row in rows).items())),
        "invariants": [
            "each question has exactly four options",
            "correct answer text is unchanged",
            "answer label matches the shuffled option",
            "A/B/C/D labels are balanced",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_rows(rows: list[dict]) -> int:
    with SessionLocal() as db:
        by_id = {item.id: item for item in db.scalars(select(CorpusItem).where(CorpusItem.id.in_([row["id"] for row in rows])))}
        for row in rows:
            item = by_id[row["id"]]
            metadata = dict(item.metadata_json or {})
            metadata.update(
                {
                    "previous_generation_method": metadata.get("generation_method", OLD_METHOD),
                    "generation_method": NEW_METHOD,
                    "shuffle_seed": 20260624,
                    "correct_label": row["correct_label"],
                }
            )
            item.question = row["question"]
            item.answer = row["answer"]
            item.metadata_json = metadata
            item.quality_flags = sorted(set(item.quality_flags or []) | {"human_review_required", "options_shuffled_v2"})
            item.version += 1
        db.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild textbook MCQ option order.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with SessionLocal() as db:
        items = list(
            db.scalars(
                select(CorpusItem)
                .where(
                    CorpusItem.task_type == "textbook_multiple_choice",
                    CorpusItem.metadata_json["generation_method"].astext == OLD_METHOD,
                )
                .order_by(CorpusItem.id)
            )
        )
    if len(items) != 500:
        raise ValueError(f"expected 500 source records, found {len(items)}")
    rows = rebuild(items, seed=20260624)
    write_outputs(rows)
    updated = 0 if args.dry_run else apply_rows(rows)
    print(json.dumps({"total": len(rows), "updated": updated, "labels": dict(Counter(row["correct_label"] for row in rows))}, ensure_ascii=False, indent=2))
    for row in rows[:8]:
        print(f"\nID {row['id']}\n{row['question']}\nANSWER: {row['answer']}")


if __name__ == "__main__":
    main()
