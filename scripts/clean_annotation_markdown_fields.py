from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem, ReviewEvent  # noqa: E402


def clean_question_answer_text(text: str) -> str:
    text = text or ""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        while line.startswith("#") and (len(line) == 1 or line[1] == " " or line[1] == "#"):
            line = line[1:].lstrip()
        if line and set(line) == {"#"}:
            line = ""
        line = re.sub(r"#{2,}", "", line)
        line = re.sub(r"(^|\s)#{1,6}(?=\s)", r"\1", line).strip()
        line = re.sub(r"([：:，,。；;])\s+(?=[\u4e00-\u9fff一二三四五六七八九十百\d(（])", r"\1", line)
        line = line.replace("关于 ", "关于")
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def remove_leading_order_marker(text: str) -> str:
    text = text or ""
    patterns = [
        r"^\s*(第[一二三四五六七八九十百千万\d]+[个条题项点、.．:：)）]\s*)+",
        r"^\s*([（(]?[一二三四五六七八九十百千万\d]+[)）、.．:：]\s*)+",
        r"^\s*([A-Da-d][).．:：]\s+)+",
    ]
    previous = None
    while previous != text:
        previous = text
        for pattern in patterns:
            text = re.sub(pattern, "", text).lstrip()
    return text.strip()


def clean_question_answer_value(text: str) -> str:
    return remove_leading_order_marker(clean_question_answer_text(text))


def normalize_numeric_slashes(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"(?<![\w/])(\d+(?:/\d+){1,})(?![\w/])", lambda m: m.group(1).replace("/", "／"), text)


def normalize_evidence_sources(metadata: dict) -> tuple[dict, bool]:
    metadata = dict(metadata or {})
    evidence_sources = metadata.get("evidence_sources")
    if not isinstance(evidence_sources, dict):
        return metadata, False
    changed = False
    normalized_sources = dict(evidence_sources)
    for key, payload in list(normalized_sources.items()):
        if not isinstance(payload, dict):
            continue
        next_payload = dict(payload)
        for field in ("evidence", "context"):
            old_value = str(next_payload.get(field) or "")
            new_value = normalize_numeric_slashes(old_value)
            if new_value != old_value:
                next_payload[field] = new_value
                changed = True
        normalized_sources[key] = next_payload
    if changed:
        metadata["evidence_sources"] = normalized_sources
    return metadata, changed


def snapshot(item: CorpusItem) -> dict[str, str | int]:
    return {
        "question": item.question,
        "answer": item.answer,
        "evidence": item.evidence,
        "metadata_json": item.metadata_json,
        "version": item.version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Markdown markers from annotation question/answer fields.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reviewer", default="system")
    args = parser.parse_args()

    load_env()
    updated = 0
    unchanged = 0

    with SessionLocal() as db:
        statement = (
            select(CorpusItem)
            .where(CorpusItem.review_status != "deleted")
            .order_by(CorpusItem.id)
        )
        if args.limit:
            statement = statement.limit(args.limit)

        for item in db.scalars(statement):
            new_question = clean_question_answer_value(item.question)
            new_answer = clean_question_answer_value(item.answer)
            new_evidence = normalize_numeric_slashes(item.evidence or "")
            new_metadata, metadata_changed = normalize_evidence_sources(item.metadata_json)
            if (
                new_question == item.question
                and new_answer == item.answer
                and new_evidence == item.evidence
                and not metadata_changed
            ):
                unchanged += 1
                continue

            before = snapshot(item)
            if args.dry_run:
                print(
                    {
                        "id": item.id,
                        "question_before": item.question,
                        "question_after": new_question,
                        "answer_before": item.answer,
                        "answer_after": new_answer,
                        "evidence_changed": new_evidence != item.evidence,
                        "metadata_changed": metadata_changed,
                    }
                )
                updated += 1
                continue

            item.question = new_question
            item.answer = new_answer
            item.evidence = new_evidence
            if metadata_changed:
                item.metadata_json = new_metadata
            item.version += 1
            db.add(
                ReviewEvent(
                    item=item,
                    action="clean_markdown_fields",
                    reviewer=args.reviewer,
                    comment="Removed Markdown formatting markers from question and answer.",
                    snapshot={"before": before, "after": snapshot(item)},
                )
            )
            db.commit()
            updated += 1

    print({"updated": updated, "unchanged": unchanged, "dry_run": args.dry_run})


if __name__ == "__main__":
    main()
