"""Rebuild strict, source-grounded textbook concept explanation QA records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "annotation_system" / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.database import SessionLocal  # noqa: E402
from app.import_data import insert_items, stable_id, upsert_document  # noqa: E402
from app.models import CorpusItem  # noqa: E402
import expand_current_non_terminology_corpus as source  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "data" / "textbook_concept_qa_v2"
GENERATION_METHOD = "rebuilt_textbook_concept_v2"
QUESTION_SUFFIXES = (
    "有哪些特点？",
    "有什么作用？",
    "用于哪些场景或位置？",
    "有哪些功能？",
    "包括哪些内容？",
    "分为哪些类型？",
    "由哪些部分组成？",
    "是什么？",
)
BAD_START_RE = re.compile(r"^(其中|同时|另外|但|而|则|因此|由于|虽然|在|当|若|如果|这种|该|其)")
BAD_TOPIC_RE = re.compile(
    r"^(?:现在|目前|一头|另外|虽然|其中|同时|这种|该|其|本|此|所有|任何|根据|依据|按照|随着|将|主要|高铁目前)|"
    r"(完成|采用|设定|输入|交清|运用|根据|目前|一般)|"
    r"(?:特点|主要|一般|又|多|中)$"
)
MARKUP_RE = re.compile(r"(#{1,6}|<table|</table>|!\[|<\|ref\|>|<\|det\|>)", re.I)
INCOMPLETE_RE = re.compile(r"(如下|下列|包括以下|分为下列|如图所示|如表所示|见图|见表)[：:。；;]*$")


def topic_from_question(question: str) -> str:
    for suffix in QUESTION_SUFFIXES:
        if question.endswith(suffix):
            return question[: -len(suffix)].strip()
    return ""


def rebuild_question(row: dict) -> str | None:
    answer = source.normalize(row.get("answer", ""))
    original = source.normalize(row.get("question", ""))
    named_match = re.search(r"^(?:在|当).{2,40}?称为([^,，。；;]{2,20})", answer)
    if named_match:
        named_topic = named_match.group(1).strip(" ，,;；：:。！？（）()")
        if 2 <= len(named_topic) <= 20 and not BAD_TOPIC_RE.search(named_topic):
            return f"{named_topic}是什么？"
    patterns = [
        (r"^(.{2,36}?)(?:的)?特点是", "characteristic"),
        (r"^(.{2,36}?)，具有.{0,24}(?:特点|优点)", "characteristic"),
        (r"^(.{2,36}?)(?:的|其|主要)?作用(?:是|为|在于)", "function"),
        (r"^(.{2,20}?)通过.{0,60}(?:其|主要)?作用(?:是|为|在于)", "function"),
        (r"^(.{2,36}?)(?:的|主要)?功能(?:是|包括|为)", "feature"),
        (r"^(?:所谓)?(.{2,36}?)(?:也称为|又称为|称为|简称为?|是指)", "definition"),
        (r"^(.{2,28}?)(?:主要)?包括", "contents"),
        (r"^(.{2,28}?)(?:可|可以)?分为", "types"),
        (r"^(.{2,28}?)(?:是)?由.{2,120}组成", "composition"),
        (r"^(.{2,30}?)(?<!不)用于", "usage"),
    ]
    for pattern, intent in patterns:
        match = re.search(pattern, answer)
        if not match:
            continue
        topic = source.textbook_source.strip_leading_numbering(match.group(1))
        topic = topic.strip(" ，,;；：:。！？（）()")
        topic = re.sub(r"[（(][^）)]*[）)]", "", topic).strip()
        if "与" not in topic or "相比" not in topic:
            topic = re.split(r"(?:，|,|是|为|通过)", topic, maxsplit=1)[0].strip()
        topic = re.sub(r"的作用$", "", topic).strip()
        topic = re.sub(r"的$", "", topic).strip()
        if not (2 <= len(topic) <= 28) or BAD_TOPIC_RE.search(topic):
            break
        suffixes = {
            "definition": "是什么？",
            "characteristic": "有哪些特点？",
            "function": "有什么作用？",
            "usage": "用于哪些场景或位置？",
            "feature": "有哪些功能？",
            "contents": "包括哪些内容？",
            "types": "分为哪些类型？",
            "composition": "由哪些部分组成？",
        }
        return topic + suffixes[intent]
    return None


def han_bigrams(text: str) -> set[str]:
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    return {han[index : index + 2] for index in range(max(0, len(han) - 1))}


def quality_reason(row: dict) -> str | None:
    question = source.normalize(row.get("question", ""))
    answer = source.normalize(row.get("answer", ""))
    topic = topic_from_question(question)
    if not topic or topic == "该知识点" or not (2 <= len(topic) <= 28):
        return "invalid_topic"
    if BAD_TOPIC_RE.search(topic):
        return "invalid_topic"
    if topic.count("（") != topic.count("）") or topic.count("(") != topic.count(")"):
        return "invalid_topic"
    if re.search(r"[\dA-Za-z]", topic) or re.search(r"[，,;；：:]", topic):
        return "invalid_topic"
    if not (28 <= len(answer) <= 320):
        return "invalid_answer_length"
    if "\n" in row.get("answer", ""):
        return "multiline_answer"
    if not answer.endswith(("。", "！", "？")):
        return "incomplete_sentence"
    if BAD_START_RE.search(answer):
        return "context_dependent_start"
    if MARKUP_RE.search(answer) or source.CROSS_REFERENCE_RE.search(answer) or INCOMPLETE_RE.search(answer):
        return "markup_or_reference"
    if topic not in answer:
        topic_pairs = han_bigrams(topic)
        answer_pairs = han_bigrams(answer[:160])
        if not topic_pairs or len(topic_pairs & answer_pairs) / len(topic_pairs) < 0.5:
            return "topic_answer_mismatch"
    if question.endswith("有哪些特点？") and not re.search(r"(特点|具有|优点|特性)", answer):
        return "unsupported_characteristic_question"
    if question.endswith("有什么作用？") and not re.search(r"(作用|用于|目的|可以|能够)", answer):
        return "unsupported_function_question"
    if question.endswith("有哪些功能？") and "功能" not in answer:
        return "unsupported_feature_question"
    if question.endswith("用于哪些场景或位置？") and "用于" not in answer:
        return "unsupported_usage_question"
    if question.endswith("包括哪些内容？") and "包括" not in answer:
        return "unsupported_contents_question"
    if question.endswith("分为哪些类型？") and "分为" not in answer:
        return "unsupported_types_question"
    if question.endswith("由哪些部分组成？") and not re.search(r"由.{2,120}组成", answer):
        return "unsupported_composition_question"
    return None


def build_candidates() -> tuple[list[dict], dict[str, int]]:
    accepted_by_question: dict[str, dict] = {}
    rejected: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in source.iter_textbook_base_rows():
        question = rebuild_question(row)
        if not question:
            continue
        row = {**row, "task_type": "concept_explanation_qa", "question": question}
        reason = quality_reason(row)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        key = (row["question"], row["answer"])
        if key in seen:
            continue
        seen.add(key)
        candidate = {**row, "generation_method": GENERATION_METHOD}
        current = accepted_by_question.get(row["question"])
        if current is None or len(row["answer"]) > len(current["answer"]):
            accepted_by_question[row["question"]] = candidate
    return list(accepted_by_question.values()), rejected


def write_outputs(rows: list[dict], rejected: dict[str, int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "answer", "source", "page_number", "line_number", "source_text"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    summary = {
        "generation_method": GENERATION_METHOD,
        "accepted": len(rows),
        "rejected_by_rule": rejected,
        "rules": [
            "single complete sentence only",
            "no figure/table/cross-reference text",
            "no markdown or section headings",
            "question topic must align with answer",
            "question intent must be explicitly supported",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def import_rows(rows: list[dict]) -> int:
    with SessionLocal() as db:
        documents = {
            path: upsert_document(
                db,
                title=path.parent.name,
                source_path=str(path.relative_to(REPO_ROOT)),
                document_type="textbook",
                domain_category=source.DOMAIN_CATEGORY,
                total_pages=None,
                metadata_json={"concept_qa_version": "v2"},
            )
            for path in source.textbook_source.BOOK_PATHS
        }
        db_rows = []
        for row in rows:
            source_path = row["source_path"]
            external_id = stable_id(
                "textbook_concept_v2",
                source_path,
                str(row.get("line_number") or ""),
                row["question"],
                row["answer"],
            )
            db_rows.append(
                {
                    "external_id": external_id,
                    "source_type": "textbook_original_md",
                    "task_type": "concept_explanation_qa",
                    "review_status": "pending",
                    "domain_category": source.DOMAIN_CATEGORY,
                    "knowledge_category": "教材",
                    "question": row["question"],
                    "answer": row["answer"],
                    "evidence": row["answer"],
                    "source_text": row["source_text"],
                    "original_question": row["question"],
                    "original_answer": row["answer"],
                    "document_id": documents[REPO_ROOT / source_path].id,
                    "source_document": row["source"],
                    "source_path": source_path,
                    "chapter": row.get("chapter", ""),
                    "page_number": row.get("page_number"),
                    "quality_flags": ["human_review_required", "rebuilt_v2"],
                    "metadata_json": {
                        "generation_method": GENERATION_METHOD,
                        "line_number": row.get("line_number"),
                        "replaces_generation_method": source.TARGET_METHOD if hasattr(source, "TARGET_METHOD") else "expanded_current_textbook_long_context",
                    },
                }
            )
        inserted = insert_items(db, db_rows)
        db.commit()
        return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild strict textbook concept explanation QA.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows, rejected = build_candidates()
    write_outputs(rows, rejected)
    inserted = 0 if args.dry_run else import_rows(rows)
    print(json.dumps({"accepted": len(rows), "inserted": inserted, "rejected_by_rule": rejected}, ensure_ascii=False, indent=2))
    for row in rows[:20]:
        print(f"Q: {row['question']}\nA: {row['answer']}\n")


if __name__ == "__main__":
    main()
