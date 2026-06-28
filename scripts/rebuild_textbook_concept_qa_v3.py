"""Rebuild textbook concept QA with context-resolved subjects."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import chain
from pathlib import Path

from sqlalchemy import delete, select


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "annotation_system" / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.database import SessionLocal  # noqa: E402
from app.import_data import insert_items, stable_id, upsert_document  # noqa: E402
from app.models import CorpusItem, ReviewEvent  # noqa: E402
import expand_current_non_terminology_corpus as source  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "data" / "textbook_concept_qa_v3"
GENERATION_METHOD = "rebuilt_textbook_concept_v3"
TARGET_COUNT = 80

CROSS_REFERENCE_RE = re.compile(r"(如图|见图|如下图|见表|如表|图\s*\d|表\s*\d|所示)")
MARKUP_RE = re.compile(r"(<table|</table>|!\[|<\|ref\|>|<\|det\|>)", re.I)
BAD_TOPIC_RE = re.compile(r"^(其|它|该|这种|此|上述|有关|相关|主要|一般|其中|同时|另外|以下|如下)")
FIGURE_LINE_RE = re.compile(r"(图\s*\d|表\s*\d|如图|见图|如表|见表|<table|</table>|!\[)", re.I)
LINE_PREFIX_RE = re.compile(r"^\s*L(?P<line>\d+)\s*\|\s*第\s*(?P<page>\d+)\s*页\s*\|\s*(?P<text>.*)$")
GENERIC_TOPIC_RE = re.compile(
    r"^(观察|监控对象|可靠性|传输介质|报警信息|维修项目|天窗|护层|不均匀度|试运行申请报告|"
    r"每组.*抢修车列|车站、段负荷|区间负荷)$"
)
QUESTION_SUFFIX = {
    "definition": "是什么？",
    "function": "有什么作用？",
    "contents": "包括哪些内容？",
    "composition": "由哪些部分组成？",
    "types": "分为哪些类型？",
    "usage": "用于哪些场景或位置？",
    "characteristic": "有哪些特点？",
}


def normalize(text: str) -> str:
    text = source.normalize(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sentence(text: str) -> str:
    text = normalize(text)
    text = re.sub(r"^[0-9一二三四五六七八九十]+[.、）)]\s*", "", text)
    text = re.sub(r"^[（(][0-9一二三四五六七八九十]+[）)]\s*", "", text)
    text = re.sub(r"\s*如图\s*[\d\-—–.]+.*?所示[。；;]?", "。", text)
    text = re.sub(r"\s*见表\s*[\d\-—–.]+[。；;]?", "。", text)
    text = re.sub(r"\s*见图\s*[\d\-—–.]+[。；;]?", "。", text)
    text = re.sub(r"\s*如下图.*?[。；;]?", "。", text)
    text = text.replace("。。", "。").strip(" ，,；;：:")
    if text and not text.endswith(("。", "？", "！")):
        text += "。"
    return text


def context_lines(source_text: str) -> list[dict]:
    rows: list[dict] = []
    for raw in (source_text or "").splitlines():
        match = LINE_PREFIX_RE.match(raw)
        if match:
            text = normalize(match.group("text"))
            rows.append(
                {
                    "line": int(match.group("line")),
                    "page": int(match.group("page")),
                    "text": text,
                    "raw": raw,
                }
            )
        elif raw.strip():
            rows.append({"line": None, "page": None, "text": normalize(raw), "raw": raw})
    return rows


def strip_topic(topic: str) -> str:
    topic = source.textbook_source.strip_leading_numbering(topic)
    topic = re.split(r"[（(]", topic, maxsplit=1)[0]
    topic = topic.strip(" ，,。；;：:（）()")
    topic = re.sub(r"^[0-9一二三四五六七八九十]+[.、）)]\s*", "", topic)
    topic = re.sub(r"为铸造铝合金.*$", "", topic)
    topic = re.sub(r"为[^，,。；;]{2,20}$", "", topic)
    topic = re.sub(r"^(?:所谓|其中|同时|这种|该|其)", "", topic)
    topic = re.sub(r"(?:的作用|的功能|的组成|的主要内容)$", "", topic)
    return topic.strip(" ，,。；;：:（）()")


def infer_scope(text: str) -> str:
    source_text = normalize(text)
    rules = [
        (r"接触网.*抢修|抢修车列|故障抢修|应急处置", "接触网抢修"),
        (r"接触网.*运行管理|巡视|观察|检测|鉴定", "接触网设备运行管理"),
        (r"支撑定位装置|腕臂|定位器|定位管|防风拉线|吊线固定钩|旋转双耳", "接触网支撑定位装置"),
        (r"远动|SCADA|遥信|遥测|遥调|信道|传输介质|报警信息", "牵引供电远动系统"),
        (r"27\\.5\\s*kV.*电缆|电气化铁路.*电缆|电缆", "电气化铁路电缆"),
        (r"变配电所|配电所|母线|调压变压器|二次回路", "铁路变配电所"),
        (r"箱变|智能箱变", "铁路电力远动智能箱变"),
        (r"负荷|贯通线|配电系统", "铁路配电系统"),
    ]
    for pattern, scope in rules:
        if re.search(pattern, source_text):
            return scope
    return ""


def infer_topic_and_intent(answer: str, source_text: str) -> tuple[str, str] | None:
    answer = clean_sentence(answer)
    patterns = [
        (r"^(.{2,42}?)[：:]\s*由.{2,160}组成", "composition"),
        (r"^(.{2,42}?)[：:]\s*(?:是指|指|为)", "definition"),
        (r"^(.{2,42}?)[：:]\s*用于", "usage"),
        (r"^(.{2,42}?)(?:是指|，是指|是用来|，是用来|又称为|也称为|简称为)", "definition"),
        (r"^(.{2,42}?)为.{0,80}?(?:其|主要)?作用(?:是|为|在于)", "function"),
        (r"^(.{2,42}?)(?:的|主要)?作用(?:是|为|在于)", "function"),
        (r"^(.{2,42}?)起到.{0,28}作用", "function"),
        (r"^(.{2,42}?)(?:主要)?包括", "contents"),
        (r"^(.{2,42}?)(?:是)?由.{2,160}组成", "composition"),
        (r"^(.{2,42}?)(?:按.{1,18})?分为", "types"),
        (r"^(.{2,42}?)(?:一般)?(?<!不)用于", "usage"),
        (r"^(.{2,42}?)(?:的)?特点是", "characteristic"),
        (r"^(.{2,42}?)，具有.{0,36}(?:特点|优点|优势)", "characteristic"),
    ]
    for pattern, intent in patterns:
        match = re.search(pattern, answer)
        if not match:
            continue
        topic = strip_topic(match.group(1))
        topic = re.split(r"(?:，|,|是|通过|在|由)", topic, maxsplit=1)[0].strip(" ，,。；;：:")
        if not (2 <= len(topic) <= 32):
            continue
        if (
            BAD_TOPIC_RE.search(topic)
            or topic.startswith(("向", "对", "把"))
            or topic.endswith(("因", "为", "由", "在", "的", "情况下", "至今"))
            or re.search(r"[“”\"']", topic)
        ):
            continue
        scope = infer_scope(source_text)
        if scope and GENERIC_TOPIC_RE.search(topic) and scope not in topic and scope[-4:] not in topic:
            topic = f"{scope}{topic}"
        return topic, intent
    return None


def topic_supported(topic: str, answer: str, source_text: str) -> bool:
    if topic in answer:
        return True
    if len(topic) > 4 and topic in normalize(source_text):
        return True
    topic_tail = topic[-4:]
    return len(topic_tail) >= 2 and topic_tail in answer


def relevant_extra_sentence(topic: str, text: str) -> bool:
    if not text or len(text) < 18 or len(text) > 180:
        return False
    if FIGURE_LINE_RE.search(text) or MARKUP_RE.search(text) or CROSS_REFERENCE_RE.search(text):
        return False
    if topic in text or topic[-3:] in text:
        return True
    return bool(re.search(r"(作用|用于|保证|满足|提高|保护|连接|固定|支撑|传递|监控|供电|可靠|安全)", text))


def enrich_answer(row: dict, topic: str) -> str | None:
    answer = clean_sentence(row.get("answer", ""))
    if MARKUP_RE.search(answer):
        return None
    if CROSS_REFERENCE_RE.search(answer):
        answer = clean_sentence(answer)
    if not (28 <= len(answer) <= 360):
        return None
    if len(answer) >= 35:
        return answer

    rows = context_lines(row.get("source_text", ""))
    line_number = row.get("line_number")
    index = next((i for i, item in enumerate(rows) if item.get("line") == line_number), -1)
    extras: list[str] = []
    if index >= 0:
        for item in rows[index + 1 : index + 5]:
            text = clean_sentence(item["text"])
            if topic in text and relevant_extra_sentence(topic, text):
                extras.append(text)
            if len(extras) >= 1:
                break
    merged = answer
    for extra in extras:
        if extra and extra not in merged:
            merged = merged.rstrip("。") + "；" + extra
    return merged if 35 <= len(merged) <= 420 else None


def quality_reason(question: str, answer: str, topic: str, source_text: str) -> str | None:
    if not question.endswith("？"):
        return "bad_question_punctuation"
    if re.search(r"(该知识点|其|它|这种|上述|有关|相关)", question):
        return "ambiguous_question"
    if CROSS_REFERENCE_RE.search(question) or CROSS_REFERENCE_RE.search(answer):
        return "cross_reference"
    if MARKUP_RE.search(answer):
        return "markup_or_table"
    if not topic_supported(topic, answer, source_text):
        return "topic_not_supported"
    if not (35 <= len(answer) <= 420):
        return "answer_length"
    if not answer.endswith(("。", "！", "？")):
        return "incomplete_answer"
    return None


def build_candidates(limit: int) -> tuple[list[dict], dict[str, int]]:
    accepted: list[dict] = []
    rejected: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    approved_pairs = load_existing_approved_pairs()

    source_rows = (row for row in source.iter_textbook_base_rows() if row.get("task_type") == "concept_explanation_qa")
    for row in chain(iter_old_unapproved_rows(), source_rows):
        if row.get("task_type") != "concept_explanation_qa":
            continue
        if FIGURE_LINE_RE.search(row.get("source_text", "")):
            # Only reject if the target answer itself is tied to a figure/table.
            if CROSS_REFERENCE_RE.search(row.get("answer", "")) or MARKUP_RE.search(row.get("answer", "")):
                rejected["figure_or_table_answer"] = rejected.get("figure_or_table_answer", 0) + 1
                continue
        inferred = infer_topic_and_intent(row.get("answer", ""), row.get("source_text", ""))
        if not inferred:
            rejected["topic_infer_failed"] = rejected.get("topic_infer_failed", 0) + 1
            continue
        topic, intent = inferred
        answer = enrich_answer(row, topic)
        if not answer:
            rejected["answer_enrich_failed"] = rejected.get("answer_enrich_failed", 0) + 1
            continue
        question = topic + QUESTION_SUFFIX[intent]
        reason = quality_reason(question, answer, topic, row.get("source_text", ""))
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        key = (question, answer)
        if key in seen or key in approved_pairs:
            continue
        seen.add(key)
        accepted.append(
            {
                **row,
                "task_type": "concept_explanation_qa",
                "question": question,
                "answer": answer,
                "evidence": answer,
                "generation_method": GENERATION_METHOD,
                "metadata_json": {
                    "generation_method": GENERATION_METHOD,
                    "line_number": row.get("line_number"),
                    "subject_resolution": "context_scope_prefix" if topic not in row.get("answer", "") else "answer_explicit",
                    "source_generation_method": row.get("generation_method"),
                },
            }
        )
        if len(accepted) >= limit:
            break
    return accepted, rejected


def iter_old_unapproved_rows() -> list[dict]:
    with SessionLocal() as db:
        items = db.scalars(
            select(CorpusItem).where(
                CorpusItem.task_type == "concept_explanation_qa",
                CorpusItem.review_status.in_(["pending", "rejected", "needs_revision"]),
            )
        ).all()
        return [
            {
                "task_type": item.task_type,
                "question": item.question,
                "answer": item.answer,
                "evidence": item.evidence,
                "source_text": item.source_text,
                "source": item.source_document,
                "source_path": item.source_path,
                "page_number": item.page_number,
                "line_number": (item.metadata_json or {}).get("line_number"),
                "chapter": item.chapter,
                "generation_method": (item.metadata_json or {}).get("generation_method", "existing_unapproved"),
            }
            for item in items
        ]


def load_existing_approved_pairs() -> set[tuple[str, str]]:
    with SessionLocal() as db:
        rows = db.scalars(
            select(CorpusItem).where(
                CorpusItem.task_type == "concept_explanation_qa",
                CorpusItem.review_status == "approved",
            )
        ).all()
        return {(item.question, item.answer) for item in rows}


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
            "question subject is resolved from answer and nearby OCR context",
            "ambiguous pronoun topics are rejected",
            "figure/table references are removed or rejected",
            "short answers are enriched only from adjacent OCR lines",
            "approved v2 records are preserved and deduplicated",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def delete_old_unapproved() -> int:
    with SessionLocal() as db:
        ids = list(
            db.scalars(
                select(CorpusItem.id).where(
                    CorpusItem.task_type == "concept_explanation_qa",
                    CorpusItem.review_status.in_(["pending", "rejected", "needs_revision"]),
                )
            )
        )
        if ids:
            db.execute(delete(ReviewEvent).where(ReviewEvent.item_id.in_(ids)))
            db.execute(delete(CorpusItem).where(CorpusItem.id.in_(ids)))
        db.commit()
        return len(ids)


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
                metadata_json={"concept_qa_version": "v3"},
            )
            for path in source.textbook_source.BOOK_PATHS
        }
        db_rows = []
        for row in rows:
            source_path = row["source_path"]
            external_id = stable_id(
                "textbook_concept_v3",
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
                    "quality_flags": ["human_review_required", "rebuilt_v3", "context_resolved_subject"],
                    "metadata_json": row.get("metadata_json") or {},
                }
            )
        inserted = insert_items(db, db_rows)
        db.commit()
        return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild context-resolved textbook concept QA.")
    parser.add_argument("--limit", type=int, default=TARGET_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-db", action="store_true")
    args = parser.parse_args()

    rows, rejected = build_candidates(args.limit)
    write_outputs(rows, rejected)
    deleted = 0
    inserted = 0
    if args.replace_db and not args.dry_run:
        deleted = delete_old_unapproved()
        inserted = import_rows(rows)
    print(
        json.dumps(
            {
                "accepted": len(rows),
                "deleted_old_unapproved": deleted,
                "inserted": inserted,
                "rejected_by_rule": rejected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for row in rows[:20]:
        print(f"Q: {row['question']}\nA: {row['answer']}\n")


if __name__ == "__main__":
    main()
