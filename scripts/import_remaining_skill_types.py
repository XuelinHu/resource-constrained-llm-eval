from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete, select, func

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.import_data import (  # noqa: E402
    build_regulation_contexts,
    insert_items,
    jsonl_rows,
    stable_id,
    upsert_document,
)
from app.models import CorpusItem  # noqa: E402
import import_textbook_extractive_from_combined_md as textbook_source  # noqa: E402


REGQA_DIR = REPO_ROOT / "data" / "domain_regqa_refined"
TERM_PATH = REPO_ROOT / "data" / "processed" / "railway_education" / "terminology_by_category.jsonl"
REG_DOCX = REPO_ROOT / "data" / "corpus" / "railway" / "规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx"
TERM_DOCX = REPO_ROOT / "data" / "corpus" / "railway" / "铁路中英文词汇（全）.docx"

REGULATION_TASKS = {
    "regulation_clause_qa",
    "regulation_definition_qa",
    "regulation_inspection_qa",
    "regulation_standard_qa",
    "regulation_prohibition_qa",
    "regulation_responsibility_qa",
    "regulation_principle_qa",
}

TARGET_TASKS = sorted(
    REGULATION_TASKS
    | {
        "regulation_extractive_qa",
        "regulation_judgment",
        "terminology_explanation",
        "terminology_pair",
        "terminology_translation",
        "textbook_definition_qa",
        "textbook_judgment",
        "textbook_multiple_choice",
        "textbook_operation_qa",
        "concept_explanation_qa",
    }
)

SOURCE_WORD_RE = re.compile(r"(教材中|规章中|原文|根据本文|依据材料|上述内容|Markdown)")
BAD_QUESTION_RE = re.compile(r"(原文是什么|表述是什么|从教材中|按教材原文)")
RESPONSIBILITY_RE = re.compile(r"(审批|批准|审查|申请报告|试运行任务|主要职责|职责|负责)")


def clean_question(question: str) -> str:
    question = textbook_source.normalize(question)
    question = re.sub(r"^回答以下铁路专业知识问题。", "", question).strip()
    question = re.sub(r"^只返回最终答案.*?标签。", "", question).strip()
    question = question.replace("教材中关于", "").replace("规章中关于", "")
    question = question.replace("的原文表述是什么？", "是什么？")
    return question


def rewrite_regulation_question(task_type: str, question: str) -> str:
    match = re.fullmatch(r"铁路领域中，关于(.+?)的专业要求是什么？", question)
    if match:
        subject = match.group(1).strip()
        suffix = {
            "regulation_clause_qa": "有哪些规章要求？",
            "regulation_definition_qa": "是什么？",
            "regulation_inspection_qa": "有哪些检查、检测或维护要求？",
            "regulation_standard_qa": "应符合哪些标准或要求？",
            "regulation_prohibition_qa": "有哪些禁止性要求？",
            "regulation_responsibility_qa": "有哪些职责要求？",
            "regulation_principle_qa": "应遵循哪些原则？",
        }.get(task_type, "有什么要求？")
        return f"{subject}{suffix}"
    question = question.replace("铁路领域中，关于", "").replace("的专业要求是什么？", "有什么要求？")
    return question


def usable_qa(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if SOURCE_WORD_RE.search(question) or BAD_QUESTION_RE.search(question):
        return False
    if len(answer.strip(" ；;。")) < 18:
        return False
    if len(question) > 120 or len(answer) > 800:
        return False
    return True


def take_limited(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return rows
    return rows[:limit]


def make_mcq(question: str, answer: str, distractors: list[str]) -> tuple[str, str] | None:
    choices = [answer] + [item for item in distractors if item != answer][:3]
    if len(choices) < 4:
        return None
    option_text = "\n".join(f"{label}. {choice}" for label, choice in zip("ABCD", choices, strict=True))
    return f"{question}\n{option_text}", f"A. {answer}"


def regulation_rows(limit_per_task: int) -> list[dict]:
    contexts = build_regulation_contexts(REG_DOCX)
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    for split in ("train", "valid", "test"):
        for row in jsonl_rows(REGQA_DIR / f"{split}.jsonl"):
            task_type = row.get("category", "")
            if task_type not in REGULATION_TASKS:
                continue
            question = clean_question(row.get("question") or row.get("prompt", "").splitlines()[-1].strip())
            question = rewrite_regulation_question(task_type, question)
            answer = textbook_source.normalize(row.get("answer", ""))
            if not usable_qa(question, answer):
                continue
            if task_type != "regulation_responsibility_qa" and RESPONSIBILITY_RE.search(question + answer):
                continue
            key = (task_type, question, answer)
            if key in seen:
                continue
            seen.add(key)
            paragraph_id = row.get("paragraph_id", "")
            grouped[task_type].append(
                {
                    "split": split,
                    "task_type": task_type,
                    "question": question,
                    "answer": answer,
                    "evidence": textbook_source.normalize(row.get("evidence", answer)),
                    "source_text": contexts.get(paragraph_id, textbook_source.normalize(row.get("evidence", answer))),
                    "source": row.get("source", REG_DOCX.name),
                    "paragraph_id": paragraph_id,
                    "answer_start": row.get("answer_start"),
                    "generation_method": row.get("generation_method", "refined_regqa_seed"),
                }
            )

    output: list[dict] = []
    for task_type in sorted(grouped):
        output.extend(take_limited(grouped[task_type], limit_per_task))
    return output


def derived_regulation_rows(base_rows: list[dict], limit_per_task: int) -> list[dict]:
    source = [row for row in base_rows if row["task_type"] in REGULATION_TASKS]
    output: list[dict] = []

    for row in source:
        output.append(
            {
                **row,
                "task_type": "regulation_extractive_qa",
                "question": row["question"].replace("有哪些规章要求？", "的具体规定是什么？"),
                "generation_method": "derived_regulation_extractive_seed",
            }
        )
        if 0 < limit_per_task <= len([item for item in output if item["task_type"] == "regulation_extractive_qa"]):
            break

    judgment_count = 0
    for row in source:
        if len(row["answer"]) > 180:
            continue
        output.append(
            {
                **row,
                "task_type": "regulation_judgment",
                "question": f"判断题：{row['answer']}",
                "answer": "正确",
                "generation_method": "derived_regulation_judgment_seed",
            }
        )
        judgment_count += 1
        if 0 < limit_per_task <= judgment_count:
            break

    return output


def terminology_rows(limit: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in jsonl_rows(TERM_PATH):
        term_zh = textbook_source.normalize(row.get("term_zh", ""))
        term_en = textbook_source.normalize(row.get("term_en", ""))
        category = row.get("domain_category", "")
        if not (term_zh and term_en):
            continue
        for task_type, question, answer in (
            ("terminology_explanation", f"{term_zh}是什么意思？", f"{term_zh}对应的英文铁路术语为 {term_en}。"),
            ("terminology_pair", term_zh, term_en),
            ("terminology_translation", f"{term_en} 对应的中文铁路术语是什么？", term_zh),
        ):
            key = (task_type, question, answer)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task_type": task_type,
                    "question": question,
                    "answer": answer,
                    "evidence": f"{term_zh} | {term_en}",
                    "source_text": f"{term_zh} | {term_en}",
                    "source": row.get("source", TERM_DOCX.name),
                    "domain_category": category,
                    "metadata_json": {
                        "domain_category_en": row.get("domain_category_en", ""),
                        "domain_category_key": row.get("domain_category_key", ""),
                        "source_block": row.get("source_block"),
                        "abbreviation": row.get("abbreviation", ""),
                        "full_name_en": row.get("full_name_en", ""),
                    },
                }
            )
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows


def textbook_rows(limit_per_task: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    for path in textbook_source.BOOK_PATHS:
        for candidate in textbook_source.build_candidates(path, before=20, after=20):
            answer = candidate.answer
            question = candidate.question
            task_types: list[str] = []
            if question.endswith("是什么？") and re.search(r"(是指|系由|称为|简称|定义|是)", answer):
                task_types.append("textbook_definition_qa")
            if re.search(r"(检查|检修|巡视|检测|维护|维修|故障|抢修|处理|运行)", answer + question):
                task_types.append("textbook_operation_qa")
            if re.search(r"(作用|特点|包括|组成|用于|具有)", answer + question):
                task_types.append("concept_explanation_qa")
            for task_type in task_types:
                key = (task_type, question, answer)
                if key in seen or not usable_qa(question, answer):
                    continue
                seen.add(key)
                grouped[task_type].append(
                    {
                        "task_type": task_type,
                        "question": question,
                        "answer": answer,
                        "evidence": answer,
                        "source_text": candidate.source_text,
                        "source": candidate.book_title,
                        "source_path": str(candidate.source_path.relative_to(REPO_ROOT)),
                        "page_number": candidate.page,
                        "line_number": candidate.line_number,
                        "metadata_json": {
                            "source_kind": "original_combined_md",
                            "line_number": candidate.line_number,
                            "generation_method": "rule_based_remaining_skill_seed",
                        },
                    }
                )

    output: list[dict] = []
    for task_type in sorted(grouped):
        output.extend(take_limited(grouped[task_type], limit_per_task))

    base_candidates: list[dict] = []
    for rows in grouped.values():
        base_candidates.extend(rows)
    base_candidates = base_candidates[: max(limit_per_task * 3, limit_per_task)]

    judgment_count = 0
    for row in base_candidates:
        if len(row["answer"]) > 180:
            continue
        output.append(
            {
                **row,
                "task_type": "textbook_judgment",
                "question": f"判断题：{row['answer']}",
                "answer": "正确",
            }
        )
        judgment_count += 1
        if 0 < limit_per_task <= judgment_count:
            break

    answers = [row["answer"] for row in base_candidates if 25 <= len(row["answer"]) <= 160]
    mcq_count = 0
    for index, row in enumerate(base_candidates):
        if not (25 <= len(row["answer"]) <= 160):
            continue
        mcq = make_mcq(row["question"], row["answer"], answers[index + 1 :] + answers[:index])
        if not mcq:
            continue
        question, answer = mcq
        output.append(
            {
                **row,
                "task_type": "textbook_multiple_choice",
                "question": question,
                "answer": answer,
            }
        )
        mcq_count += 1
        if 0 < limit_per_task <= mcq_count:
            break
    return output


def build_db_rows(db, *, limit_per_task: int, term_limit: int) -> list[dict]:
    reg_document = upsert_document(
        db,
        title="ECRL牵引供电设备运行维护管理办法",
        source_path=str(REG_DOCX.relative_to(REPO_ROOT)),
        document_type="regulation",
        domain_category="牵引供电",
        total_pages=None,
        metadata_json={"dataset": str(REGQA_DIR.relative_to(REPO_ROOT))},
    )
    term_document = upsert_document(
        db,
        title="铁路中英文专业词汇",
        source_path=str(TERM_DOCX.relative_to(REPO_ROOT)),
        document_type="terminology",
        domain_category="铁道教育",
        total_pages=None,
        metadata_json={"import_source": str(TERM_PATH.relative_to(REPO_ROOT))},
    )
    textbook_documents = {
        path: upsert_document(
            db,
            title=path.parent.name,
            source_path=str(path.relative_to(REPO_ROOT)),
            document_type="textbook",
            domain_category="牵引供电",
            total_pages=None,
            metadata_json={"source_kind": "original_combined_md"},
        )
        for path in textbook_source.BOOK_PATHS
    }

    rows: list[dict] = []
    reg_items = regulation_rows(limit_per_task)
    for item in reg_items + derived_regulation_rows(reg_items, limit_per_task):
        rows.append(
            {
                "external_id": stable_id("remaining_reg", item["task_type"], item["paragraph_id"], item["question"], item["answer"]),
                "source_type": "regulation_qa",
                "task_type": item["task_type"],
                "review_status": "pending",
                "domain_category": "牵引供电",
                "knowledge_category": "规章制度",
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "source_text": item["source_text"],
                "original_question": item["question"],
                "original_answer": item["answer"],
                "document_id": reg_document.id,
                "source_document": item["source"],
                "source_path": reg_document.source_path,
                "chapter": "",
                "page_number": None,
                "quality_flags": ["human_review_required"],
                "metadata_json": {
                    "split": item["split"],
                    "paragraph_id": item["paragraph_id"],
                    "answer_start": item["answer_start"],
                    "generation_method": item["generation_method"],
                    "skill_file": "/ds1/workspace/ai/multilingual-railway-llm-edu/skills/all_qa_skills.md",
                },
                "reviewer": "",
                "review_comment": "",
            }
        )

    for item in terminology_rows(term_limit):
        rows.append(
            {
                "external_id": stable_id("remaining_term", item["task_type"], item["question"], item["answer"]),
                "source_type": "terminology",
                "task_type": item["task_type"],
                "review_status": "pending",
                "domain_category": item["domain_category"],
                "knowledge_category": "专业术语",
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "source_text": item["source_text"],
                "original_question": item["question"],
                "original_answer": item["answer"],
                "document_id": term_document.id,
                "source_document": term_document.title,
                "source_path": term_document.source_path,
                "chapter": item["domain_category"],
                "page_number": None,
                "quality_flags": ["human_review_required"],
                "metadata_json": item["metadata_json"],
                "reviewer": "",
                "review_comment": "",
            }
        )

    for item in textbook_rows(limit_per_task):
        source_path = item["source_path"]
        document = textbook_documents[REPO_ROOT / source_path]
        rows.append(
            {
                "external_id": stable_id("remaining_textbook", item["task_type"], source_path, str(item["line_number"]), item["answer"]),
                "source_type": "textbook_original_md",
                "task_type": item["task_type"],
                "review_status": "pending",
                "domain_category": "牵引供电",
                "knowledge_category": "教材",
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "source_text": item["source_text"],
                "original_question": item["question"],
                "original_answer": item["answer"],
                "document_id": document.id,
                "source_document": item["source"],
                "source_path": source_path,
                "chapter": "",
                "page_number": item["page_number"],
                "quality_flags": ["human_review_required"],
                "metadata_json": item["metadata_json"],
                "reviewer": "",
                "review_comment": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import conservative seed data for remaining skill types.")
    parser.add_argument("--limit-per-task", type=int, default=120)
    parser.add_argument("--term-limit", type=int, default=300)
    parser.add_argument("--clear", action="store_true", help="Delete non-approved rows for target tasks before import.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    with SessionLocal() as db:
        if args.clear and not args.dry_run:
            db.execute(
                delete(CorpusItem).where(
                    CorpusItem.task_type.in_(TARGET_TASKS),
                    CorpusItem.review_status != "approved",
                )
            )
            db.commit()

        rows = build_db_rows(db, limit_per_task=args.limit_per_task, term_limit=args.term_limit)
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["task_type"]] += 1
        if args.dry_run:
            print({"candidate_count": len(rows), "by_task_type": dict(sorted(counts.items()))})
            for row in rows[:20]:
                print(row["task_type"], row["question"], "=>", row["answer"][:160])
            return

        inserted = insert_items(db, rows)
        db.commit()
        print({"candidate_count": len(rows), "inserted": inserted, "by_task_type": dict(sorted(counts.items()))})


if __name__ == "__main__":
    main()
