from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from .database import Base, SessionLocal, engine
from .models import CorpusItem, Document


REPO_ROOT = Path(__file__).resolve().parents[3]
SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[。！？；：]")
RULE_KEYWORD_RE = re.compile(
    r"(应|须|必须|不得|严禁|禁止|负责|职责|原则|方针|标准|要求|范围|包括|分为|"
    r"检查|检测|检修|维修|维护|试验|管理|安全|可靠|定期|周期|组织|执行|制定|落实|确保|适用于)"
)


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def jsonl_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return SPACE_RE.sub(" ", text).strip()


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        text = normalize_text("".join((node.text or "") for node in paragraph.findall(".//w:t", ns)))
        if text:
            paragraphs.append(text)
    return paragraphs


def is_heading_or_toc(text: str) -> bool:
    if PUNCT_RE.search(text):
        return False
    return bool(
        re.search(r"(目录|附件\s*\d+|第[一二三四五六七八九十0-9]+[章节部分])", text)
        or (len(text) <= 35 and re.search(r"(管理|标准|要求|记录|办法)\d*$", text))
    )


def is_rule_paragraph(text: str) -> bool:
    if not (16 <= len(text) <= 650):
        return False
    if not HAN_RE.search(text) or is_heading_or_toc(text) or not RULE_KEYWORD_RE.search(text):
        return False
    alpha_or_han = sum(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in text)
    return alpha_or_han / max(1, len(text)) >= 0.35


def source_prefix(path: Path) -> str:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
    return f"{path.stem[:12]}_{digest}"


def build_regulation_contexts(source_path: Path, before: int = 20, after: int = 20) -> dict[str, str]:
    if not source_path.is_file():
        return {}

    prefix = source_prefix(source_path)
    rule_paragraphs: list[tuple[str, str]] = []
    for raw_paragraph in read_docx_paragraphs(source_path):
        paragraph = normalize_text(raw_paragraph)
        if not is_rule_paragraph(paragraph):
            continue
        paragraph_id = f"{prefix}_p{len(rule_paragraphs) + 1:04d}"
        rule_paragraphs.append((paragraph_id, paragraph))

    contexts: dict[str, str] = {}
    for index, (paragraph_id, _paragraph) in enumerate(rule_paragraphs):
        start = max(0, index - before)
        end = min(len(rule_paragraphs), index + after + 1)
        contexts[paragraph_id] = "\n".join(paragraph for _pid, paragraph in rule_paragraphs[start:end])
    return contexts


def upsert_document(db, **values) -> Document:
    document = db.scalar(select(Document).where(Document.source_path == values["source_path"]))
    if document is None:
        document = Document(**values)
        db.add(document)
        db.flush()
    else:
        for key, value in values.items():
            setattr(document, key, value)
    return document


def insert_items(db, rows: list[dict]) -> int:
    if not rows:
        return 0
    statement = (
        insert(CorpusItem)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["external_id"])
        .returning(CorpusItem.id)
    )
    result = db.execute(statement)
    return len(result.scalars().all())


def import_terminology(db) -> int:
    path = REPO_ROOT / "data" / "processed" / "railway_education" / "terminology_by_category.jsonl"
    document = upsert_document(
        db,
        title="铁路中英文专业词汇",
        source_path="data/corpus/railway/铁路中英文词汇（全）.docx",
        document_type="terminology",
        domain_category="铁道教育",
        total_pages=None,
        metadata_json={"import_source": str(path.relative_to(REPO_ROOT))},
    )
    batch: list[dict] = []
    inserted = 0
    for row in jsonl_rows(path):
        term_zh = row.get("term_zh", "")
        term_en = row.get("term_en", "")
        category = row.get("domain_category", "")
        batch.append(
            {
                "external_id": stable_id("term", category, term_zh, term_en),
                "source_type": "terminology",
                "task_type": "terminology_pair",
                "review_status": "pending",
                "domain_category": category,
                "knowledge_category": "专业术语",
                "question": term_zh,
                "answer": term_en,
                "evidence": f"{term_zh} | {term_en}",
                "source_text": f"{term_zh} | {term_en}",
                "original_question": term_zh,
                "original_answer": term_en,
                "document_id": document.id,
                "source_document": document.title,
                "source_path": document.source_path,
                "chapter": category,
                "page_number": None,
                "quality_flags": [],
                "metadata_json": {
                    "domain_category_en": row.get("domain_category_en", ""),
                    "domain_category_key": row.get("domain_category_key", ""),
                    "source_block": row.get("source_block"),
                    "abbreviation": row.get("abbreviation", ""),
                    "full_name_en": row.get("full_name_en", ""),
                },
                "reviewer": "",
                "review_comment": "",
            }
        )
        if len(batch) >= 1000:
            inserted += insert_items(db, batch)
            batch.clear()
    inserted += insert_items(db, batch)
    return inserted


def import_regqa(db) -> int:
    base = REPO_ROOT / "data" / "domain_regqa_refined"
    source_docx = REPO_ROOT / "data" / "corpus" / "railway" / "规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx"
    contexts = build_regulation_contexts(source_docx)
    document = upsert_document(
        db,
        title="ECRL牵引供电设备运行维护管理办法",
        source_path="data/corpus/railway/规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx",
        document_type="regulation",
        domain_category="牵引供电",
        total_pages=None,
        metadata_json={"dataset": "domain_regqa_refined"},
    )
    rows: list[dict] = []
    for split in ("train", "valid", "test"):
        for row in jsonl_rows(base / f"{split}.jsonl"):
            paragraph_id = row.get("paragraph_id", "")
            question = row.get("prompt", "").splitlines()[-1].strip()
            answer = row.get("answer", "")
            rows.append(
                {
                    "external_id": stable_id("regqa", paragraph_id, question, answer),
                    "source_type": "regulation_qa",
                    "task_type": row.get("category", "regulation_qa"),
                    "review_status": "pending",
                    "domain_category": "牵引供电",
                    "knowledge_category": "规章制度",
                    "question": question,
                    "answer": answer,
                    "evidence": row.get("evidence", ""),
                    "source_text": contexts.get(paragraph_id, row.get("evidence", "")),
                    "original_question": question,
                    "original_answer": answer,
                    "document_id": document.id,
                    "source_document": row.get("source", document.title),
                    "source_path": document.source_path,
                    "chapter": "",
                    "page_number": None,
                    "quality_flags": [],
                    "metadata_json": {
                        "split": split,
                        "paragraph_id": paragraph_id,
                        "answer_start": row.get("answer_start"),
                        "generation_method": row.get("generation_method", ""),
                    },
                    "reviewer": "",
                    "review_comment": "",
                }
            )
    inserted = insert_items(db, rows)
    if contexts:
        existing_items = db.scalars(select(CorpusItem).where(CorpusItem.source_type == "regulation_qa")).all()
        for item in existing_items:
            paragraph_id = (item.metadata_json or {}).get("paragraph_id", "")
            context = contexts.get(paragraph_id)
            if context and item.source_text != context:
                item.source_text = context
    return inserted


def import_generated_dataset(db, dataset_dir: Path, *, document_title: str, source_type: str) -> int:
    if not dataset_dir.exists():
        return 0
    document = upsert_document(
        db,
        title=document_title,
        source_path=str(dataset_dir.relative_to(REPO_ROOT)),
        document_type="generated_dataset",
        domain_category="铁道教育",
        total_pages=None,
        metadata_json={"dataset_dir": str(dataset_dir.relative_to(REPO_ROOT))},
    )
    rows: list[dict] = []
    for split in ("train", "valid", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.exists():
            continue
        for row in jsonl_rows(split_path):
            question = row.get("question") or row.get("prompt", "").splitlines()[-1].strip()
            answer = row.get("answer", "")
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "split": split,
                    "unit_id": row.get("unit_id") or row.get("paragraph_id", ""),
                    "answer_start": row.get("answer_start"),
                    "generation_method": row.get("generation_method", ""),
                    "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
                }
            )
            source_id = row.get("id") or stable_id(source_type, split, question, answer)
            external_id = (
                stable_id(source_type, source_id)
                if source_type == "generated_eval_review"
                else source_id
            )
            rows.append(
                {
                    "external_id": external_id,
                    "source_type": source_type,
                    "task_type": row.get("task_type") or row.get("category", source_type),
                    "review_status": "pending",
                    "domain_category": row.get("domain_category", "铁道教育"),
                    "knowledge_category": row.get("knowledge_category", ""),
                    "question": question,
                    "answer": answer,
                    "evidence": row.get("evidence", ""),
                    "source_text": row.get("evidence", ""),
                    "original_question": question,
                    "original_answer": answer,
                    "document_id": document.id,
                    "source_document": row.get("source", document.title),
                    "source_path": row.get("source_path", document.source_path),
                    "chapter": row.get("chapter") or "",
                    "page_number": row.get("page"),
                    "quality_flags": ["human_review_required"],
                    "metadata_json": metadata,
                    "reviewer": "",
                    "review_comment": "",
                }
            )
    return insert_items(db, rows)


def import_generated_all(db) -> int:
    datasets = [
        (
            REPO_ROOT / "data" / "domain_regqa_expanded",
            "铁道教育规章增强QA",
            "generated_regulation_qa",
        ),
        (
            REPO_ROOT / "data" / "textbook_qa_generated",
            "铁道教育教材QA",
            "generated_textbook_qa",
        ),
        (
            REPO_ROOT / "data" / "railway_eval_review",
            "铁道教育人工审核测试候选集",
            "generated_eval_review",
        ),
    ]
    inserted = 0
    for dataset_dir, title, source_type in datasets:
        inserted += import_generated_dataset(db, dataset_dir, document_title=title, source_type=source_type)
    return inserted


def clean_ocr_markdown(text: str) -> str:
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def import_ocr(db) -> int:
    root = REPO_ROOT / "data" / "ocr" / "railway"
    inserted = 0
    for book_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = book_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        source_pdf = Path(config.get("pdf", book_dir.name + ".pdf"))
        total_pages = config.get("total_pages")
        document = upsert_document(
            db,
            title=source_pdf.stem,
            source_path=str(source_pdf),
            document_type="textbook",
            domain_category="牵引供电",
            total_pages=total_pages,
            metadata_json={
                "ocr_dir": str(book_dir.relative_to(REPO_ROOT)),
                "dpi": config.get("dpi"),
                "ocr_model": "DeepSeek-OCR-2",
            },
        )
        batch: list[dict] = []
        for page_path in sorted((book_dir / "pages").glob("page_*.md")):
            page_number = int(page_path.stem.rsplit("_", 1)[-1])
            raw_text = page_path.read_text(encoding="utf-8")
            clean_text = clean_ocr_markdown(raw_text)
            flags = ["ocr_error"] if len(clean_text) < 20 else []
            batch.append(
                {
                    "external_id": stable_id("ocr", document.source_path, str(page_number)),
                    "source_type": "ocr_page",
                    "task_type": "textbook_source",
                    "review_status": "pending",
                    "domain_category": "牵引供电",
                    "knowledge_category": "教材原文",
                    "question": "",
                    "answer": "",
                    "evidence": clean_text,
                    "source_text": raw_text,
                    "original_question": "",
                    "original_answer": "",
                    "document_id": document.id,
                    "source_document": document.title,
                    "source_path": document.source_path,
                    "chapter": "",
                    "page_number": page_number,
                    "quality_flags": flags,
                    "metadata_json": {
                        "ocr_page_path": str(page_path.relative_to(REPO_ROOT)),
                        "ocr_format": "deepseek_markdown",
                    },
                    "reviewer": "",
                    "review_comment": "",
                }
            )
        inserted += insert_items(db, batch)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Import railway corpora into the annotation database.")
    parser.add_argument(
        "--only",
        choices=["all", "terminology", "regqa", "ocr", "generated"],
        default="all",
    )
    args = parser.parse_args()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        actions = {
            "terminology": import_terminology,
            "regqa": import_regqa,
            "ocr": import_ocr,
            "generated": import_generated_all,
        }
        selected = actions if args.only == "all" else {args.only: actions[args.only]}
        for name, importer in selected.items():
            count = importer(db)
            db.commit()
            print(f"{name}: inserted={count}")


if __name__ == "__main__":
    main()
