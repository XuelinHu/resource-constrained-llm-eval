from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem, ReviewEvent  # noqa: E402


BOOK_CONTEXT_ROOT = REPO_ROOT / "data" / "ocr" / "railway_context"
TASK_TYPES = {
    "concept_explanation_qa",
    "textbook_definition_qa",
    "textbook_extractive_qa",
    "textbook_judgment",
    "textbook_multiple_choice",
    "textbook_operation_qa",
}
SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[^。！？；]+[。！？；]?")
BAD_TEXT_RE = re.compile(
    r"(来源页：|原图：|本文件由|教材：|来源 PDF|<a id=|图书在版编目|ISBN|责任编辑|责任校对|责任印制|"
    r"中国铁道出版社|铁路职工培训系列教材|购买铁道版图书|印制质量问题|读者服务部|版权所有|定价|网址|邮编)"
)
COMPLETE_ENDINGS = tuple("。！？；")


@dataclass
class Paragraph:
    book: str
    book_context_path: str
    page: int
    chapter: str
    text: str


@dataclass
class Candidate:
    task_type: str
    question: str
    answer: str
    evidence: str
    book: str
    book_context_path: str
    page: int
    chapter: str
    generation_method: str


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").replace("\u3000", " ")).strip()


def stable_id(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def clean_topic(sentence: str) -> str:
    segment = re.split(
        r"(应|须|必须|不得|严禁|禁止|负责|包括|分为|是|指|由|可|具有|用于|采用)",
        sentence,
        maxsplit=1,
    )[0]
    segment = re.sub(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\(?\d+\)?[.、）)]?)\s*", "", segment)
    segment = segment.strip(" ，,；;：:。！？（）()“”\"'《》")
    segment = re.sub(r"^(其中|同时|并|且|或|以及|对于|对|凡|有关|由|均|还|也|但|当|在|若|如)", "", segment).strip()
    segment = re.sub(r"[不无未]$", "", segment).strip()
    if not (2 <= len(segment) <= 42) or not HAN_RE.search(segment):
        return "该知识点"
    return segment[-32:]


def split_sentences(text: str, *, max_len: int = 260) -> list[str]:
    sentences: list[str] = []
    for raw in SENTENCE_RE.findall(text):
        sentence = normalize(raw)
        sentence = re.sub(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\(?\d+\)?[.、）)]?)\s*", "", sentence)
        if 18 <= len(sentence) <= max_len and sentence.endswith(COMPLETE_ENDINGS):
            if HAN_RE.search(sentence) and not BAD_TEXT_RE.search(sentence):
                sentences.append(sentence)
    return sentences


def parse_book_context(path: Path) -> list[Paragraph]:
    book = path.parent.name
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    paragraphs: list[Paragraph] = []
    current_page = 0
    current_chapter = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_chapter
        text = normalize(" ".join(buffer))
        buffer = []
        if not text or BAD_TEXT_RE.search(text):
            return
        if 4 <= len(text) <= 42 and HAN_RE.search(text) and re.search(r"^(第.+[章节篇]|[一二三四五六七八九十]+、|\(?\d+\)?[.、])", text):
            current_chapter = text
        if len(text) >= 18:
            paragraphs.append(
                Paragraph(
                    book=book,
                    book_context_path=rel_path,
                    page=current_page,
                    chapter=current_chapter,
                    text=text,
                )
            )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        page_match = re.match(r"\*\*第\s*(\d+)\s*页\*\*", line)
        if page_match:
            flush()
            current_page = int(page_match.group(1))
            continue
        if not line:
            flush()
            continue
        if line.startswith("<a id=") or line.startswith("来源页：") or line.startswith("原图："):
            continue
        if line.startswith("**教材：") or line.startswith("**来源 PDF：") or line.startswith("<!--"):
            continue
        buffer.append(line)
    flush()
    return paragraphs


def context_window(paragraphs: list[Paragraph], index: int, *, min_before: int = 220, min_after: int = 220) -> str:
    start = index
    before_chars = 0
    while start > 0 and paragraphs[start - 1].book == paragraphs[index].book and before_chars < min_before:
        start -= 1
        before_chars += len(paragraphs[start].text)

    end = index + 1
    after_chars = 0
    while end < len(paragraphs) and paragraphs[end].book == paragraphs[index].book and after_chars < min_after:
        after_chars += len(paragraphs[end].text)
        end += 1

    chunks: list[str] = []
    for paragraph in paragraphs[start:end]:
        page_prefix = f"【第 {paragraph.page} 页】" if paragraph.page else ""
        chunks.append(f"{page_prefix}{paragraph.text}")
    return "\n\n".join(chunks)


def questions_for_sentence(sentence: str) -> list[tuple[str, str]]:
    topic = clean_topic(sentence)
    if topic == "该知识点":
        return []
    questions = [
        ("textbook_extractive_qa", f"教材中关于{topic}的表述是什么？"),
        ("concept_explanation_qa", f"教材中关于{topic}的说明是什么？"),
    ]
    if re.search(r"(组成|包括|分为|由|内容|部分)", sentence):
        questions.append(("textbook_definition_qa", f"{topic}由哪些部分或内容组成？"))
    if re.search(r"(检查|检修|巡视|检测|维护|故障|运行|抢修|处理|安装|调整)", sentence):
        questions.append(("textbook_operation_qa", f"{topic}在运行检修中需要关注什么？"))
    if re.search(r"(应|必须|需要|不得|严禁|禁止|安全|要求)", sentence):
        questions.append(("textbook_judgment", f"判断题：{topic}是否有明确的运行或安全要求？"))
    return questions


def build_candidates() -> dict[str, list[Candidate]]:
    context_paths = sorted(BOOK_CONTEXT_ROOT.glob("*/book_context.md"))
    paragraphs: list[Paragraph] = []
    for path in context_paths:
        paragraphs.extend(parse_book_context(path))

    by_type: dict[str, list[Candidate]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    base_records: list[Candidate] = []
    for index, paragraph in enumerate(paragraphs):
        evidence = context_window(paragraphs, index)
        for sentence in split_sentences(paragraph.text):
            if sentence not in evidence:
                continue
            for task_type, question in questions_for_sentence(sentence):
                key = (task_type, question, sentence)
                if key in seen:
                    continue
                seen.add(key)
                candidate = Candidate(
                    task_type=task_type,
                    question=question,
                    answer=sentence,
                    evidence=evidence,
                    book=paragraph.book,
                    book_context_path=paragraph.book_context_path,
                    page=paragraph.page,
                    chapter=paragraph.chapter,
                    generation_method="book_context_window_rule_based",
                )
                by_type[task_type].append(candidate)
                if task_type in {"textbook_extractive_qa", "concept_explanation_qa"}:
                    base_records.append(candidate)

    rng = random.Random(42)
    answers_by_book: dict[str, list[str]] = defaultdict(list)
    for candidate in base_records:
        answers_by_book[candidate.book].append(candidate.answer)
    for source in base_records:
        pool = [answer for answer in answers_by_book[source.book] if answer != source.answer]
        rng.shuffle(pool)
        distractors: list[str] = []
        for answer in pool:
            if abs(len(answer) - len(source.answer)) <= 80 and answer not in distractors:
                distractors.append(answer)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = [source.answer] + distractors
        question = "\n".join(
            [
                source.question,
                f"A. {options[0]}",
                f"B. {options[1]}",
                f"C. {options[2]}",
                f"D. {options[3]}",
            ]
        )
        by_type["textbook_multiple_choice"].append(
            Candidate(
                task_type="textbook_multiple_choice",
                question=question,
                answer=source.answer,
                evidence=source.evidence,
                book=source.book,
                book_context_path=source.book_context_path,
                page=source.page,
                chapter=source.chapter,
                generation_method="book_context_window_rule_based_mcq",
            )
        )

    for task_type in by_type:
        by_type[task_type].sort(key=lambda item: (item.book, item.page, item.question, item.answer))
    return by_type


def book_page_for_item(item: CorpusItem) -> tuple[str, int]:
    metadata = item.metadata_json or {}
    context_path = str(metadata.get("book_context_path") or "")
    if context_path:
        parts = Path(context_path).parts
        if "railway_context" in parts:
            index = parts.index("railway_context")
            if index + 1 < len(parts):
                return parts[index + 1], int(metadata.get("book_context_page") or item.page_number or 0)

    markdown_path = str(metadata.get("ocr_page_path") or metadata.get("markdown") or "")
    if markdown_path:
        parts = Path(markdown_path).parts
        if "railway" in parts and "pages" in parts:
            page_index = parts.index("pages")
            book = parts[page_index - 1] if page_index > 0 else ""
            match = re.search(r"page_(\d+)\.md$", markdown_path)
            return book, int(match.group(1)) if match else int(item.page_number or 0)
    return "", int(item.page_number or 0)


def select_candidate(
    item: CorpusItem,
    candidates_by_type: dict[str, list[Candidate]],
    used: set[tuple[str, int]],
) -> Candidate | None:
    book, page = book_page_for_item(item)
    pool = candidates_by_type.get(item.task_type) or []
    if book:
        book_pool = [candidate for candidate in pool if candidate.book == book]
        if book_pool:
            pool = book_pool
    available = [
        (index, candidate)
        for index, candidate in enumerate(pool)
        if (candidate.task_type, id(candidate)) not in used
    ]
    if not available:
        available = list(enumerate(pool))
    if not available:
        return None
    index, candidate = min(
        available,
        key=lambda pair: (
            abs((pair[1].page or 0) - page) if page else 10_000,
            pair[1].page,
            pair[1].question,
        ),
    )
    used.add((candidate.task_type, id(candidate)))
    return candidate


def metadata_for(item: CorpusItem, candidate: Candidate) -> dict:
    metadata = dict(item.metadata_json or {})
    metadata.update(
        {
            "regenerated_from_book_context": True,
            "book_context_path": candidate.book_context_path,
            "book_context_page": candidate.page,
            "book_context_book": candidate.book,
            "generation_method": candidate.generation_method,
            "ocr_regeneration_id": stable_id(item.id, candidate.task_type, candidate.question, candidate.answer),
        }
    )
    evidence_sources = metadata.get("evidence_sources") if isinstance(metadata.get("evidence_sources"), dict) else {}
    payload = {
        "provider": "source-context",
        "evidence": candidate.evidence,
        "error": "",
        "model": "book-context-window",
        "context": candidate.evidence,
        "context_path": candidate.book_context_path,
        "source_label": f"OCR book context: {candidate.book_context_path} page {candidate.page}",
    }
    metadata["evidence_sources"] = {**evidence_sources, "codex": payload, "deepseek": payload}
    return metadata


def snapshot(item: CorpusItem) -> dict:
    return {
        "task_type": item.task_type,
        "question": item.question,
        "answer": item.answer,
        "evidence": item.evidence,
        "source_text": item.source_text,
        "chapter": item.chapter,
        "page_number": item.page_number,
        "metadata_json": item.metadata_json,
        "version": item.version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate existing OCR-derived questions from full book context Markdown.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env()
    candidates_by_type = build_candidates()
    updated = 0
    skipped = 0
    counts: Counter[str] = Counter()

    with SessionLocal() as db:
        items = list(
            db.scalars(
                select(CorpusItem)
                .where(CorpusItem.review_status != "deleted", CorpusItem.task_type.in_(TASK_TYPES))
                .order_by(CorpusItem.task_type, CorpusItem.id)
            )
        )
        items = [
            item
            for item in items
            if isinstance(item.metadata_json, dict)
            and (item.metadata_json.get("ocr_page_path") or item.metadata_json.get("markdown") or item.metadata_json.get("book_context_path"))
        ]
        if args.limit:
            items = items[: args.limit]

        used: set[tuple[str, int]] = set()
        for item in items:
            candidate = select_candidate(item, candidates_by_type, used)
            if candidate is None:
                skipped += 1
                continue
            before = snapshot(item)

            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "id": item.id,
                            "task_type": item.task_type,
                            "old_question": item.question,
                            "new_question": candidate.question,
                            "new_answer": candidate.answer,
                            "book_context_path": candidate.book_context_path,
                            "page": candidate.page,
                        },
                        ensure_ascii=False,
                    )
                )
                updated += 1
                counts[item.task_type] += 1
                continue

            item.question = candidate.question
            item.answer = candidate.answer
            item.evidence = candidate.evidence
            item.source_text = candidate.evidence
            item.chapter = candidate.chapter
            item.page_number = candidate.page
            item.metadata_json = metadata_for(item, candidate)
            item.version += 1
            db.add(
                ReviewEvent(
                    item=item,
                    action="regen_ocr_book_context",
                    reviewer="system",
                    comment="Regenerated OCR-derived question and answer from full book context Markdown.",
                    snapshot={"before": before, "after": snapshot(item)},
                )
            )
            db.commit()
            updated += 1
            counts[item.task_type] += 1

    print(
        json.dumps(
            {
                "updated": updated,
                "skipped": skipped,
                "by_task_type": dict(sorted(counts.items())),
                "candidate_counts": {key: len(value) for key, value in sorted(candidates_by_type.items())},
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
