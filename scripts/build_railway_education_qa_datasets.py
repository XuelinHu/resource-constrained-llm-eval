"""Build expanded railway education QA datasets from local traceable sources.

The generator is deliberately rule based. Every answer is grounded in an
existing regulation sentence, textbook OCR sentence, or terminology record.
It does not call an LLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
RAILWAY_DIR = REPO_ROOT / "data" / "corpus" / "railway"
OCR_DIR = REPO_ROOT / "data" / "ocr" / "railway"
TERM_PATH = REPO_ROOT / "data" / "processed" / "railway_education" / "terminology_by_category.jsonl"

REG_OUT = REPO_ROOT / "data" / "domain_regqa_expanded"
TEXTBOOK_OUT = REPO_ROOT / "data" / "textbook_qa_generated"
EVAL_OUT = REPO_ROOT / "data" / "railway_eval_review"

SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[^。！？；]+[。！？；]?")
MARKDOWN_NOISE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>")
HEADING_RE = re.compile(r"^#{1,6}\s*")
LIST_MARK_RE = re.compile(
    r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|[（(]?[0-9一二三四五六七八九十]{1,3}[）)、.．])\s*"
)
RULE_KEYWORD_RE = re.compile(
    r"(应|须|必须|不得|严禁|禁止|负责|职责|原则|方针|标准|要求|范围|包括|分为|"
    r"检查|检测|检修|维修|维护|试验|管理|安全|可靠|定期|周期|组织|执行|制定|落实|确保|适用于)"
)
TEXTBOOK_KEYWORD_RE = re.compile(
    r"(接触网|牵引供电|变电所|供电臂|承力索|接触线|吊弦|支柱|定位装置|补偿装置|"
    r"电分相|分段绝缘器|隔离开关|避雷器|保护线|回流线|检修|巡视|检测|运行|故障|安全|电压|电流)"
)
BAD_TEXT_RE = re.compile(r"(铁路职工培训系列教材|中国铁道出版社|编委会|ISBN|CIP|责任编辑|封面设计|印刷|开本|字数|版次)")
COMPLETE_ENDINGS = ("。", "！", "？", "；")


@dataclass(frozen=True)
class QARecord:
    id: str
    task_type: str
    question: str
    answer: str
    evidence: str
    source: str
    source_path: str
    unit_id: str
    domain_category: str
    knowledge_category: str
    page: int | None = None
    chapter: str = ""
    answer_start: int = 0
    split: str = "train"
    generation_method: str = "rule_based_extractive"
    metadata: dict | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "prompt": (
                "回答以下铁道教育专业问题。\n"
                "只返回最终答案，不要解释，不要添加前缀、后缀、引号或标签。\n"
                f"{self.question}"
            ),
            "question": self.question,
            "answer": self.answer,
            "text": f"Question: {self.question}\nAnswer: {self.answer}",
            "category": self.task_type,
            "source": self.source,
            "source_path": self.source_path,
            "unit_id": self.unit_id,
            "paragraph_id": self.unit_id,
            "evidence": self.evidence,
            "answer_start": self.answer_start,
            "domain_category": self.domain_category,
            "knowledge_category": self.knowledge_category,
            "page": self.page,
            "chapter": self.chapter,
            "split": self.split,
            "generation_method": self.generation_method,
            "metadata": self.metadata or {},
        }


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return SPACE_RE.sub(" ", text).strip()


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = normalize("".join((node.text or "") for node in para.findall(".//w:t", ns)))
        if text:
            paragraphs.append(text)
    return paragraphs


def split_sentences(text: str, *, max_len: int = 260) -> list[str]:
    output: list[str] = []
    for raw in SENTENCE_RE.findall(text):
        sentence = normalize(LIST_MARK_RE.sub("", raw))
        if 18 <= len(sentence) <= max_len and sentence.endswith(COMPLETE_ENDINGS):
            if HAN_RE.search(sentence) and not BAD_TEXT_RE.search(sentence):
                output.append(sentence)
    return output


def clean_topic(sentence: str) -> str:
    segment = re.split(r"(应|须|必须|不得|严禁|禁止|负责|包括|分为|是|指|由|可|具有|用于)", sentence, maxsplit=1)[0]
    segment = LIST_MARK_RE.sub("", segment)
    segment = segment.strip(" ，,；;：:。！？（）()“”\"'《》")
    segment = re.sub(r"^(其中|同时|并|且|或|以及|对|对于|凡|有关|由|均|还|也|但|当|在)", "", segment).strip()
    if len(segment) < 2 or len(segment) > 34 or not HAN_RE.search(segment):
        return "该知识点"
    return segment[-28:]


def valid_answer(answer: str, evidence: str) -> bool:
    if answer not in evidence:
        return False
    if not (8 <= len(answer) <= 260):
        return False
    if BAD_TEXT_RE.search(answer):
        return False
    if answer.count("（") != answer.count("）") or answer.count("(") != answer.count(")"):
        return False
    return True


def make_record(
    *,
    prefix: str,
    task_type: str,
    question: str,
    answer: str,
    evidence: str,
    source: str,
    source_path: str,
    unit_id: str,
    domain_category: str,
    knowledge_category: str,
    page: int | None = None,
    chapter: str = "",
    metadata: dict | None = None,
) -> QARecord | None:
    question = normalize(question)
    answer = normalize(answer)
    evidence = normalize(evidence)
    if not question.endswith("？"):
        question += "？"
    answer_start = evidence.find(answer)
    if answer_start < 0 or not valid_answer(answer, evidence):
        return None
    return QARecord(
        id=stable_id(prefix, task_type, unit_id, question, answer),
        task_type=task_type,
        question=question,
        answer=answer,
        evidence=evidence,
        source=source,
        source_path=source_path,
        unit_id=unit_id,
        domain_category=domain_category,
        knowledge_category=knowledge_category,
        page=page,
        chapter=chapter,
        answer_start=answer_start,
        metadata=metadata or {},
    )


def make_mcq_record(
    *,
    prefix: str,
    source_record: QARecord,
    distractors: list[str],
    task_type: str,
) -> QARecord | None:
    options = [source_record.answer] + distractors[:3]
    if len(options) < 4:
        return None
    labels = ["A", "B", "C", "D"]
    option_lines = [f"{label}. {option}" for label, option in zip(labels, options)]
    evidence = source_record.evidence + " 选项：" + " ".join(option_lines)
    question = source_record.question.rstrip("？") + "？\n" + "\n".join(option_lines)
    answer = f"A. {source_record.answer}"
    return QARecord(
        id=stable_id(prefix, task_type, source_record.unit_id, source_record.question, answer),
        task_type=task_type,
        question=question,
        answer=answer,
        evidence=evidence,
        source=source_record.source,
        source_path=source_record.source_path,
        unit_id=source_record.unit_id + "_mcq",
        domain_category=source_record.domain_category,
        knowledge_category=source_record.knowledge_category,
        page=source_record.page,
        chapter=source_record.chapter,
        answer_start=evidence.find(answer),
        metadata={**(source_record.metadata or {}), "base_record_id": source_record.id, "option_order": labels},
    )


def add_multiple_choice(records: list[QARecord], *, prefix: str, task_type: str, limit: int, seed: int) -> list[QARecord]:
    rng = random.Random(seed)
    candidates = [record for record in records if 20 <= len(record.answer) <= 180]
    by_domain: dict[str, list[QARecord]] = defaultdict(list)
    for record in candidates:
        by_domain[record.domain_category].append(record)
    output = list(records)
    rng.shuffle(candidates)
    added = 0
    for record in candidates:
        pool = [item.answer for item in by_domain[record.domain_category] if item.answer != record.answer]
        rng.shuffle(pool)
        distractors: list[str] = []
        for answer in pool:
            if abs(len(answer) - len(record.answer)) <= 80 and answer not in distractors:
                distractors.append(answer)
            if len(distractors) >= 3:
                break
        mcq = make_mcq_record(prefix=prefix, source_record=record, distractors=distractors, task_type=task_type)
        if mcq is None:
            continue
        output.append(mcq)
        added += 1
        if added >= limit:
            break
    return output


def regulation_questions(sentence: str) -> list[tuple[str, str]]:
    topic = clean_topic(sentence)
    if topic == "该知识点":
        return []
    questions: list[tuple[str, str]] = [
        ("regulation_extractive_qa", f"根据规章，{topic}的具体规定是什么？"),
    ]
    if re.search(r"(应|须|必须|确保|落实|执行|制定)", sentence):
        questions.append(("regulation_requirement_qa", f"{topic}应满足什么要求？"))
    if re.search(r"(标准|要求)", sentence):
        questions.append(("regulation_standard_qa", f"{topic}的标准或要求是什么？"))
    if re.search(r"(检查|检测|检修|维修|维护|试验)", sentence):
        questions.append(("regulation_inspection_qa", f"{topic}的检查、检测或维护要求是什么？"))
    if re.search(r"(不得|严禁|禁止|不应)", sentence):
        questions.append(("regulation_judgment", f"判断题：{topic}是否存在禁止性要求？"))
    if re.search(r"(负责|职责)", sentence):
        questions.append(("regulation_responsibility_qa", f"{topic}负责哪些工作或职责？"))
    if "包括" in sentence:
        questions.append(("regulation_definition_qa", f"{topic}包括哪些内容？"))
    if "分为" in sentence:
        questions.append(("regulation_definition_qa", f"{topic}分为哪些类型？"))
    return questions


def build_regulation(limit: int) -> list[QARecord]:
    records: list[QARecord] = []
    for docx in sorted(RAILWAY_DIR.glob("*规章*.docx")):
        source_path = str(docx.relative_to(REPO_ROOT))
        para_index = 0
        for paragraph in read_docx_paragraphs(docx):
            paragraph = normalize(paragraph)
            if not (20 <= len(paragraph) <= 900 and RULE_KEYWORD_RE.search(paragraph)):
                continue
            para_index += 1
            unit_id = f"{docx.stem[:16]}_{para_index:04d}"
            for sent_index, sentence in enumerate(split_sentences(paragraph), 1):
                if not RULE_KEYWORD_RE.search(sentence):
                    continue
                sentence_unit = f"{unit_id}_s{sent_index:02d}"
                for task_type, question in regulation_questions(sentence):
                    record = make_record(
                        prefix="regx",
                        task_type=task_type,
                        question=question,
                        answer=sentence,
                        evidence=paragraph,
                        source=docx.name,
                        source_path=source_path,
                        unit_id=sentence_unit,
                        domain_category="牵引供电",
                        knowledge_category="规章制度",
                    )
                    if record:
                        records.append(record)
                if len(records) >= limit * 2:
                    break
            if len(records) >= limit * 2:
                break
    return dedupe(records)[:limit]


def clean_markdown(text: str) -> str:
    text = MARKDOWN_NOISE_RE.sub("", text)
    text = HEADING_RE.sub("", text)
    return normalize(text)


def infer_chapter(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = normalize(HEADING_RE.sub("", line))
        if 4 <= len(line) <= 36 and HAN_RE.search(line) and not BAD_TEXT_RE.search(line):
            return line
    return fallback


def textbook_questions(sentence: str) -> list[tuple[str, str]]:
    topic = clean_topic(sentence)
    if topic == "该知识点":
        return []
    questions = [
        ("textbook_extractive_qa", f"教材中关于{topic}的表述是什么？"),
        ("concept_explanation_qa", f"{topic}在铁道教育中的含义或作用是什么？"),
    ]
    if re.search(r"(组成|包括|分为|由)", sentence):
        questions.append(("textbook_definition_qa", f"{topic}由哪些部分或内容组成？"))
    if re.search(r"(检查|检修|巡视|检测|维护|故障|运行)", sentence):
        questions.append(("textbook_operation_qa", f"{topic}在运行检修中需要关注什么？"))
    if re.search(r"(应|必须|需要|不得|严禁|安全)", sentence):
        questions.append(("textbook_judgment", f"判断题：{topic}是否有明确的运行或安全要求？"))
    return questions


def build_textbook(limit: int) -> list[QARecord]:
    records: list[QARecord] = []
    for page_path in sorted(OCR_DIR.glob("*/pages/page_*.md")):
        raw = page_path.read_text(encoding="utf-8")
        text = clean_markdown(raw)
        if len(text) < 80 or BAD_TEXT_RE.search(text):
            continue
        book_dir = page_path.parents[1]
        config_path = book_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        source_pdf = Path(config.get("pdf", book_dir.name + ".pdf"))
        page = int(page_path.stem.rsplit("_", 1)[-1])
        chapter = infer_chapter(raw, source_pdf.stem)
        sentences = [s for s in split_sentences(text, max_len=240) if TEXTBOOK_KEYWORD_RE.search(s)]
        if not sentences:
            continue
        evidence = " ".join(sentences[:6])
        for sent_index, sentence in enumerate(sentences[:8], 1):
            unit_id = f"{source_pdf.stem}_p{page:04d}_s{sent_index:02d}"
            for task_type, question in textbook_questions(sentence):
                record = make_record(
                    prefix="tbqa",
                    task_type=task_type,
                    question=question,
                    answer=sentence,
                    evidence=evidence,
                    source=source_pdf.name,
                    source_path=str(source_pdf),
                    unit_id=unit_id,
                    domain_category="牵引供电",
                    knowledge_category="教材知识",
                    page=page,
                    chapter=chapter,
                    metadata={"ocr_page_path": str(page_path.relative_to(REPO_ROOT))},
                )
                if record:
                    records.append(record)
            if len(records) >= limit * 2:
                break
        if len(records) >= limit * 2:
            break
    return dedupe(records)[:limit]


def read_terms() -> list[dict]:
    terms: list[dict] = []
    with TERM_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                terms.append(json.loads(line))
    return terms


def build_term_tasks(limit: int = 1000) -> list[QARecord]:
    records: list[QARecord] = []
    for row in read_terms():
        zh = normalize(row.get("term_zh", ""))
        en = normalize(row.get("term_en", ""))
        category = row.get("domain_category", "")
        if not zh or not en:
            continue
        translation_evidence = f"{zh} | {en} | {category}"
        explanation = f"{zh}属于{category}专业类别。"
        explanation_evidence = f"{zh} | {en} | {explanation}"
        for task_type, question, answer in [
            ("terminology_translation", f"{zh}的英文术语是什么？", en),
            ("terminology_explanation", f"{zh}属于哪个铁道专业类别？", explanation),
        ]:
            evidence = translation_evidence if task_type == "terminology_translation" else explanation_evidence
            record = make_record(
                prefix="termqa",
                task_type=task_type,
                question=question,
                answer=answer,
                evidence=evidence,
                source=row.get("source", "铁路中英文词汇（全）.docx"),
                source_path="data/corpus/railway/铁路中英文词汇（全）.docx",
                unit_id=stable_id("term_unit", category, zh, en),
                domain_category=category,
                knowledge_category="专业术语",
                chapter=category,
                metadata={
                    "term_zh": zh,
                    "term_en": en,
                    "domain_category_en": row.get("domain_category_en", ""),
                    "source_block": row.get("source_block"),
                },
            )
            if record:
                records.append(record)
        if len(records) >= limit:
            break
    return dedupe(records)[:limit]


def dedupe(records: Iterable[QARecord]) -> list[QARecord]:
    seen: set[tuple[str, str, str]] = set()
    output: list[QARecord] = []
    for record in records:
        key = (record.task_type, record.question, record.answer)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def assign_splits(records: list[QARecord], *, seed: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> list[QARecord]:
    rng = random.Random(seed)
    by_unit: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        by_unit[record.unit_id].append(record)
    units = list(by_unit)
    rng.shuffle(units)
    train_end = round(len(units) * train_ratio)
    valid_end = train_end + round(len(units) * valid_ratio)
    split_by_unit = {
        **{unit: "train" for unit in units[:train_end]},
        **{unit: "valid" for unit in units[train_end:valid_end]},
        **{unit: "test" for unit in units[valid_end:]},
    }
    output: list[QARecord] = []
    for record in records:
        output.append(QARecord(**{**record.__dict__, "split": split_by_unit[record.unit_id]}))
    return output


def rebalance(records: list[QARecord], limit: int, seed: int) -> list[QARecord]:
    rng = random.Random(seed)
    by_type: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        by_type[record.task_type].append(record)
    for values in by_type.values():
        rng.shuffle(values)
    selected: list[QARecord] = []
    while len(selected) < limit and by_type:
        for task_type in sorted(list(by_type)):
            if not by_type[task_type]:
                del by_type[task_type]
                continue
            selected.append(by_type[task_type].pop())
            if len(selected) >= limit:
                break
    rng.shuffle(selected)
    return selected


def write_dataset(out_dir: Path, records: list[QARecord]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                if record.split == split:
                    handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    with (out_dir / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "task_type", "domain_category", "knowledge_category", "source", "page", "question", "answer", "evidence"])
        for record in records:
            writer.writerow([
                record.split,
                record.task_type,
                record.domain_category,
                record.knowledge_category,
                record.source,
                record.page or "",
                record.question,
                record.answer,
                record.evidence,
            ])
    counts = Counter(record.task_type for record in records)
    split_counts = Counter(record.split for record in records)
    lines = [
        f"# {out_dir.name}",
        "",
        "Rule-based, source-grounded railway education QA dataset.",
        "Every answer is an exact substring of the evidence field.",
        "",
        "## Splits",
        "",
    ]
    for split in ("train", "valid", "test"):
        lines.append(f"- {split}: {split_counts[split]}")
    lines.extend(["", "## Task Types", ""])
    for task_type, count in sorted(counts.items()):
        lines.append(f"- {task_type}: {count}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_eval_set(reg_records: list[QARecord], textbook_records: list[QARecord], term_records: list[QARecord], limit: int, seed: int) -> list[QARecord]:
    pool = [record for record in reg_records + textbook_records + term_records if record.split == "test"]
    selected = rebalance(pool, min(limit, len(pool)), seed)
    return [QARecord(**{**record.__dict__, "split": "test"}) for record in selected]


def verify(records: list[QARecord]) -> None:
    for record in records:
        payload = record.to_json()
        if payload["answer"] not in payload["evidence"]:
            raise ValueError(f"answer not grounded: {record.id}")
    by_split_unit: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_split_unit[record.split].add(record.unit_id)
    if by_split_unit["train"] & by_split_unit["test"]:
        raise ValueError("train/test unit leakage detected")
    if by_split_unit["valid"] & by_split_unit["test"]:
        raise ValueError("valid/test unit leakage detected")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build expanded railway education QA datasets.")
    parser.add_argument("--reg-limit", type=int, default=4000)
    parser.add_argument("--textbook-limit", type=int, default=2500)
    parser.add_argument("--term-task-limit", type=int, default=1200)
    parser.add_argument("--eval-limit", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_reg_records = build_regulation(args.reg_limit)
    reg_records = assign_splits(rebalance(raw_reg_records, args.reg_limit, args.seed), seed=args.seed)
    raw_textbook_records = build_textbook(args.textbook_limit)
    raw_textbook_records = add_multiple_choice(
        raw_textbook_records,
        prefix="tbmcq",
        task_type="textbook_multiple_choice",
        limit=700,
        seed=args.seed + 11,
    )
    textbook_records = assign_splits(
        rebalance(raw_textbook_records, args.textbook_limit, args.seed + 1),
        seed=args.seed + 1,
    )
    term_records = assign_splits(
        rebalance(build_term_tasks(args.term_task_limit), args.term_task_limit, args.seed + 2),
        seed=args.seed + 2,
    )
    eval_records = build_eval_set(reg_records, textbook_records, term_records, args.eval_limit, args.seed + 3)

    verify(reg_records)
    verify(textbook_records)
    verify(term_records)
    verify(eval_records)

    write_dataset(REG_OUT, reg_records)
    write_dataset(TEXTBOOK_OUT, textbook_records)
    write_dataset(EVAL_OUT, eval_records)

    for name, records in [
        ("regulation", reg_records),
        ("textbook", textbook_records),
        ("term_tasks", term_records),
        ("eval", eval_records),
    ]:
        print(f"{name}: total={len(records)} splits={dict(Counter(r.split for r in records))}")
        for task_type, count in sorted(Counter(r.task_type for r in records).items()):
            print(f"  {task_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
