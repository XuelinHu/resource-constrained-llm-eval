from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sqlalchemy import delete, func, select

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


TARGET_PER_TASK = 500
DOMAIN_CATEGORY = "牵引供电"
REG_DOCX = REPO_ROOT / "data" / "corpus" / "railway" / "规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx"
TERM_PATH = REPO_ROOT / "data" / "processed" / "railway_education" / "terminology_by_category.jsonl"
TERM_DOCX = REPO_ROOT / "data" / "corpus" / "railway" / "铁路中英文词汇（全）.docx"
REGQA_DIRS = [
    REPO_ROOT / "data" / "domain_regqa_expanded",
    REPO_ROOT / "data" / "domain_regqa_refined",
    REPO_ROOT / "data" / "domain_regqa",
]

REGULATION_TASKS = {
    "regulation_clause_qa",
    "regulation_definition_qa",
    "regulation_extractive_qa",
    "regulation_inspection_qa",
    "regulation_judgment",
    "regulation_principle_qa",
    "regulation_prohibition_qa",
    "regulation_requirement_qa",
    "regulation_responsibility_qa",
    "regulation_standard_qa",
}

TEXTBOOK_TASKS = {
    "concept_explanation_qa",
    "textbook_definition_qa",
    "textbook_extractive_qa",
    "textbook_judgment",
    "textbook_multiple_choice",
    "textbook_operation_qa",
}

TERMINOLOGY_SMALL_TASKS = {
    "terminology_explanation",
    "terminology_translation",
}

TARGET_TASKS = sorted(REGULATION_TASKS | TEXTBOOK_TASKS | TERMINOLOGY_SMALL_TASKS)

SOURCE_WORD_RE = re.compile(r"(根据|依据|按照|结合).{0,8}(规章|制度|教材|材料|原文|语料|文中|书中)")
PROMPT_PREFIX_RE = re.compile(r"^回答以下[^。\n]*问题。\s*只返回最终答案[^。\n]*。\s*")
SPACE_RE = re.compile(r"\s+")
BAD_QUESTION_RE = re.compile(r"(原文|语料|Markdown|OCR|上下文|这段|材料中|文中|书中|教材中|规章中)")
BAD_GENERATED_QUESTION_RE = re.compile(
    r"^(其|它|该|这种|这类|上述|相关|有关|一般|同时|其中|因此|另外|以下|如下|主要内容|内容)"
    r"|^维修的检查、检测或维护要求是什么"
    r"|^定期检查、分析、鉴定设备运行状态"
    r"|^各级调度、供电段及沿线所亭、工区"
)
LIST_LEAD_RE = re.compile(r"(如下|以下|包括|分为|分成|主要有|主要内容|内容如下)[:：。；;]?$")
TEXTBOOK_LONG_HINT_RE = re.compile(
    r"(是指|称为|定义|简称|应|必须|需要|要求|不得|严禁|禁止|包括|组成|分为|分成|"
    r"作用|特点|用于|具有|检查|检修|巡视|检测|维护|维修|故障|抢修|处理|运行|作业)"
)
DEFINITION_RE = re.compile(r"(是指|称为|定义|简称|是由|是指由)")
OPERATION_RE = re.compile(r"(检查|检修|巡视|检测|维护|维修|抢修|处理|作业)")
CONCEPT_RE = re.compile(r"(作用|特点|包括|组成|分为|分成|用于|具有|由.+组成)")
CONCEPT_DIRECT_RE = re.compile(
    r"(特点是|具有.{0,24}(?:特点|优点)|(?:其|主要)?作用(?:是|为|在于)|"
    r"起到.{0,24}作用|(?<!不)用于|(?:主要)?功能(?:是|包括|为)|是指|称为|定义为|简称)"
)
CROSS_REFERENCE_RE = re.compile(r"(如图|见图|如表|见表|详见|参见|图\s*\d|表\s*\d)")
TRAILING_INCOMPLETE_RE = re.compile(r"(如下|下列|包括以下|分为下列|内容如下|如图所示|见图|见表)[：:?。；;]*$")
SECTION_LINE_RE = re.compile(
    r"^(?:第[一二三四五六七八九十\d]+章|[一二三四五六七八九十]+[\、.]|[\(（]?一|\d+[\、.])"
)
BAD_TOPIC_RE = re.compile(r"^(由于|因为|当|在|若|如|如果|凡|对|对于|由|按|根据|按照|该|其|此|故|但|同时|其中)")
STOP_TOPICS = {"定期", "及时", "正常", "一般", "同时", "其中", "因此", "另外"}
NON_OPERATION_CONTEXT_RE = re.compile(
    r"(维修任务量|施工和维修|便于.*检查|安全检查.*补偿滑轮|"
    r"贯彻执行|管理职责|职责和范围|下达.*计划|组织.*培训|组织.*考核|"
    r"定期检查分析设备运行状态|制定改进措施|技术革新|职工培训|"
    r"铁路局应|供电段应按|各级调度|沿线所亭|工区.*房屋)"
)
OPERATION_ACTION_RE = re.compile(
    r"(应|须|必须|按规定|定期|计划|组织|进行|实施|接管|开通|抢修|处理).{0,30}"
    r"(检查|检修|巡视|检测|维护|维修|抢修|处理|验收|接管|运行)"
)
NUMBERED_START_RE = re.compile(r"^\s*(?:[（(]?(\d{1,3})[）)]|([①②③④⑤⑥⑦⑧⑨⑩])|([一二三四五六七八九十]{1,3})[、.）)])")
FIRST_NUMBER_MARKS = {"1", "①", "一"}
CONTEXT_SUBJECT_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9＋+×x/（）()、，]{2,32}?"
    r"(?:设备|设施|装置|系统|机构|部件|零部件|绝缘子|接触网|承力索|接触线|"
    r"支柱|吊弦|定位器|补偿装置|隔离开关|断路器|变压器|电缆|馈线|保护线|"
    r"供电段|工区|车间|铁路局|计划|检修|维修|巡视|检测|试验|验收|接管|抢修|施工))"
)


def normalize(text: str) -> str:
    return textbook_source.normalize(text or "")


def clean_question(question: str) -> str:
    question = normalize(question)
    question = PROMPT_PREFIX_RE.sub("", question)
    question = re.sub(r"^Question:\s*", "", question, flags=re.I)
    question = SOURCE_WORD_RE.sub("", question)
    question = question.replace("，，", "，").replace("，？", "？").strip(" ，,。；;：:")
    if question and not question.endswith(("？", "?", "。")):
        question += "？"
    return question


def usable_pair(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if BAD_QUESTION_RE.search(question):
        return False
    if BAD_GENERATED_QUESTION_RE.search(question.strip()):
        return False
    if len(question) > 180 or len(answer) > 900:
        return False
    if len(answer.strip(" ；;。")) < 24:
        return False
    if CROSS_REFERENCE_RE.search(answer) or TRAILING_INCOMPLETE_RE.search(answer):
        return False
    answer_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(answer_lines) > 1 and any(SECTION_LINE_RE.search(line) for line in answer_lines[1:]):
        return False
    numbered = NUMBERED_START_RE.search(answer.strip())
    if numbered:
        mark = numbered.group(1) or numbered.group(2) or numbered.group(3)
        if mark not in FIRST_NUMBER_MARKS:
            return False
    return True


def choose_long_answer(answer: str, evidence: str) -> str:
    answer = normalize(answer)
    evidence = normalize(evidence)
    if len(answer) < 80 and len(evidence) > len(answer) + 30 and len(evidence) <= 700:
        return evidence
    return answer


def row_question(row: dict) -> str:
    question = row.get("question") or ""
    if not question:
        prompt = row.get("prompt", "")
        question = prompt.splitlines()[-1] if prompt else ""
    return clean_question(question)


def iter_regulation_source_rows() -> Iterable[dict]:
    contexts = build_regulation_contexts(REG_DOCX)
    seen: set[tuple[str, str, str]] = set()
    for source_dir in REGQA_DIRS:
        if not source_dir.exists():
            continue
        for split in ("train", "valid", "test"):
            path = source_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            for row in jsonl_rows(path):
                task_type = row.get("task_type") or row.get("category") or ""
                if task_type not in REGULATION_TASKS:
                    continue
                question = row_question(row)
                evidence = normalize(row.get("evidence") or row.get("source_text") or row.get("answer", ""))
                answer = choose_long_answer(row.get("answer", ""), evidence)
                if not usable_pair(question, answer):
                    continue
                paragraph_id = row.get("paragraph_id", "")
                source_text = contexts.get(paragraph_id) or evidence or answer
                key = (task_type, question, answer)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "task_type": task_type,
                    "question": question,
                    "answer": answer,
                    "evidence": evidence or answer,
                    "source_text": source_text,
                    "source": row.get("source", REG_DOCX.name),
                    "source_path": row.get("source_path", str(REG_DOCX.relative_to(REPO_ROOT))),
                    "chapter": row.get("chapter", "规章制度"),
                    "page_number": row.get("page"),
                    "paragraph_id": paragraph_id,
                    "split": split,
                    "source_dataset": str(source_dir.relative_to(REPO_ROOT)),
                    "generation_method": row.get("generation_method", "expanded_current_regulation"),
                }


def collect_textbook_answer(lines: list, index: int, sentence: str) -> str | None:
    answer = None
    if LIST_LEAD_RE.search(sentence):
        answer = textbook_source.collect_list_answer(lines, index, sentence)
    if answer and "\n" in answer:
        return normalize(answer)

    answer = normalize(sentence)
    if CROSS_REFERENCE_RE.search(answer) or TRAILING_INCOMPLETE_RE.search(answer):
        return None
    if 28 <= len(answer) <= 320:
        return answer
    return None


def infer_context_subject(lines: list, index: int, sentence: str) -> str:
    candidates = [normalize(sentence)]
    for line in reversed(lines[max(0, index - 6) : index]):
        text = normalize(line.text)
        if not text:
            continue
        if re.match(r"^第[一二三四五六七八九十\d]+章", text):
            continue
        candidates.append(text)
    for text in candidates:
        match = CONTEXT_SUBJECT_RE.search(text)
        if match:
            subject = textbook_source.strip_leading_numbering(match.group(1)).strip(" ，,。；;：:（）()")
            subject = re.split(r"(是|为|指|用于)", subject, maxsplit=1)[0].strip(" ，,。；;：:（）()")
            if 2 <= len(subject) <= 28 and subject not in STOP_TOPICS and not BAD_TOPIC_RE.search(subject):
                return subject
    return ""


def textbook_topic(sentence: str, context_subject: str = "") -> str:
    topic = textbook_source.clean_topic(sentence)
    if topic == "该知识点":
        text = normalize(sentence)
        topic = re.split(r"(是指|称为|应|必须|需要|要求|包括|组成|分为|用于|具有|检查|检修)", text, maxsplit=1)[0]
        topic = textbook_source.strip_leading_numbering(topic).strip(" ，,。；;：:（）()")
    if (
        not topic
        or topic == "该知识点"
        or topic in STOP_TOPICS
        or len(topic) > 24
        or BAD_TOPIC_RE.search(topic)
        or topic.endswith(("不", "应", "须", "需", "要", "为", "是"))
        or re.search(r"(时|情况下|比较普遍|广泛采用|紧贴|滑行取流)$", topic)
    ):
        return context_subject or "该知识点"
    return topic


def textbook_question(task_type: str, topic: str, answer: str) -> str | None:
    topic = topic if topic != "该知识点" else "该知识点"
    if task_type == "textbook_definition_qa":
        return f"{topic}是什么？"
    if task_type == "textbook_operation_qa":
        if re.search(r"(开通运行前|接管运行|检查验收)", answer):
            return f"{topic}应满足哪些接管运行或检查验收条件？"
        return f"{topic}有哪些运行检修要求？"
    if task_type == "concept_explanation_qa":
        if DEFINITION_RE.search(answer):
            return f"{topic}是什么？"
        if "特点是" in answer or re.search(r"具有.{0,24}(?:特点|优点)", answer):
            return f"{topic}有哪些特点？"
        if re.search(r"(?<!不)用于", answer) and not re.search(r"(?:其|主要)?作用(?:是|为|在于)", answer):
            return f"{topic}用于哪些场景或位置？"
        if re.search(r"(?:其|主要)?作用(?:是|为|在于)|起到.{0,24}作用", answer):
            return f"{topic}有什么作用？"
        if re.search(r"(?:主要)?功能(?:是|包括|为)", answer):
            return f"{topic}有哪些功能？"
        return f"{topic}是什么？"
    if task_type == "textbook_extractive_qa":
        if re.search(r"(不得|严禁|禁止|不应)", answer):
            return f"{topic}有哪些禁止性或限制性要求？"
        if LIST_LEAD_RE.search(answer.splitlines()[0]):
            return f"{topic}包括哪些具体内容？"
        return f"{topic}有哪些具体要求？"
    return None


def is_textbook_operation_candidate(sentence: str, answer: str) -> bool:
    if NON_OPERATION_CONTEXT_RE.search(sentence):
        return False
    return bool(OPERATION_ACTION_RE.search(sentence))


def iter_textbook_base_rows() -> Iterable[dict]:
    seen: set[tuple[str, str, str]] = set()
    for path in textbook_source.BOOK_PATHS:
        book_title = path.parent.name
        lines = textbook_source.read_markdown_lines(path)
        for index, line in enumerate(lines):
            if textbook_source.has_figure_node_context(lines, index):
                continue
            for sentence in textbook_source.split_sentences(line.text):
                sentence = normalize(sentence)
                if not (28 <= len(sentence) <= 320):
                    continue
                if not TEXTBOOK_LONG_HINT_RE.search(sentence):
                    continue
                if textbook_source.should_skip_sentence(sentence):
                    continue
                answer = collect_textbook_answer(lines, index, sentence)
                if not answer:
                    continue
                context_subject = infer_context_subject(lines, index, sentence)
                topic = textbook_topic(sentence, context_subject)
                if topic == "该知识点":
                    continue
                task_types = ["textbook_extractive_qa"]
                if DEFINITION_RE.search(answer):
                    task_types.append("textbook_definition_qa")
                if is_textbook_operation_candidate(sentence, answer):
                    task_types.append("textbook_operation_qa")
                if CONCEPT_DIRECT_RE.search(answer):
                    task_types.append("concept_explanation_qa")
                for task_type in task_types:
                    question = textbook_question(task_type, topic, answer)
                    if not question or not usable_pair(question, answer):
                        continue
                    key = (task_type, question, answer)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "task_type": task_type,
                        "question": question,
                        "answer": answer,
                        "evidence": answer,
                        "source_text": textbook_source.context_for(lines, index, before=20, after=20),
                        "source": book_title,
                        "source_path": str(path.relative_to(REPO_ROOT)),
                        "page_number": line.page,
                        "line_number": line.number,
                        "chapter": book_title,
                        "generation_method": "expanded_current_textbook_long_context",
                    }


def make_mcq(question: str, answer: str, distractors: list[str], seed: str) -> tuple[str, str] | None:
    short_answer = answer.splitlines()[0]
    if not (30 <= len(short_answer) <= 180):
        return None
    choices = [short_answer]
    for item in distractors:
        item = item.splitlines()[0]
        if item != short_answer and 20 <= len(item) <= 180 and item not in choices:
            choices.append(item)
        if len(choices) == 4:
            break
    if len(choices) < 4:
        return None
    random.Random(seed).shuffle(choices)
    correct_label = "ABCD"[choices.index(short_answer)]
    options = "\n".join(f"{label}. {choice}" for label, choice in zip("ABCD", choices, strict=True))
    stem = question.rstrip("？?。")
    return f"{stem}，下列哪一项正确？\n{options}", f"{correct_label}. {short_answer}"


def derived_textbook_rows(base_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    answers = [row["answer"] for row in base_rows if 30 <= len(row["answer"].splitlines()[0]) <= 180]
    judgment_count = 0
    mcq_count = 0
    for index, row in enumerate(base_rows):
        lead = row["answer"].splitlines()[0]
        if 30 <= len(lead) <= 180:
            rows.append(
                {
                    **row,
                    "task_type": "textbook_judgment",
                    "question": f"判断题：{lead}",
                    "answer": "正确",
                    "generation_method": "expanded_current_textbook_judgment",
                }
            )
            judgment_count += 1
        mcq = make_mcq(
            row["question"],
            row["answer"],
            answers[index + 1 :] + answers[:index],
            seed=f"{row.get('source_path', '')}:{row.get('line_number', '')}:{row['question']}",
        )
        if mcq:
            question, answer = mcq
            rows.append(
                {
                    **row,
                    "task_type": "textbook_multiple_choice",
                    "question": question,
                    "answer": answer,
                    "generation_method": "expanded_current_textbook_multiple_choice",
                }
            )
            mcq_count += 1
        if judgment_count >= TARGET_PER_TASK and mcq_count >= TARGET_PER_TASK:
            break
    return rows


def take_by_task(rows: Iterable[dict], remaining: dict[str, int]) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        task_type = row["task_type"]
        if remaining.get(task_type, 0) <= counts[task_type]:
            continue
        selected.append(row)
        counts[task_type] += 1
        if all(counts[task] >= max(0, amount) for task, amount in remaining.items()):
            break
    return selected


def iter_terminology_small_rows() -> Iterable[dict]:
    seen: set[tuple[str, str, str]] = set()
    for row in jsonl_rows(TERM_PATH):
        term_zh = normalize(row.get("term_zh", ""))
        term_en = normalize(row.get("term_en", ""))
        if not term_zh or not term_en:
            continue
        category = row.get("domain_category", "")
        variants = [
            (
                "terminology_explanation",
                f"{term_zh}是什么意思？",
                f"{term_zh}对应的英文铁路术语为 {term_en}。",
            ),
            (
                "terminology_translation",
                f"{term_en} 对应的中文铁路术语是什么？",
                term_zh,
            ),
        ]
        for task_type, question, answer in variants:
            key = (task_type, question, answer)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "task_type": task_type,
                "question": question,
                "answer": answer,
                "evidence": f"{term_zh} | {term_en}",
                "source_text": f"{term_zh} | {term_en}",
                "source": row.get("source", TERM_DOCX.name),
                "source_path": str(TERM_DOCX.relative_to(REPO_ROOT)),
                "chapter": category,
                "domain_category": category or "铁道教育",
                "generation_method": "expanded_current_terminology_small",
                "metadata_json": {
                    "domain_category_en": row.get("domain_category_en", ""),
                    "domain_category_key": row.get("domain_category_key", ""),
                    "source_block": row.get("source_block"),
                    "abbreviation": row.get("abbreviation", ""),
                    "full_name_en": row.get("full_name_en", ""),
                },
            }


def current_approved_counts(db) -> dict[str, int]:
    rows = db.execute(
        select(CorpusItem.task_type, func.count())
        .where(CorpusItem.task_type.in_(TARGET_TASKS), CorpusItem.review_status == "approved")
        .group_by(CorpusItem.task_type)
    ).all()
    return {task_type: count for task_type, count in rows}


def build_rows(db, target_per_task: int) -> list[dict]:
    approved = current_approved_counts(db)
    remaining = {task: max(0, target_per_task - approved.get(task, 0)) for task in TARGET_TASKS}

    selected: list[dict] = []
    regulation_rows = take_by_task(iter_regulation_source_rows(), remaining)
    selected.extend(regulation_rows)
    for row in regulation_rows:
        remaining[row["task_type"]] -= 1

    textbook_base = list(iter_textbook_base_rows())
    textbook_rows = take_by_task(textbook_base + derived_textbook_rows(textbook_base), remaining)
    selected.extend(textbook_rows)
    for row in textbook_rows:
        remaining[row["task_type"]] -= 1

    term_rows = take_by_task(iter_terminology_small_rows(), remaining)
    selected.extend(term_rows)
    return selected


def to_db_rows(db, rows: list[dict]) -> list[dict]:
    reg_document = upsert_document(
        db,
        title="ECRL牵引供电设备运行维护管理办法",
        source_path=str(REG_DOCX.relative_to(REPO_ROOT)),
        document_type="regulation",
        domain_category=DOMAIN_CATEGORY,
        total_pages=None,
        metadata_json={"expansion_source": "current_non_terminology_corpus"},
    )
    textbook_documents = {
        path: upsert_document(
            db,
            title=path.parent.name,
            source_path=str(path.relative_to(REPO_ROOT)),
            document_type="textbook",
            domain_category=DOMAIN_CATEGORY,
            total_pages=None,
            metadata_json={"expansion_source": "current_non_terminology_corpus"},
        )
        for path in textbook_source.BOOK_PATHS
    }
    term_document = upsert_document(
        db,
        title="铁路中英文专业词汇",
        source_path=str(TERM_DOCX.relative_to(REPO_ROOT)),
        document_type="terminology",
        domain_category="铁道教育",
        total_pages=None,
        metadata_json={"expansion_source": "current_terminology_small_tasks"},
    )

    db_rows: list[dict] = []
    for row in rows:
        if row["task_type"].startswith("regulation_"):
            source_type = "regulation_qa"
        elif row["task_type"].startswith("terminology_"):
            source_type = "terminology"
        else:
            source_type = "textbook_original_md"
        source_path = row.get("source_path", "")
        document_id = reg_document.id
        if source_type == "textbook_original_md":
            document_id = textbook_documents[REPO_ROOT / source_path].id
        elif source_type == "terminology":
            document_id = term_document.id
        external_id = stable_id(
            "expanded_current",
            row["task_type"],
            source_path,
            str(row.get("paragraph_id") or row.get("line_number") or ""),
            row["question"],
            row["answer"],
        )
        db_rows.append(
            {
                "external_id": external_id,
                "source_type": source_type,
                "task_type": row["task_type"],
                "review_status": "pending",
                "domain_category": row.get("domain_category", DOMAIN_CATEGORY),
                "knowledge_category": (
                    "规章制度"
                    if source_type == "regulation_qa"
                    else "专业术语"
                    if source_type == "terminology"
                    else "教材"
                ),
                "question": row["question"],
                "answer": row["answer"],
                "evidence": row.get("evidence", row["answer"]),
                "source_text": row.get("source_text", row.get("evidence", row["answer"])),
                "original_question": row["question"],
                "original_answer": row["answer"],
                "document_id": document_id,
                "source_document": row.get("source", ""),
                "source_path": source_path,
                "chapter": row.get("chapter", ""),
                "page_number": row.get("page_number"),
                "quality_flags": ["human_review_required"],
                "metadata_json": {
                    "generation_method": row.get("generation_method", "expanded_current"),
                    "source_dataset": row.get("source_dataset", ""),
                    "paragraph_id": row.get("paragraph_id", ""),
                    "line_number": row.get("line_number"),
                    "split": row.get("split", ""),
                    **row.get("metadata_json", {}),
                },
            }
        )
    return db_rows


def insert_items_chunked(db, rows: list[dict], chunk_size: int = 500) -> int:
    inserted = 0
    for start in range(0, len(rows), chunk_size):
        inserted += insert_items(db, rows[start : start + chunk_size])
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand current non-terminology review corpus to larger long-context candidates.")
    parser.add_argument("--target-per-task", type=int, default=TARGET_PER_TASK)
    parser.add_argument("--clear", action="store_true", help="Delete non-approved rows in target non-terminology tasks before import.")
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

        rows = build_rows(db, args.target_per_task)
        db_rows = to_db_rows(db, rows)
        counts: dict[str, int] = defaultdict(int)
        for row in db_rows:
            counts[row["task_type"]] += 1

        if args.dry_run:
            print(json.dumps({"candidate_count": len(db_rows), "by_task_type": dict(sorted(counts.items()))}, ensure_ascii=False, indent=2))
            for row in db_rows[:20]:
                print(row["task_type"], row["question"], "=>", row["answer"][:220].replace("\n", " / "))
            return

        inserted = insert_items_chunked(db, db_rows)
        db.commit()
        print(json.dumps({"candidate_count": len(db_rows), "inserted": inserted, "by_task_type": dict(sorted(counts.items()))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
