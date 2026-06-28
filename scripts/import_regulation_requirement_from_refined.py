from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.import_data import build_regulation_contexts, insert_items, jsonl_rows, stable_id, upsert_document  # noqa: E402


TASK_TYPE = "regulation_requirement_qa"
SOURCE_TYPE = "regulation_qa"
DOMAIN_CATEGORY = "牵引供电"
KNOWLEDGE_CATEGORY = "规章制度"
DATASET_DIR = REPO_ROOT / "data" / "domain_regqa_refined"
SOURCE_DOCX = REPO_ROOT / "data" / "corpus" / "railway" / "规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx"
SOURCE_TITLE = "ECRL牵引供电设备运行维护管理办法"

SPACE_RE = re.compile(r"\s+")
REQUIREMENT_RE = re.compile(r"(应|须|必须|不得|严禁|禁止|允许|需|需要|要求|应当|宜|正常|预留|进行|保持)")
CONTEXT_LEADING_RE = re.compile(r"^(在|当|若|如|对|对于|凡|其中|同时|因此|由于|由|按|根据|按照|依据)")
GENERIC_SUBJECTS = {"设备", "设施", "系统", "工作", "情况", "柜体", "部件", "材料", "人员"}
FIGURE_TABLE_RE = re.compile(r"(如图|见图|图\s*\d|图\d|表\s*\d|表\d)")
PROCESS_FLOW_RE = re.compile(r"(审批|批准|审查|申请报告|承接试运行|试运行任务|组织实施)")
RESPONSIBILITY_FLOW_RE = re.compile(r"(主要职责|职责|负责|责任分工|监督、检查、指导|协调[^。；;]*管理工作)")
TEST_TABLE_RE = re.compile(r"(测试结果|试验项目|建议周期|初始值|出厂值|测得值|产品技术条件|相差|差别)")
SINGLE_ACTION_RE = re.compile(
    r"应(?:查明原因|及时组织实施|符合产品技术条件的规定|放置在所内|预留|选择在[^。；;]+进行|由[^。；;]+批准|由[^。；;]+审查)[。；;]?$"
)
LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（(]?[一二三四五六七八九十\d]+[）)][.、）)]?|[一二三四五六七八九十\d]+[.、）)]|[A-Za-z][.、）)])\s*"
)
CHAPTER_RESIDUE_RE = re.compile(r"^\s*[一二三四五六七八九十\d]+[）)]\s*[^\s，,。；;：:]{1,12}\s*$")
NODE_NUMBER_RE = re.compile(r"(?:^|[，,。；;：:\s])节点\s*[A-Za-z0-9一二三四五六七八九十]+(?:$|[，,。；;：:\s])")
LIST_LEAD_RE = re.compile(r"(如下|下列|以下|具有以下[^。；;：:]{0,12}|主要内容如下|内容如下|规定如下|要求如下|包括如下|分述如下)[：:。；;]?$")
LIST_ITEM_RE = re.compile(r"^\s*(?:L\d+\s*\|\s*[^|]+\s*\|\s*)?(?:[①②③④⑤⑥⑦⑧⑨⑩]|\(?[一二三四五六七八九十\d]+\)?[.、）)]?|[A-Za-z][.、）)]|[-*+])\s*\S+")
GREEK_CHARS = "α-ωΑ-ΩβγδλμΩ"
LATEX_GREEK = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\Omega": "Ω",
}
LATEX_SYMBOLS = {
    r"\sim": "~",
    r"\pm": "±",
    r"\times": "×",
    r"\cdot": "·",
    r"\,": " ",
}


def normalize(text: str) -> str:
    text = (text or "").replace("\u3000", " ").replace("\xa0", " ")
    for raw, converted in LATEX_GREEK.items():
        text = text.replace(raw, converted)
    for raw, converted in LATEX_SYMBOLS.items():
        text = text.replace(raw, converted)
    text = text.replace(r"\%", "%")
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = re.sub(r"\^\{?\\?circ\}?", "°", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[()]|\\(?=\s|$)", " ", text)
    text = text.replace("／", "/").replace("∕", "/").replace("⁄", "/")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])/(?=[\u4e00-\u9fff])", "、", text)
    text = re.sub(rf"(?<![A-Za-z0-9{GREEK_CHARS}])/(?![A-Za-z0-9{GREEK_CHARS}])", "", text)
    return SPACE_RE.sub(" ", text).strip()


def strip_leading_numbering(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = LEADING_NUMBER_RE.sub("", current).strip()
    return current


def split_subject_predicate(answer: str) -> tuple[str, str] | None:
    answer = normalize(answer)
    match = re.search(r"(应当|必须|不得|严禁|禁止|应|须|需|需要|宜|允许|正常|预留|进行|保持)", answer)
    if not match:
        return None
    subject = strip_leading_numbering(answer[: match.start()].strip(" ，,；;：:"))
    predicate = answer[match.start() :].strip()
    subject = re.sub(r"^(为保证[^，,。；;]+[，,])", "", subject).strip(" ，,；;：:")
    return subject, predicate


def subject_is_usable(subject: str) -> bool:
    subject = strip_leading_numbering(subject)
    if not subject or len(subject) < 3 or len(subject) > 42:
        return False
    if CHAPTER_RESIDUE_RE.fullmatch(subject):
        return False
    if NODE_NUMBER_RE.search(subject):
        return False
    if CONTEXT_LEADING_RE.search(subject):
        return False
    if subject in GENERIC_SUBJECTS:
        return False
    if subject.count("、") >= 4 and not re.search(r"(等设备|等设施|等资料|等项目)$", subject):
        return False
    return True


def is_requirement_answer_usable(answer: str) -> bool:
    lead = normalize(answer).splitlines()[0]
    if PROCESS_FLOW_RE.search(lead):
        return False
    if RESPONSIBILITY_FLOW_RE.search(lead):
        return False
    if TEST_TABLE_RE.search(lead) and len(lead) < 90:
        return False
    if SINGLE_ACTION_RE.search(lead):
        return False
    if len(lead) < 45 and not re.search(r"(不得|严禁|禁止|不应|不宜)", lead):
        return False
    return True


def make_question(answer: str) -> str | None:
    answer = normalize(answer)
    lead = answer.splitlines()[0]
    if FIGURE_TABLE_RE.search(answer):
        return None
    if CHAPTER_RESIDUE_RE.fullmatch(lead):
        return None
    if NODE_NUMBER_RE.search(answer):
        return None
    if len(lead) < 18 or len(lead) > 260:
        return None
    if not is_requirement_answer_usable(answer):
        return None
    if LIST_LEAD_RE.search(lead):
        subject = strip_leading_numbering(LIST_LEAD_RE.sub("", lead).strip(" ，,；;：:。"))
        if subject_is_usable(subject):
            if re.search(r"(要求|规定|应|须|必须|不得|严禁|禁止)", lead):
                return f"{subject}有哪些具体要求？"
            return f"{subject}包括哪些具体内容？"
        return None
    if not REQUIREMENT_RE.search(answer):
        return None

    window_match = re.search(r"为保证([^，,。；;]+)，在列车运行图中须预留([^。；;]+)", answer)
    if window_match:
        return f"{window_match.group(2)}在列车运行图中应如何安排？"

    before_operation_match = re.search(r"(.+?)，?在投运前应进行([^。；;]+)", answer)
    if before_operation_match:
        subject = before_operation_match.group(1).strip(" ，,；;：:")
        if subject_is_usable(subject):
            return f"{subject}在投运前应进行哪些检查或试验？"

    normal_match = re.search(r"(.+?)应正常[。；;]?$", answer)
    if normal_match:
        subject = normal_match.group(1).strip(" ，,；;：:")
        if "通风、空调、安全环境监控、消防、照明" in subject:
            return "牵引变电所辅助设备应保持什么状态？"
        if subject_is_usable(subject):
            return f"{subject}应保持什么状态？"

    result = split_subject_predicate(answer)
    if not result:
        return None
    subject, _predicate = result
    if not subject_is_usable(subject):
        if subject == "柜体":
            return "设备柜体的外观、紧固和接地状态应满足哪些要求？"
        return None

    if re.search(r"(试验|测量|检查|检测)", answer):
        return f"{subject}应进行哪些检查或试验？"
    if re.search(r"(不得|严禁|禁止)", answer):
        return f"{subject}有哪些禁止性要求？"
    return f"{subject}应满足哪些要求？"


def strip_context_prefix(text: str) -> str:
    return re.sub(r"^L\d+\s*\|\s*[^|]+\s*\|\s*", "", text).strip()


def expand_list_answer(answer: str, source_text: str) -> str | None:
    answer = normalize(answer)
    if not LIST_LEAD_RE.search(answer):
        return answer

    lines = [strip_context_prefix(normalize(line)) for line in (source_text or "").splitlines()]
    chunks = [answer]
    try:
        start = next(i for i, line in enumerate(lines) if answer in line or line in answer)
    except StopIteration:
        start = -1
    for text in lines[start + 1 : start + 25]:
        if not text:
            continue
        if LIST_ITEM_RE.match(text):
            chunks.append(text)
            continue
        if len(chunks) > 1:
            break
        break
    return "\n".join(chunks) if len(chunks) > 1 else None


def build_rows(limit: int = 0) -> list[dict]:
    contexts = build_regulation_contexts(SOURCE_DOCX)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for split in ("train", "valid", "test"):
        path = DATASET_DIR / f"{split}.jsonl"
        for row in jsonl_rows(path):
            if row.get("category") != TASK_TYPE:
                continue
            answer = normalize(row.get("answer", ""))
            source_text = contexts.get(row.get("paragraph_id", ""), normalize(row.get("evidence", answer)))
            answer = expand_list_answer(answer, source_text)
            if not answer:
                continue
            question = make_question(answer)
            if not question:
                continue
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)
            paragraph_id = row.get("paragraph_id", "")
            rows.append(
                {
                    "split": split,
                    "paragraph_id": paragraph_id,
                    "question": question,
                    "answer": answer,
                    "evidence": normalize(row.get("evidence", answer)),
                    "source_text": source_text,
                    "source": row.get("source", SOURCE_DOCX.name),
                    "answer_start": row.get("answer_start"),
                    "generation_method": "rule_based_requirement_regenerated_from_rejection_feedback",
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate regulation_requirement_qa rows with stricter question quality rules.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    candidates = build_rows(limit=args.limit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "candidate_count": len(candidates),
                    "preview": candidates[:12],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    with SessionLocal() as db:
        document = upsert_document(
            db,
            title=SOURCE_TITLE,
            source_path=str(SOURCE_DOCX.relative_to(REPO_ROOT)),
            document_type="regulation",
            domain_category=DOMAIN_CATEGORY,
            total_pages=None,
            metadata_json={
                "dataset": str(DATASET_DIR.relative_to(REPO_ROOT)),
                "import_script": "scripts/import_regulation_requirement_from_refined.py",
            },
        )
        db_rows = []
        for candidate in candidates:
            db_rows.append(
                {
                    "external_id": stable_id("regqa_req", candidate["paragraph_id"], candidate["question"], candidate["answer"]),
                    "source_type": SOURCE_TYPE,
                    "task_type": TASK_TYPE,
                    "review_status": "pending",
                    "domain_category": DOMAIN_CATEGORY,
                    "knowledge_category": KNOWLEDGE_CATEGORY,
                    "question": candidate["question"],
                    "answer": candidate["answer"],
                    "evidence": candidate["evidence"],
                    "source_text": candidate["source_text"],
                    "original_question": candidate["question"],
                    "original_answer": candidate["answer"],
                    "document_id": document.id,
                    "source_document": candidate["source"],
                    "source_path": document.source_path,
                    "chapter": "",
                    "page_number": None,
                    "quality_flags": [],
                    "metadata_json": {
                        "split": candidate["split"],
                        "paragraph_id": candidate["paragraph_id"],
                        "answer_start": candidate["answer_start"],
                        "generation_method": candidate["generation_method"],
                        "skill_file": "/ds1/workspace/ai/multilingual-railway-llm-edu/skills/all_qa_skills.md",
                        "skill_task_type": TASK_TYPE,
                    },
                    "reviewer": "",
                    "review_comment": "",
                }
            )
        inserted = insert_items(db, db_rows)
        db.commit()

    print(
        json.dumps(
            {
                "task_type": TASK_TYPE,
                "candidates": len(candidates),
                "inserted": inserted,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
