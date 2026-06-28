from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.import_data import insert_items, stable_id, upsert_document  # noqa: E402


TASK_TYPE = "textbook_extractive_qa"
SOURCE_TYPE = "textbook_original_md"
DOMAIN_CATEGORY = "牵引供电"
KNOWLEDGE_CATEGORY = "教材"

BOOK_PATHS = [
    REPO_ROOT / "data" / "ocr" / "railway" / "牵引供电接触网运行与检修_14183109" / "combined.md",
    REPO_ROOT / "data" / "ocr" / "railway" / "高速铁路供电" / "combined.md",
]

SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
PAGE_RE = re.compile(r"^## Page\s+(\d+)\s*$")
SENTENCE_RE = re.compile(r"[^。！？；]+[。！？；]")
LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（(]?[一二三四五六七八九十\d]+[）)][.、）)]?|[一二三四五六七八九十\d]+[.、）)]|[A-Za-z][.、）)])\s*"
)
CHAPTER_RESIDUE_RE = re.compile(r"^\s*[一二三四五六七八九十\d]+[）)]\s*[^\s，,。；;：:]{1,12}\s*$")
NODE_NUMBER_RE = re.compile(
    r"(?:^|[，,。；;：:\s])节点\s*[A-Za-z0-9一二三四五六七八九十]+(?:[、,，]\s*[A-Za-z0-9一二三四五六七八九十]+)*(?:$|[，,。；;：:\s])"
)
FIGURE_CONTEXT_RE = re.compile(r"(图\s*\d|图\d|示意图|节点\s*\d|节点\s*[一二三四五六七八九十])")
RESPONSIBILITY_LIST_RE = re.compile(r"^(?:负责|监督、检查、指导|监督|检查|指导|协调).{0,45}(?:工作|管理|职责)[；;。]?$")
UNEXPANDED_SUMMARY_RE = re.compile(r"(?:分为|分)(?:[^。；;]{0,30})(?:几个|四个|三种|两类|两种|部分|方面|类型)[。；;]?$")
LIST_LEAD_RE = re.compile(r"(如下|下列|以下|具有以下[^。；;：:]{0,12}|主要内容如下|内容如下|规定如下|要求如下|包括如下|分述如下)[：:。；;]?$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\(?[一二三四五六七八九十\d]+\)?[.、）)]?|[A-Za-z][.、）)]|[-*+])\s*\S+")
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
TOPIC_SPLIT_RE = re.compile(
    r"(是指|可以|称为|简称|必须|需要|包括|分为|具有|用于|采用|系由|是|由|应|可)"
)
BAD_TEXT_RE = re.compile(
    r"(图书在版编目|ISBN|责任编辑|责任校对|责任印制|版权所有|定价|开本|印张|字数|版次|印次|"
    r"中国铁道出版社|铁路职工培训系列教材|购买铁道版图书|印制质量问题|读者服务部|网址|邮编|"
    r"TIELU|QIANYIN|GONGDIAN|GAOSU|CHINA RAILWAY|PUBLISHING HOUSE)",
    re.IGNORECASE,
)
EXTRACTIVE_HINT_RE = re.compile(
    r"(是指|是|包括|分为|由|具有|用于|采用|应|必须|需要|可|可以|称为|简称|检查|检修|运行|维护|故障|安全|要求)"
)
FIGURE_REF_RE = re.compile(r"(如图|见图|图\s*\d|图\d|表\s*\d|表\d)")
LIST_ITEM_RE = re.compile(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\(?\d+\)?[.、）)]?)\s*")
VARIABLE_TOPIC_RE = re.compile(r"^[A-Za-zＡ-Ｚａ-ｚ]\s*值$|^值$")
CONTEXT_DEPENDENT_TOPIC_RE = re.compile(r"^(其|该|此|这种|上述|下面|以上|以下|前者|后者|曲线地段|温度变化时)")
STOP_TOPICS = {"高速", "低速", "一般", "主要", "有关", "相关"}


@dataclass(frozen=True)
class MdLine:
    number: int
    page: int | None
    text: str


@dataclass(frozen=True)
class Candidate:
    book_title: str
    source_path: Path
    page: int | None
    line_number: int
    question: str
    answer: str
    source_text: str


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
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


def is_noise_line(text: str) -> bool:
    if not text:
        return True
    if text.startswith("![](") or text.startswith("<a id="):
        return True
    if BAD_TEXT_RE.search(text):
        return True
    if re.fullmatch(r"[#\s\dA-Za-z._-]+", text):
        return True
    return False


def read_markdown_lines(path: Path) -> list[MdLine]:
    lines: list[MdLine] = []
    current_page: int | None = None
    in_body = False
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        page_match = PAGE_RE.match(stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if re.match(r"^#{1,3}\s*第[一二三四五六七八九十0-9]+章\b", stripped):
            in_body = True
        if not in_body:
            continue
        text = normalize(stripped.lstrip("#").strip())
        if is_noise_line(text):
            continue
        lines.append(MdLine(number=number, page=current_page, text=text))
    return lines


def split_sentences(text: str) -> list[str]:
    sentences = [normalize(match.group(0)) for match in SENTENCE_RE.finditer(text)]
    if sentences:
        return sentences
    text = normalize(text)
    return [text] if text else []


def clean_topic(sentence: str) -> str:
    segment = TOPIC_SPLIT_RE.split(sentence, maxsplit=1)[0]
    segment = strip_leading_numbering(segment)
    for _ in range(2):
        segment = segment.strip(" ，,；;：:。！？（）()“”\"'《》")
        segment = re.sub(
            r"^(其中|同时|并|且|或|以及|对于|对|凡|有关|由|均|还|也|但|当|在|若|如|因此|另外|由于|而)",
            "",
            segment,
        )
    segment = re.sub(r"^所谓", "", segment)
    segment = re.split(r"(?:根据|按照|依据)", segment, maxsplit=1)[0]
    segment = segment.strip(" ，,；;：:。！？（）()“”\"'《》")
    segment = re.sub(r"系$", "", segment).strip()
    segment = re.sub(r"(也|还|则|虽然)$", "", segment).strip()
    if re.match(r"^(与|和|及|以及|向|为|对|受|按|以|这|该|其|图|表|温度|时间|条件)", segment):
        return "该知识点"
    if CONTEXT_DEPENDENT_TOPIC_RE.search(segment):
        return "该知识点"
    if VARIABLE_TOPIC_RE.fullmatch(segment):
        return "该知识点"
    if segment in STOP_TOPICS or "已普遍" in segment:
        return "该知识点"
    if re.search(r"(时|情况下)$", segment) and len(segment) <= 10:
        return "该知识点"
    if re.search(r"[，,。；;：:]", segment):
        return "该知识点"
    if not (2 <= len(segment) <= 28) or not HAN_RE.search(segment):
        return "该知识点"
    return segment


def is_context_dependent_question(question: str) -> bool:
    return bool(
        re.search(r"(下面与|曲线地段|温度变化时|主要有什么作用|值是什么|^值|^[A-Za-z]\s*值)", question)
    )


def make_question(topic: str, sentence: str) -> str | None:
    topic = topic.strip(" ，,；;：:。！？（）()“”\"'《》")
    if re.search(r"(不得|严禁|禁止|不应)", sentence):
        return f"{topic}有哪些禁止性要求？"
    if re.search(r"(应|必须|需要|要求|须|宜)", sentence):
        return f"{topic}应满足哪些要求？"
    if re.search(r"(包括|分为|由.+组成|内容)", sentence):
        if topic.endswith("主要内容"):
            return f"{topic}有哪些？"
        return f"{topic}包括哪些内容？"
    if re.search(r"(检查|检修|巡视|检测|维护|维修|故障|抢修|处理|运行)", sentence):
        return f"{topic}在运行检修中需要注意什么？"
    if re.search(r"(用于|采用|具有|作用|功能)", sentence):
        return f"{topic}有什么作用或特点？"
    if re.search(r"(是指|系由|称为|简称|定义|[^否]是)", sentence):
        return f"{topic}是什么？"
    return None


def should_skip_sentence(sentence: str) -> bool:
    if FIGURE_REF_RE.search(sentence):
        return True
    if CHAPTER_RESIDUE_RE.fullmatch(sentence):
        return True
    if NODE_NUMBER_RE.search(sentence):
        return True
    if RESPONSIBILITY_LIST_RE.search(sentence):
        return True
    if UNEXPANDED_SUMMARY_RE.search(sentence) and not LIST_LEAD_RE.search(sentence):
        return True
    if re.search(r"^[A-Za-zＡ-Ｚａ-ｚ]?\s*值是指", sentence):
        return True
    if LIST_ITEM_RE.match(sentence) and len(sentence) < 90:
        return True
    if sentence.startswith(("这种", "该", "其", "此", "上述", "以上", "以下")):
        return True
    return False


def has_figure_node_context(lines: list[MdLine], index: int) -> bool:
    start = max(0, index - 8)
    end = min(len(lines), index + 9)
    context = " ".join(line.text for line in lines[start:end])
    return bool(FIGURE_CONTEXT_RE.search(context) and NODE_NUMBER_RE.search(context))


def is_candidate_sentence(sentence: str) -> bool:
    if not (24 <= len(sentence) <= 220):
        return False
    if not HAN_RE.search(sentence):
        return False
    if BAD_TEXT_RE.search(sentence):
        return False
    if should_skip_sentence(sentence):
        return False
    if not EXTRACTIVE_HINT_RE.search(sentence):
        return False
    han_count = sum(1 for char in sentence if "\u4e00" <= char <= "\u9fff")
    return han_count / max(1, len(sentence)) >= 0.45


def collect_list_answer(lines: list[MdLine], index: int, lead_sentence: str) -> str | None:
    if not LIST_LEAD_RE.search(lead_sentence):
        return lead_sentence

    chunks = [lead_sentence]
    for line in lines[index + 1 : index + 25]:
        text = normalize(line.text)
        if not text:
            continue
        if re.match(r"^第[一二三四五六七八九十\d]+章", text):
            break
        if LIST_ITEM_RE.match(text):
            chunks.append(text)
            continue
        if len(chunks) > 1:
            break
        break
    return "\n".join(chunks) if len(chunks) > 1 else None


def context_for(lines: list[MdLine], index: int, before: int, after: int) -> str:
    start = max(0, index - before)
    end = min(len(lines), index + after + 1)
    chunks: list[str] = []
    for line in lines[start:end]:
        page = f"第 {line.page} 页" if line.page else "未知页"
        chunks.append(f"L{line.number} | {page} | {line.text}")
    return "\n".join(chunks)


def build_candidates(path: Path, *, before: int, after: int) -> list[Candidate]:
    book_title = path.parent.name
    lines = read_markdown_lines(path)
    candidates: list[Candidate] = []
    seen_answers: set[str] = set()
    for index, line in enumerate(lines):
        if has_figure_node_context(lines, index):
            continue
        for sentence in split_sentences(line.text):
            if not is_candidate_sentence(sentence):
                continue
            answer = collect_list_answer(lines, index, sentence)
            if not answer:
                continue
            if len(answer.strip(" ；;。")) < 35:
                continue
            if sentence in seen_answers:
                continue
            seen_answers.add(sentence)
            topic = clean_topic(sentence)
            if topic == "该知识点":
                continue
            question = make_question(topic, sentence)
            if not question:
                continue
            if is_context_dependent_question(question):
                continue
            candidates.append(
                Candidate(
                    book_title=book_title,
                    source_path=path,
                    page=line.page,
                    line_number=line.number,
                    question=question,
                    answer=answer,
                    source_text=context_for(lines, index, before, after),
                )
            )
    return candidates


def row_for(document_id: int, candidate: Candidate) -> dict:
    rel_path = str(candidate.source_path.relative_to(REPO_ROOT))
    external_id = stable_id(SOURCE_TYPE, rel_path, str(candidate.line_number), candidate.answer)
    return {
        "external_id": external_id,
        "source_type": SOURCE_TYPE,
        "task_type": TASK_TYPE,
        "review_status": "pending",
        "domain_category": DOMAIN_CATEGORY,
        "knowledge_category": KNOWLEDGE_CATEGORY,
        "question": candidate.question,
        "answer": candidate.answer,
        "evidence": candidate.answer,
        "source_text": candidate.source_text,
        "original_question": candidate.question,
        "original_answer": candidate.answer,
        "document_id": document_id,
        "source_document": candidate.book_title,
        "source_path": rel_path,
        "chapter": "",
        "page_number": candidate.page,
        "quality_flags": [],
        "metadata_json": {
            "skill_file": "/ds1/workspace/ai/multilingual-railway-llm-edu/skills/all_qa_skills.md",
            "skill_task_type": TASK_TYPE,
            "source_kind": "original_combined_md",
            "line_number": candidate.line_number,
            "context_before_lines": 20,
            "context_after_lines": 20,
            "generation_method": "rule_based_textbook_extractive_from_original_md",
        },
        "reviewer": "",
        "review_comment": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import textbook_extractive_qa items from the original combined.md textbook files."
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to import across all books after per-book sampling. 0 means no limit.")
    parser.add_argument("--limit-per-book", type=int, default=100, help="Maximum rows to import from each book. 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--context-before", type=int, default=20)
    parser.add_argument("--context-after", type=int, default=20)
    args = parser.parse_args()

    load_env()
    all_candidates: list[Candidate] = []
    for path in BOOK_PATHS:
        book_candidates = build_candidates(path, before=args.context_before, after=args.context_after)
        book_candidates.sort(key=lambda item: (item.page or 0, item.line_number, item.question))
        if args.limit_per_book > 0:
            book_candidates = book_candidates[: args.limit_per_book]
        all_candidates.extend(book_candidates)

    all_candidates.sort(key=lambda item: (item.book_title, item.page or 0, item.line_number, item.question))
    if args.limit > 0:
        all_candidates = all_candidates[: args.limit]

    if args.dry_run:
        preview = [
            {
                "book": item.book_title,
                "page": item.page,
                "line": item.line_number,
                "question": item.question,
                "answer": item.answer,
            }
            for item in all_candidates[:10]
        ]
        print(json.dumps({"candidate_count": len(all_candidates), "preview": preview}, ensure_ascii=False, indent=2))
        return

    with SessionLocal() as db:
        documents = {}
        for path in BOOK_PATHS:
            rel_path = str(path.relative_to(REPO_ROOT))
            documents[path] = upsert_document(
                db,
                title=path.parent.name,
                source_path=rel_path,
                document_type="textbook",
                domain_category=DOMAIN_CATEGORY,
                total_pages=None,
                metadata_json={
                    "source_kind": "original_combined_md",
                    "import_script": "scripts/import_textbook_extractive_from_combined_md.py",
                },
            )
        rows = [row_for(documents[candidate.source_path].id, candidate) for candidate in all_candidates]
        inserted = insert_items(db, rows)
        db.commit()

    print(
        json.dumps(
            {
                "task_type": TASK_TYPE,
                "source_type": SOURCE_TYPE,
                "candidates": len(all_candidates),
                "inserted": inserted,
                "source_files": [str(path.relative_to(REPO_ROOT)) for path in BOOK_PATHS],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
