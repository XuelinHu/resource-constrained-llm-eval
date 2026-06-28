from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.import_data import insert_items, stable_id, upsert_document  # noqa: E402
from app.models import CorpusItem  # noqa: E402
import import_textbook_extractive_from_combined_md as textbook_source  # noqa: E402


TASK_TYPE = "textbook_judgment"
SOURCE_TYPE = "textbook_original_md"
DOMAIN_CATEGORY = "牵引供电"
KNOWLEDGE_CATEGORY = "教材"
GENERATION_METHOD = "rebuilt_textbook_judgment_v4"
OUTPUT_DIR = REPO_ROOT / "data" / "textbook_judgment_v4"

RULE_HINT_RE = re.compile(
    r"(应|须|必须|需要|要求|不得|严禁|禁止|不应|宜|可|可以|"
    r"包括|组成|分为|分成|由.+组成|是指|称为|简称|用于|具有|作用|特点|"
    r"检查|检修|巡视|检测|维护|维修|运行|验收|接管|安全)"
)
BAD_FRAGMENT_RE = re.compile(
    r"(如下|下列|以下|包括以下|分为下列|内容如下|要求如下|规定如下|"
    r"如图所示|见图|如表|见表|满足|包括|分为)[：:；;，,。]*$"
)
BAD_OCR_RE = re.compile(
    r"(<table|</td>|</tr>|!\[\]|☐|□|�|图\s*\d|表\s*\d|如图|见图|所示|"
    r"地瓦|线插头|故测仪|有\d+根\d|IEC\d|重复[（(]\d+[）)]|Document generated|Anna)"
)
BAD_START_RE = re.compile(r"^[（(]?(同时|其中|因此|另外|以下|如下|这|该|其|上述|前者|后者|若|如|如果|当|图中)")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?![A-Za-z0-9])")
NUMBER_UNIT_RE = re.compile(
    r"^(?:\s*(?:mm|cm|m|km|km/h|kV|V|A|Ω|kg|h|min|s|%|‰|mm²|m²|"
    r"台|根|条|组|个|处|次|遍|年|月|日|小时|分钟|秒|倍|级|度|℃|人|处所))"
)
BUSINESS_NUMBER_CONTEXT_RE = re.compile(
    r"(不大于|不小于|不应小于|不得超过|允许|偏差|范围|距离|高度|长度|宽度|厚度|"
    r"半径|速度|温度|电压|电流|电阻|负荷|面积|截面积|磨耗|周期|频次|每隔|"
    r"连续|不少于|不超过|大于|小于|至少|约|距|重|长|高)"
)
NUMBERED_LEAD_RE = re.compile(
    r"^\s*(?:[（(]\d{1,3}[）)]|\d{1,3}[.、）)]|[一二三四五六七八九十]{1,3}[、.）)]|[①②③④⑤⑥⑦⑧⑨⑩])"
)

STRICT_POLARITY_REPLACEMENTS = [
    (re.compile(r"应将"), "不应将"),
    (re.compile(r"应采用"), "不应采用"),
    (re.compile(r"应设置"), "不应设置"),
    (re.compile(r"应满足"), "不应满足"),
    (re.compile(r"应位于"), "不应位于"),
    (re.compile(r"不得超过"), "可以超过"),
    (re.compile(r"不应小于"), "应小于"),
    (re.compile(r"不大于"), "大于"),
    (re.compile(r"不小于"), "小于"),
]

STRICT_OBJECT_REPLACEMENTS = [
    (re.compile(r"电力机车上部的受电弓滑动接触"), "电力机车上部的承力索滑动接触"),
    (re.compile(r"接触线通过吊弦悬挂在承力索上"), "接触线通过支柱悬挂在承力索上"),
    (re.compile(r"接触线是否进行补偿"), "承力索是否进行补偿"),
    (re.compile(r"按受电弓运行轨迹"), "按承力索运行轨迹"),
    (re.compile(r"受电弓中心的相对位置"), "承力索中心的相对位置"),
    (re.compile(r"定位线夹把接触线"), "定位线夹把承力索"),
    (re.compile(r"接触线在跨距中间的弛度"), "承力索在跨距中间的弛度"),
]


@dataclass(frozen=True)
class BaseCandidate:
    book_title: str
    source_path: Path
    page: int | None
    line_number: int
    proposition: str
    evidence: str
    source_text: str


@dataclass(frozen=True)
class JudgmentCandidate:
    base: BaseCandidate
    label: str
    proposition: str
    perturbation_type: str


def normalize(text: str) -> str:
    return textbook_source.normalize(text or "")


def is_good_proposition(sentence: str) -> bool:
    sentence = normalize(sentence).strip()
    if not (35 <= len(sentence) <= 180):
        return False
    if NUMBERED_LEAD_RE.search(sentence):
        return False
    if not RULE_HINT_RE.search(sentence):
        return False
    if BAD_OCR_RE.search(sentence) or BAD_FRAGMENT_RE.search(sentence):
        return False
    if BAD_START_RE.search(sentence):
        return False
    if textbook_source.should_skip_sentence(sentence):
        return False
    if sentence.count("，") + sentence.count(",") > 7:
        return False
    han_count = sum(1 for char in sentence if "\u4e00" <= char <= "\u9fff")
    if han_count / max(1, len(sentence)) < 0.48:
        return False
    return True


def is_business_number(sentence: str, start: int, end: int) -> bool:
    left = sentence[max(0, start - 14) : start]
    right = sentence[end : min(len(sentence), end + 14)]
    near = left + sentence[start:end] + right
    if not (BUSINESS_NUMBER_CONTEXT_RE.search(near) or NUMBER_UNIT_RE.search(right)):
        return False
    return True


def is_structural_number(sentence: str, start: int, end: int) -> bool:
    raw = sentence[start:end]
    left = sentence[max(0, start - 4) : start]
    right = sentence[end : min(len(sentence), end + 4)]
    compact_left = left.replace(" ", "")
    compact_right = right.replace(" ", "")

    if re.search(r"[（(]\s*$", left) and re.search(r"^\s*[）)]", right):
        return True
    if re.search(r"(?:^|[。；;，,\s])\s*$", left) and re.search(r"^\s*[.、）)]", right):
        return True
    if compact_left.endswith(("第", "图", "表", "L", "节点", "Page", "page")):
        return True
    if compact_right.startswith(("：", ":", "-", "—")) or compact_left.endswith(("：", ":", "-", "—")):
        return True
    if compact_right.startswith(("～", "~", "至", "到")) or compact_left.endswith(("～", "~", "至", "到")):
        return True
    if re.search(r"[A-Za-z]$", compact_left) or re.search(r"^[A-Za-z]", compact_right):
        return True
    if re.fullmatch(r"\d{1,2}", raw) and re.search(r"^[、,，]\s*\d", right):
        return True
    return False


def replacement_number(raw: str, value: float, context: str) -> str:
    if re.search(r"(月|月底|月份)", context):
        month = int(value)
        return str(month + 1 if month < 12 else month - 1)
    if value == 0:
        return "1"
    if value < 10:
        return str(int(value + 1)) if value.is_integer() else f"{value + 0.5:g}"
    new_value = str(int(round(value * 1.2)))
    if new_value == raw:
        new_value = str(int(value) + 1)
    return new_value


def alter_number(sentence: str) -> str | None:
    matches = list(NUMBER_RE.finditer(sentence))
    for match in matches:
        raw = match.group(1)
        start = match.start(1)
        end = match.end(1)
        if is_structural_number(sentence, start, end):
            continue
        if not is_business_number(sentence, start, end):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        context = sentence[max(0, start - 12) : min(len(sentence), end + 12)]
        new_value = replacement_number(raw, value, context)
        return sentence[: match.start(1)] + new_value + sentence[match.end(1) :]
    return None


def alter_polarity(sentence: str) -> str | None:
    for pattern, replacement in STRICT_POLARITY_REPLACEMENTS:
        if pattern.search(sentence):
            return pattern.sub(replacement, sentence, count=1)
    return None


def alter_object(sentence: str) -> str | None:
    for pattern, replacement in STRICT_OBJECT_REPLACEMENTS:
        if pattern.search(sentence):
            return pattern.sub(replacement, sentence, count=1)
    return None


def make_false_proposition(sentence: str, *, allow_semantic_perturbations: bool = False) -> tuple[str, str] | None:
    numbered = alter_number(sentence)
    if numbered and numbered != sentence:
        return numbered, "number_replacement"
    if allow_semantic_perturbations:
        polarity = alter_polarity(sentence)
        if polarity and polarity != sentence:
            return polarity, "polarity_flip"
        obj = alter_object(sentence)
        if obj and obj != sentence:
            return obj, "object_replacement"
    return None


def build_base_candidates(context_before: int, context_after: int) -> list[BaseCandidate]:
    candidates: list[BaseCandidate] = []
    seen: set[str] = set()
    for path in textbook_source.BOOK_PATHS:
        book_title = path.parent.name
        lines = textbook_source.read_markdown_lines(path)
        for index, line in enumerate(lines):
            if textbook_source.has_figure_node_context(lines, index):
                continue
            for sentence in textbook_source.split_sentences(line.text):
                proposition = normalize(sentence).strip(" ；;")
                if not proposition.endswith("。"):
                    proposition = proposition.rstrip("。") + "。"
                if not is_good_proposition(proposition):
                    continue
                if proposition in seen:
                    continue
                seen.add(proposition)
                candidates.append(
                    BaseCandidate(
                        book_title=book_title,
                        source_path=path,
                        page=line.page,
                        line_number=line.number,
                        proposition=proposition,
                        evidence=proposition,
                        source_text=textbook_source.context_for(
                            lines, index, before=context_before, after=context_after
                        ),
                    )
                )
    return candidates


def build_judgments(
    base_candidates: list[BaseCandidate],
    target: int,
    true_ratio: float,
    seed: int,
    *,
    allow_semantic_perturbations: bool = False,
) -> list[JudgmentCandidate]:
    rng = random.Random(seed)
    shuffled = list(base_candidates)
    rng.shuffle(shuffled)

    true_target = int(round(target * true_ratio))
    false_target = target - true_target
    selected: list[JudgmentCandidate] = []
    true_count = 0
    false_count = 0

    for base in shuffled:
        if false_count < false_target:
            altered = make_false_proposition(
                base.proposition,
                allow_semantic_perturbations=allow_semantic_perturbations,
            )
            if altered:
                proposition, perturbation_type = altered
                if is_good_proposition(proposition):
                    selected.append(
                        JudgmentCandidate(
                            base=base,
                            label="错误",
                            proposition=proposition,
                            perturbation_type=perturbation_type,
                        )
                    )
                    false_count += 1
                    continue

        if true_count < true_target:
            selected.append(
                JudgmentCandidate(
                    base=base,
                    label="正确",
                    proposition=base.proposition,
                    perturbation_type="none",
                )
            )
            true_count += 1

        if true_count >= true_target and false_count >= false_target:
            break

    selected.sort(key=lambda item: (item.base.book_title, item.base.page or 0, item.base.line_number, item.label))
    return selected


def question_for(proposition: str) -> str:
    proposition = proposition.rstrip("。；;：:")
    return f"判断题：{proposition}。请判断正误。"


def to_record(candidate: JudgmentCandidate) -> dict:
    rel_path = str(candidate.base.source_path.relative_to(REPO_ROOT))
    ocr_page_path = ""
    if candidate.base.page is not None:
        ocr_page_path = str(candidate.base.source_path.parent / "pages" / f"page_{candidate.base.page:04d}.md")
        ocr_page_path = str(Path(ocr_page_path).relative_to(REPO_ROOT))
    source_key = f"{rel_path}:{candidate.base.line_number}:{candidate.label}:{candidate.proposition}"
    return {
        "id": stable_id("textbook_judgment_v4", source_key),
        "task_type": TASK_TYPE,
        "domain_category": DOMAIN_CATEGORY,
        "knowledge_category": KNOWLEDGE_CATEGORY,
        "question": question_for(candidate.proposition),
        "answer": candidate.label,
        "label": candidate.label,
        "evidence": candidate.base.evidence,
        "source_text": candidate.base.source_text,
        "source": candidate.base.book_title,
        "source_path": rel_path,
        "page": candidate.base.page,
        "line_number": candidate.base.line_number,
        "ocr_page_path": ocr_page_path,
        "generation_method": GENERATION_METHOD,
        "perturbation_type": candidate.perturbation_type,
        "original_proposition": candidate.base.proposition,
    }


def write_outputs(records: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "candidates.jsonl"
    csv_path = OUTPUT_DIR / "human_review.csv"
    summary_path = OUTPUT_DIR / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "id",
        "task_type",
        "label",
        "question",
        "answer",
        "evidence",
        "source",
        "page",
        "line_number",
        "perturbation_type",
        "original_proposition",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})

    by_label: dict[str, int] = {}
    by_perturbation: dict[str, int] = {}
    for record in records:
        by_label[record["label"]] = by_label.get(record["label"], 0) + 1
        by_perturbation[record["perturbation_type"]] = by_perturbation.get(record["perturbation_type"], 0) + 1
    summary = {
        "task_type": TASK_TYPE,
        "generation_method": GENERATION_METHOD,
        "count": len(records),
        "by_label": dict(sorted(by_label.items())),
        "by_perturbation_type": dict(sorted(by_perturbation.items())),
        "jsonl": str(jsonl_path.relative_to(REPO_ROOT)),
        "csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def quarantine_old_items(db) -> int:
    old_items = list(
        db.scalars(
            select(CorpusItem).where(
                CorpusItem.task_type == TASK_TYPE,
                CorpusItem.review_status.in_(["pending", "needs_revision", "rejected"]),
            )
        )
    )
    for item in old_items:
        flags = list(item.quality_flags or [])
        for flag in ("superseded_by_textbook_judgment_v4", "judgment_template_defect"):
            if flag not in flags:
                flags.append(flag)
        metadata = dict(item.metadata_json or {})
        metadata["superseded_by"] = GENERATION_METHOD
        item.review_status = "deleted"
        item.quality_flags = flags
        item.metadata_json = metadata
        item.review_comment = (item.review_comment or "旧版教材判断题模板缺陷，已由严格数值扰动版整体重建替代。").strip()
    return len(old_items)


def import_records(db, records: list[dict]) -> int:
    documents = {
        path: upsert_document(
            db,
            title=path.parent.name,
            source_path=str(path.relative_to(REPO_ROOT)),
            document_type="textbook",
            domain_category=DOMAIN_CATEGORY,
            total_pages=None,
            metadata_json={
                "source_kind": "original_combined_md",
                "import_script": "scripts/rebuild_textbook_judgment_v2.py",
            },
        )
        for path in textbook_source.BOOK_PATHS
    }

    rows: list[dict] = []
    for record in records:
        source_path = record["source_path"]
        document_id = documents[REPO_ROOT / source_path].id
        rows.append(
            {
                "external_id": record["id"],
                "source_type": SOURCE_TYPE,
                "task_type": TASK_TYPE,
                "review_status": "pending",
                "domain_category": record["domain_category"],
                "knowledge_category": record["knowledge_category"],
                "question": record["question"],
                "answer": record["answer"],
                "evidence": record["evidence"],
                "source_text": record["source_text"],
                "original_question": record["question"],
                "original_answer": record["answer"],
                "document_id": document_id,
                "source_document": record["source"],
                "source_path": source_path,
                "chapter": record["source"],
                "page_number": record["page"],
                "quality_flags": ["human_review_required", "rebuilt_v2"],
                "metadata_json": {
                    "generation_method": record["generation_method"],
                    "label": record["label"],
                    "line_number": record["line_number"],
                    "ocr_page_path": record["ocr_page_path"],
                    "perturbation_type": record["perturbation_type"],
                    "original_proposition": record["original_proposition"],
                    "dataset_dir": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
                },
                "reviewer": "",
                "review_comment": "",
            }
        )
    return insert_items(db, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild textbook judgment items with balanced true/false labels.")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--true-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--context-before", type=int, default=20)
    parser.add_argument("--context-after", type=int, default=20)
    parser.add_argument(
        "--allow-semantic-perturbations",
        action="store_true",
        help="Also allow strict object replacement and polarity flip for false judgments. Default keeps number-only false items.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-quarantine", action="store_true")
    args = parser.parse_args()

    load_env()
    base_candidates = build_base_candidates(args.context_before, args.context_after)
    judgments = build_judgments(
        base_candidates,
        args.target,
        args.true_ratio,
        args.seed,
        allow_semantic_perturbations=args.allow_semantic_perturbations,
    )
    records = [to_record(candidate) for candidate in judgments]

    label_counts: dict[str, int] = {}
    for record in records:
        label_counts[record["label"]] = label_counts.get(record["label"], 0) + 1

    if args.dry_run:
        print(
            json.dumps(
                {
                    "base_candidates": len(base_candidates),
                    "generated": len(records),
                    "by_label": dict(sorted(label_counts.items())),
                    "preview": records[:10],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    write_outputs(records)

    with SessionLocal() as db:
        quarantined = 0 if args.no_quarantine else quarantine_old_items(db)
        inserted = import_records(db, records)
        db.flush()
        status_rows = db.execute(
            select(CorpusItem.review_status, func.count())
            .where(CorpusItem.task_type == TASK_TYPE)
            .group_by(CorpusItem.review_status)
            .order_by(CorpusItem.review_status)
        ).all()
        db.commit()

    print(
        json.dumps(
            {
                "base_candidates": len(base_candidates),
                "generated": len(records),
                "inserted": inserted,
                "quarantined_old_items": quarantined,
                "by_label": dict(sorted(label_counts.items())),
                "db_status_counts_before_commit": {status: count for status, count in status_rows},
                "output_dir": str(OUTPUT_DIR.relative_to(REPO_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
