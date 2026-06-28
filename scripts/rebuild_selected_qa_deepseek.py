"""Rewrite selected QA task records with DeepSeek and source-grounded context."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic
from sqlalchemy import delete, select


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "annotation_system" / "backend"))

from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem, ReviewEvent  # noqa: E402


DEFAULT_TASKS = [
    "regulation_clause_qa",
    "regulation_definition_qa",
    "regulation_extractive_qa",
    "regulation_inspection_qa",
    "regulation_prohibition_qa",
    "regulation_requirement_qa",
    "regulation_responsibility_qa",
    "regulation_standard_qa",
    "textbook_definition_qa",
    "textbook_extractive_qa",
]
OUTPUT_DIR = REPO_ROOT / "data" / "selected_qa_deepseek_v1"
GENERATION_METHOD = "deepseek_selected_qa_v1"
FIGURE_RE = re.compile(r"(如图|如下图|见图|图\s*\d|如表|见表|表\s*\d|所示)")
TABLE_RE = re.compile(r"(<table|</table>|<tr>|<td>|\\beginaligned|\\frac)", re.I)
LINE_RE = re.compile(r"^\s*L(?P<line>\d+)\s*\|\s*第\s*(?P<page>\d+)\s*页\s*\|\s*(?P<text>.*)$")

TASK_GUIDANCE = {
    "regulation_clause_qa": "规章条款问答：问题要明确条款对象、责任主体或适用场景；答案要完整表达条款要求。",
    "regulation_definition_qa": "规章定义问答：问题要明确被定义对象；答案要说明定义、范围或组成，不能只问“主要包括哪些内容”。",
    "regulation_extractive_qa": "规章抽取问答：问题要明确从规章中抽取的对象和条件；答案要尽量贴近原文。",
    "regulation_inspection_qa": "规章检查检测维护问答：问题要明确检查对象、检测项目或维护场景；答案要整理检查周期、项目、处理要求或维护要求。",
    "regulation_prohibition_qa": "规章禁止性问答：问题要明确禁止对象或限制条件；答案要突出不得、严禁、禁止、限值等要求。",
    "regulation_requirement_qa": "规章要求问答：问题要明确设备或业务对象；答案要完整表达技术要求、管理要求或条件。",
    "regulation_responsibility_qa": "规章职责问答：问题要明确责任主体；答案要列出职责、负责范围或组织实施要求。",
    "regulation_standard_qa": "规章标准问答：问题要明确标准对象和指标；答案要保留限值、周期、允许偏差等关键数值。",
    "textbook_definition_qa": "教材定义问答：问题要明确概念对象；答案可在原文基础上做领域内相关解释，但不得脱离 OCR 上下文。",
    "textbook_extractive_qa": "教材抽取问答：问题要明确教材中的知识对象、设备对象或场景；答案要基于 OCR 原文进行抽取和适度归纳。",
}


SYSTEM_PROMPT = """你是铁路教育语料审核与抽取助手。你的任务是基于给定来源原文，重写铁路牵引供电/接触网/变配电相关问答。

必须遵守：
1. 主要依据来源原文生成问题、答案和证据；允许补充与来源语境直接相关的铁路专业常识性连接语，但不得引入不相关对象、流程、标准或数值。
2. 问题必须明确指出对象、设备、主体、条件或业务场景，不能使用“其、它、该设备、这种情况、相关内容、主要包括哪些内容”等模糊提问。
3. 如果原问题主语不清楚，请根据来源上下文补全主语，例如明确为“接触网零件”“供电段”“轨面标准线”“接触网巡视”“牵引变电所工程交接验收”等。
4. 答案必须完整，不能只截取一句残缺内容；可以基于相邻上下文归纳出完整要求、定义、职责、检查项目、禁止条件或标准指标。
5. 答案必须与来源原文直接相关。允许压缩、合并、顺序整理，但不得把来源原文没有支撑的内容写成事实。
6. 必须删除或避开“如图所示、见图、如下图、见表、如表所示”等图表依赖表达。
7. 如果来源原文主要是图表残片、目录、标题、半句、编号列表残缺、公式残片，或者无法支撑该任务类型的完整问答，则返回 usable=false。
8. 输出必须是严格 JSON，不要输出 Markdown，不要解释。

输出 JSON：
{
  "usable": true 或 false,
  "question": "重写后的问题",
  "answer": "基于来源原文并允许少量相关专业连接语生成的完整答案",
  "evidence": "最能支撑答案的来源原文摘录，必须来自输入来源原文",
  "reason": "如果 usable=false，说明原因；如果 usable=true，简述证据为何充分"
}

字段要求：
- question 必须是一个中文问句，以“？”结尾。
- answer 必须是中文完整表述，建议 40-420 字。
- evidence 必须来自输入来源原文，不得凭空改写。
- 如果 usable=false，question、answer、evidence 使用空字符串。
"""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u3000", " ")).strip()


def parse_ocr_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        match = LINE_RE.match(raw)
        if match:
            rows.append(
                {
                    "line": int(match.group("line")),
                    "page": int(match.group("page")),
                    "text": normalize(match.group("text")),
                    "raw": raw,
                }
            )
    return rows


def full_source_text(snapshot: dict[str, Any]) -> str:
    path = REPO_ROOT / (snapshot.get("source_path") or "")
    if path.is_file() and path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    return snapshot.get("source_text") or snapshot.get("evidence") or snapshot.get("answer") or ""


def source_context(snapshot: dict[str, Any]) -> str:
    text = full_source_text(snapshot)
    rows = parse_ocr_lines(text)
    if not rows:
        context = text
    else:
        metadata = snapshot.get("metadata_json") or {}
        try:
            target_line = int(metadata.get("line_number"))
        except (TypeError, ValueError):
            target_line = None
        target_page = snapshot.get("page_number")
        page_rows = [row for row in rows if row["page"] == target_page] if target_page else []
        if not page_rows:
            page_rows = rows
        target_index = next((idx for idx, row in enumerate(page_rows) if row["line"] == target_line), -1)
        if target_index < 0:
            target_index = 0
        context = "\n".join(row["raw"] for row in page_rows[max(0, target_index - 10) :])
    return context[:30000]


def client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY")
    return anthropic.Anthropic(api_key=api_key, base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"))


def make_prompt(snapshot: dict[str, Any], context: str) -> str:
    task = snapshot["task_type"]
    return f"""任务类型：
{task}

任务要求：
{TASK_GUIDANCE.get(task, "铁路教育问答：问题要明确，答案要有来源证据支撑。")}

原问题：
{snapshot.get("question", "")}

原答案：
{snapshot.get("answer", "")}

来源信息：
- 文档：{snapshot.get("source_document", "")}
- 页码：{snapshot.get("page_number") or ""}
- 命中 OCR 行：{(snapshot.get("metadata_json") or {}).get("line_number", "")}

来源原文：
{context}
"""


def extract_text(response: Any) -> str:
    return "\n".join(getattr(block, "text", "") for block in response.content if getattr(block, "text", "")).strip()


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def call_model(snapshot: dict[str, Any], context: str, retries: int = 3) -> dict[str, Any]:
    model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or "deepseek-v4-pro[1m]"
    deepseek = client()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = deepseek.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0,
                top_p=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": make_prompt(snapshot, context)}],
            )
            return parse_json(extract_text(response))
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed: {last_error}")


def compact(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text or "")


def bigrams(text: str) -> set[str]:
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return {han[index : index + 2] for index in range(max(0, len(han) - 1))}


def evidence_supported(evidence: str, context: str) -> bool:
    compact_evidence = compact(evidence)
    compact_context = compact(context)
    if not compact_evidence or not compact_context:
        return False
    if compact_evidence[:80] in compact_context:
        return True
    evidence_pairs = bigrams(evidence[:240])
    context_pairs = bigrams(context)
    return bool(evidence_pairs) and len(evidence_pairs & context_pairs) / len(evidence_pairs) >= 0.72


def validate(result: dict[str, Any], context: str) -> tuple[bool, str]:
    if not result.get("usable"):
        return False, normalize(result.get("reason", "")) or "unusable"
    question = normalize(result.get("question", ""))
    answer = normalize(result.get("answer", ""))
    evidence = normalize(result.get("evidence", ""))
    if not question.endswith("？"):
        return False, "question_not_ended_with_question_mark"
    if re.search(r"(其|它|该设备|这种情况|相关内容|主要包括哪些内容)", question):
        return False, "ambiguous_question_subject"
    if not (20 <= len(answer) <= 650):
        return False, "answer_length_out_of_range"
    if FIGURE_RE.search(question) or FIGURE_RE.search(answer):
        return False, "figure_reference_in_result"
    if TABLE_RE.search(answer):
        return False, "table_markup_in_answer"
    if len(evidence) < 12:
        return False, "missing_evidence"
    if not evidence_supported(evidence, context):
        return False, "evidence_not_from_source_context"
    return True, ""


def snapshot(item: CorpusItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "task_type": item.task_type,
        "question": item.question,
        "answer": item.answer,
        "evidence": item.evidence,
        "source_text": item.source_text,
        "source_document": item.source_document,
        "source_path": item.source_path,
        "page_number": item.page_number,
        "metadata_json": item.metadata_json or {},
    }


def process_one(data: dict[str, Any]) -> dict[str, Any]:
    context = source_context(data)
    try:
        result = call_model(data, context)
        ok, reason = validate(result, context)
    except Exception as error:  # noqa: BLE001
        result = {"usable": False, "question": "", "answer": "", "evidence": "", "reason": str(error)}
        ok, reason = False, "request_or_parse_failed"
    return {"id": data["id"], "ok": ok, "reason": reason, "result": result, "context": context, "old_question": data["question"]}


def update_item(item: CorpusItem, result: dict[str, Any], context: str) -> None:
    metadata = dict(item.metadata_json or {})
    metadata.update(
        {
            "generation_method": GENERATION_METHOD,
            "previous_generation_method": (item.metadata_json or {}).get("generation_method"),
            "deepseek_reason": result.get("reason", ""),
            "deepseek_task_rewrite": True,
        }
    )
    item.question = normalize(result["question"])
    item.answer = normalize(result["answer"])
    item.evidence = normalize(result["evidence"])
    item.source_text = context
    item.original_question = item.question
    item.original_answer = item.answer
    item.quality_flags = ["human_review_required", "deepseek_rebuilt", "context_resolved_subject"]
    item.metadata_json = metadata
    item.review_status = "pending"
    item.reviewer = ""
    item.review_comment = ""
    item.version += 1


def write_audit(rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "task_type", "status", "question", "answer", "evidence", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_audit(skip_ids: set[int]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"seen": 0, "updated": 0, "deleted": 0, "failed": 0}
    path = OUTPUT_DIR / "results.jsonl"
    if not path.exists():
        return rows, counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("id") or 0) in skip_ids:
                continue
            rows.append(row)
            counts["seen"] += 1
            status = row.get("status")
            if status in ("updated", "deleted", "failed"):
                counts[status] += 1
    return rows, counts


def run(tasks: list[str], workers: int, limit: int | None, dry_run: bool, skip_existing: bool) -> dict[str, Any]:
    stats: dict[str, Any] = {"seen": 0, "updated": 0, "deleted": 0, "failed": 0, "dry_run": int(dry_run), "tasks": tasks}
    with SessionLocal() as db:
        query = (
            select(CorpusItem)
            .where(CorpusItem.task_type.in_(tasks), CorpusItem.review_status.in_(["pending", "rejected", "needs_revision"]))
            .order_by(CorpusItem.task_type, CorpusItem.id)
        )
        items = list(db.scalars(query))
        if skip_existing:
            items = [item for item in items if (item.metadata_json or {}).get("generation_method") != GENERATION_METHOD]
        if limit:
            items = items[:limit]
        snapshots = [snapshot(item) for item in items]
        audit_rows, existing_counts = load_existing_audit({data["id"] for data in snapshots})
        stats.update(existing_counts)
        stats["skip_existing"] = int(skip_existing)
        stats["remaining_this_run"] = len(snapshots)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(process_one, data) for data in snapshots]
            for future in as_completed(futures):
                payload = future.result()
                item = db.get(CorpusItem, payload["id"])
                if item is None:
                    continue
                stats["seen"] += 1
                result = payload["result"]
                ok = payload["ok"]
                reason = payload["reason"]
                if ok:
                    row = {
                        "id": item.id,
                        "task_type": item.task_type,
                        "status": "updated",
                        "question": normalize(result["question"]),
                        "answer": normalize(result["answer"]),
                        "evidence": normalize(result["evidence"]),
                        "reason": normalize(result.get("reason", "")),
                    }
                    if not dry_run:
                        update_item(item, result, payload["context"])
                        db.commit()
                    stats["updated"] += 1
                else:
                    row = {
                        "id": item.id,
                        "task_type": item.task_type,
                        "status": "failed" if reason == "request_or_parse_failed" else "deleted",
                        "question": item.question,
                        "answer": "",
                        "evidence": "",
                        "reason": normalize(result.get("reason", "")) or reason,
                    }
                    if reason == "request_or_parse_failed":
                        stats["failed"] += 1
                        if not dry_run:
                            db.rollback()
                    else:
                        stats["deleted"] += 1
                        if not dry_run:
                            db.execute(delete(ReviewEvent).where(ReviewEvent.item_id == item.id))
                            db.delete(item)
                            db.commit()
                audit_rows.append(row)
                print(json.dumps({"processed": stats["seen"], "id": item.id, "task": item.task_type, "ok": ok, "reason": reason}, ensure_ascii=False), flush=True)
                if stats["seen"] % 25 == 0:
                    write_audit(audit_rows, stats)
    write_audit(audit_rows, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite selected QA tasks with DeepSeek.")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true", help="Process records already rebuilt by this script.")
    args = parser.parse_args()
    print(json.dumps(run(args.tasks, args.workers, args.limit, args.dry_run, not args.no_skip_existing), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
