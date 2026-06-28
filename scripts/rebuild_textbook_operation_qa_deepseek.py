"""Rewrite pending textbook_operation_qa records with DeepSeek and OCR context."""

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


OUTPUT_DIR = REPO_ROOT / "data" / "textbook_operation_qa_deepseek_v1"
GENERATION_METHOD = "deepseek_textbook_operation_v1"
FIGURE_RE = re.compile(r"(如图|如下图|见图|图\s*\d|如表|见表|表\s*\d|所示)")
TABLE_RE = re.compile(r"(<table|</table>|<tr>|<td>|\\beginaligned|\\frac)", re.I)
LINE_RE = re.compile(r"^\s*L(?P<line>\d+)\s*\|\s*第\s*(?P<page>\d+)\s*页\s*\|\s*(?P<text>.*)$")


SYSTEM_PROMPT = """你是铁路教育语料审核与抽取助手。你的任务是基于给定 OCR 原文，重写“教材操作/运行检修类问答”。

必须遵守：
1. 主要依据 OCR 原文生成问题、答案和证据；允许补充铁路牵引供电领域内与 OCR 语境直接相关的常识性连接语，但不得引入不相关的新对象、新流程或新标准。
2. 问题必须明确指出操作对象、设备对象或业务场景，不能使用“其、它、该设备、这种情况、相关内容”等模糊主语。
3. 如果原问题主语不清楚，请根据 OCR 上下文补全主语，例如明确为“接触网巡视”“接触网设备质量鉴定”“高速铁路维修计划”“腕臂支持结构维护”等。
4. 答案必须完整，不能只截取一句残缺内容；可以基于同页上下文归纳出完整操作要求、维护要求、检查要求或处理流程。
5. 答案必须与 OCR 原文直接相关。允许压缩、合并、顺序整理，但不得把 OCR 原文没有支持的内容写成事实。
6. 必须删除或避开“如图所示、见图、如下图、见表、如表所示”等图表依赖表达。
7. 如果 OCR 原文主要是图表残片、目录、标题、半句、编号列表残缺、公式残片，或者无法支撑完整问答，则返回 usable=false。
8. 输出必须是严格 JSON，不要输出 Markdown，不要解释。

输出 JSON 格式：
{
  "usable": true 或 false,
  "question": "重写后的问题",
  "answer": "基于 OCR 原文并允许少量相关领域连接语生成的完整答案",
  "evidence": "最能支撑答案的 OCR 原文摘录，必须来自输入 OCR 原文",
  "reason": "如果 usable=false，说明原因；如果 usable=true，简述为何证据充分"
}

字段要求：
- question 必须是一个中文问句，以“？”结尾。
- answer 必须是中文完整表述，建议 60-360 字。
- evidence 必须是 OCR 原文中的连续或近邻原文摘录，不得凭空改写。
- 如果 usable=false，question、answer、evidence 使用空字符串。
"""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u3000", " ")).strip()


def parse_ocr_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        match = LINE_RE.match(raw)
        if not match:
            continue
        rows.append(
            {
                "line": int(match.group("line")),
                "page": int(match.group("page")),
                "text": normalize(match.group("text")),
                "raw": raw,
            }
        )
    return rows


def get_field(item: CorpusItem | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def load_full_ocr_text(item: CorpusItem | dict[str, Any]) -> str:
    path = REPO_ROOT / (get_field(item, "source_path") or "")
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="ignore")
    return get_field(item, "source_text") or ""


def build_context(item: CorpusItem | dict[str, Any]) -> str:
    full_text = load_full_ocr_text(item)
    rows = parse_ocr_lines(full_text)
    if not rows:
        return get_field(item, "source_text") or get_field(item, "evidence") or get_field(item, "answer") or ""

    metadata = get_field(item, "metadata_json", {}) or {}
    target_line = metadata.get("line_number")
    try:
        target_line = int(target_line)
    except (TypeError, ValueError):
        target_line = None
    target_page = get_field(item, "page_number")

    page_rows = [row for row in rows if row["page"] == target_page] if target_page else []
    if not page_rows:
        page_rows = rows

    target_index = next((idx for idx, row in enumerate(page_rows) if row["line"] == target_line), -1)
    if target_index < 0:
        target_index = 0
    selected = page_rows[max(0, target_index - 10) :]
    context = "\n".join(row["raw"] for row in selected)
    if len(context) > 30000:
        context = context[:30000]
    return context


def make_user_prompt(item: CorpusItem | dict[str, Any], ocr_context: str) -> str:
    metadata = get_field(item, "metadata_json", {}) or {}
    return f"""原问题：
{get_field(item, "question")}

原答案：
{get_field(item, "answer")}

来源信息：
- 文档：{get_field(item, "source_document")}
- 页码：{get_field(item, "page_number") or ""}
- 命中 OCR 行：{metadata.get("line_number", "")}

OCR 原文：
{ocr_context}
"""


def deepseek_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


def extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def call_deepseek(client: anthropic.Anthropic, item: CorpusItem | dict[str, Any], ocr_context: str, retries: int = 3) -> dict[str, Any]:
    model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or "deepseek-v4-pro[1m]"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0,
                top_p=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": make_user_prompt(item, ocr_context)}],
            )
            return parse_json_response(extract_text(response))
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed: {last_error}")


def validate_result(result: dict[str, Any], ocr_context: str) -> tuple[bool, str]:
    if not result.get("usable"):
        return False, str(result.get("reason") or "unusable")
    question = normalize(result.get("question", ""))
    answer = normalize(result.get("answer", ""))
    evidence = normalize(result.get("evidence", ""))
    if not question.endswith("？"):
        return False, "question_not_ended_with_question_mark"
    if re.search(r"(其|它|该设备|这种情况|相关内容)", question):
        return False, "ambiguous_question_subject"
    if not (30 <= len(answer) <= 520):
        return False, "answer_length_out_of_range"
    if not evidence or len(evidence) < 15:
        return False, "missing_evidence"
    if FIGURE_RE.search(question) or FIGURE_RE.search(answer):
        return False, "figure_reference_in_result"
    if TABLE_RE.search(answer):
        return False, "table_markup_in_answer"
    if not evidence_supported(evidence, ocr_context):
        return False, "evidence_not_from_ocr_context"
    return True, ""


def compact_for_match(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text or "")


def han_bigrams(text: str) -> set[str]:
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return {han[index : index + 2] for index in range(max(0, len(han) - 1))}


def evidence_supported(evidence: str, ocr_context: str) -> bool:
    compact_evidence = compact_for_match(evidence)
    compact_context = compact_for_match(ocr_context)
    if not compact_evidence or not compact_context:
        return False
    if compact_evidence[:80] in compact_context:
        return True
    evidence_pairs = han_bigrams(evidence[:220])
    context_pairs = han_bigrams(ocr_context)
    if not evidence_pairs:
        return False
    return len(evidence_pairs & context_pairs) / len(evidence_pairs) >= 0.72


def update_item(item: CorpusItem, result: dict[str, Any], ocr_context: str) -> None:
    metadata = dict(item.metadata_json or {})
    metadata.update(
        {
            "generation_method": GENERATION_METHOD,
            "previous_generation_method": (item.metadata_json or {}).get("generation_method"),
            "deepseek_reason": result.get("reason", ""),
            "ocr_context_policy": "from_10_lines_before_hit_to_page_end",
        }
    )
    item.question = normalize(result["question"])
    item.answer = normalize(result["answer"])
    item.evidence = normalize(result["evidence"])
    item.original_question = item.question
    item.original_answer = item.answer
    item.source_text = ocr_context
    item.quality_flags = ["human_review_required", "deepseek_rebuilt", "context_resolved_subject"]
    item.metadata_json = metadata
    item.review_comment = ""
    item.reviewer = ""
    item.review_status = "pending"
    item.version += 1


def write_audit(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "status", "question", "answer", "evidence", "reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def item_snapshot(item: CorpusItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "question": item.question,
        "answer": item.answer,
        "evidence": item.evidence,
        "source_text": item.source_text,
        "source_document": item.source_document,
        "source_path": item.source_path,
        "page_number": item.page_number,
        "metadata_json": item.metadata_json or {},
    }


def process_one(snapshot: dict[str, Any]) -> dict[str, Any]:
    client = deepseek_client()
    ocr_context = build_context(snapshot)
    try:
        result = call_deepseek(client, snapshot, ocr_context)
        ok, reason = validate_result(result, ocr_context)
    except Exception as error:  # noqa: BLE001
        result = {"usable": False, "question": "", "answer": "", "evidence": "", "reason": str(error)}
        ok, reason = False, "request_or_parse_failed"
    return {
        "id": snapshot["id"],
        "old_question": snapshot["question"],
        "ok": ok,
        "reason": reason,
        "result": result,
        "ocr_context": ocr_context,
    }


def process(limit: int | None, dry_run: bool, workers: int) -> dict[str, int]:
    audit_rows: list[dict[str, Any]] = []
    stats = {"seen": 0, "updated": 0, "deleted": 0, "failed": 0}
    with SessionLocal() as db:
        query = (
            select(CorpusItem)
            .where(CorpusItem.task_type == "textbook_operation_qa", CorpusItem.review_status == "pending")
            .order_by(CorpusItem.id)
        )
        items = list(db.scalars(query))
        if limit:
            items = items[:limit]
        snapshots = [item_snapshot(item) for item in items]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(process_one, snapshot) for snapshot in snapshots]
            for future in as_completed(futures):
                payload = future.result()
                item = db.get(CorpusItem, payload["id"])
                if item is None:
                    continue
                result = payload["result"]
                ok = payload["ok"]
                reason = payload["reason"]
                ocr_context = payload["ocr_context"]
                stats["seen"] += 1

                if ok:
                    audit_rows.append(
                        {
                            "id": item.id,
                            "status": "updated",
                            "question": normalize(result["question"]),
                            "answer": normalize(result["answer"]),
                            "evidence": normalize(result["evidence"]),
                            "reason": normalize(result.get("reason", "")),
                        }
                    )
                    if not dry_run:
                        update_item(item, result, ocr_context)
                        db.commit()
                    stats["updated"] += 1
                else:
                    audit_rows.append(
                        {
                            "id": item.id,
                            "status": "deleted" if reason != "request_or_parse_failed" else "failed",
                            "question": item.question,
                            "answer": "",
                            "evidence": "",
                            "reason": normalize(result.get("reason", "")) or reason,
                        }
                    )
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
                print(json.dumps({"processed": stats["seen"], "id": item.id, "ok": ok, "reason": reason}, ensure_ascii=False), flush=True)
    write_audit(audit_rows)
    summary = {**stats, "dry_run": int(dry_run)}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite textbook_operation_qa pending records with DeepSeek.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(process(args.limit, args.dry_run, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
