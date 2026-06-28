from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_system.backend.app.database import engine


OUTPUT_DIR = Path("data/translation_en_v1")
TERM_TASKS = {"terminology_pair", "terminology_translation", "terminology_explanation"}
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class CorpusRow:
    id: int
    task_type: str
    question: str
    answer: str
    evidence: str
    source_text: str
    metadata_json: dict[str, Any]


def has_chinese(text_value: str) -> bool:
    return bool(CHINESE_RE.search(text_value or ""))


def clean_text(text_value: str) -> str:
    return re.sub(r"\s+", " ", (text_value or "").strip())


def load_term_pairs() -> list[tuple[str, str]]:
    sql = text(
        """
        SELECT question, answer
        FROM corpus_items
        WHERE task_type = 'terminology_pair'
          AND coalesce(question, '') <> ''
          AND coalesce(answer, '') <> ''
        """
    )
    pairs: dict[str, str] = {}
    with engine.connect() as conn:
        for zh, en in conn.execute(sql):
            zh_clean = clean_text(zh)
            en_clean = clean_text(en)
            if not zh_clean or not en_clean:
                continue
            if not has_chinese(zh_clean):
                continue
            pairs.setdefault(zh_clean, en_clean)
    return sorted(pairs.items(), key=lambda item: len(item[0]), reverse=True)


def matched_terms(row: CorpusRow, term_pairs: list[tuple[str, str]], limit: int = 80) -> list[dict[str, str]]:
    haystack = "\n".join([row.question, row.answer, row.evidence, row.source_text])
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for zh, en in term_pairs:
        if zh in seen:
            continue
        if zh in haystack:
            matches.append({"zh": zh, "en": en})
            seen.add(zh)
        if len(matches) >= limit:
            break
    return matches


def deterministic_terminology_translation(row: CorpusRow) -> tuple[str, str] | None:
    metadata = row.metadata_json or {}
    term_zh = clean_text(str(metadata.get("term_zh") or ""))
    term_en = clean_text(str(metadata.get("term_en") or ""))

    if row.task_type == "terminology_pair":
        term_zh = term_zh or clean_text(row.question)
        term_en = term_en or clean_text(row.answer)
        if term_zh and term_en:
            return f"What is the English railway term for {term_zh}?", term_en

    if row.task_type == "terminology_translation":
        term_zh = term_zh or clean_text(row.answer)
        if not term_en:
            question = clean_text(row.question)
            term_en = question.split("对应的中文铁路术语是什么？", 1)[0].strip()
        if term_zh and term_en:
            return f"What is the Chinese railway term for {term_en}?", f"The Chinese railway term is {term_zh}."

    if row.task_type == "terminology_explanation":
        term_zh = term_zh or clean_text(row.question).removesuffix("是什么意思？")
        if not term_en:
            match = re.search(r"英文铁路术语为\s+(.+?)[。.]?$", clean_text(row.answer))
            if match:
                term_en = match.group(1).strip()
        if term_zh and term_en:
            return f"What does the railway term {term_zh} mean?", (
                f"The Chinese railway term {term_zh} corresponds to the English railway term {term_en}."
            )
    return None


def anthropic_messages_url() -> str:
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
    if base_url.endswith("/v1/messages"):
        return base_url
    if base_url.endswith("/messages"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


def call_deepseek(prompt: str, max_retries: int = 5) -> str:
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY")

    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
        "max_tokens": int(os.getenv("TRANSLATION_MAX_TOKENS", "4000")),
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        anthropic_messages_url(),
        data=data,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("content") or []
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            wait_seconds = min(60, 2**attempt)
            time.sleep(wait_seconds)
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")


def parse_json_object(raw_text: str) -> dict[str, str]:
    text_value = raw_text.strip()
    if not text_value:
        raise ValueError("empty model response")
    if text_value.startswith("```"):
        text_value = re.sub(r"^```(?:json)?\s*", "", text_value)
        text_value = re.sub(r"\s*```$", "", text_value)
    start = text_value.find("{")
    end = text_value.rfind("}")
    if start >= 0 and end > start:
        text_value = text_value[start : end + 1]
    try:
        parsed = json.loads(text_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {text_value[:500]}") from exc
    return {
        "question_en": clean_text(str(parsed.get("question_en") or "")),
        "answer_en": clean_text(str(parsed.get("answer_en") or "")),
    }


def build_prompt(row: CorpusRow, terms: list[dict[str, str]]) -> str:
    term_lines = "\n".join(f"- {term['zh']} => {term['en']}" for term in terms) or "- None"
    source_context = clean_text(row.evidence or row.source_text)
    if len(source_context) > 5000:
        source_context = source_context[:5000]
    return f"""You are translating a Chinese railway education QA dataset into English.

Task:
Translate the Chinese question and answer into accurate English. Return only one JSON object.

Hard requirements:
1. Preserve all technical meaning, numbers, units, equipment names, standards, prohibitions, and conditions.
2. Use the provided Chinese-English railway terminology pairs exactly when the Chinese term appears and the English term is natural in the sentence.
3. Do not add unsupported facts. You may only make the English phrasing natural and complete.
4. Do not output Markdown. Do not include explanations.
5. Keep the answer as an answer, not as commentary about translation.

Terminology pairs:
{term_lines}

Source context:
{source_context}

Chinese question:
{row.question}

Chinese answer:
{row.answer}

Return JSON schema:
{{"question_en":"...","answer_en":"..."}}
"""


def validate_translation(row: CorpusRow, question_en: str, answer_en: str, terms: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    if row.question.strip() and not question_en:
        warnings.append("missing_question_en")
    if row.answer.strip() and not answer_en:
        warnings.append("missing_answer_en")
    combined_en = f"{question_en}\n{answer_en}"
    for term in terms[:30]:
        source_has_term = term["zh"] in f"{row.question}\n{row.answer}"
        if source_has_term and term["en"] and term["en"].lower() not in combined_en.lower():
            warnings.append(f"term_missing:{term['zh']}=>{term['en']}")
    return warnings[:20]


def fetch_missing_rows(limit: int | None, tasks: set[str] | None) -> list[CorpusRow]:
    clauses = [
        "(coalesce(question, '') <> '' OR coalesce(answer, '') <> '')",
        "((coalesce(question, '') <> '' AND coalesce(question_en, '') = '') OR (coalesce(answer, '') <> '' AND coalesce(answer_en, '') = ''))",
    ]
    params: dict[str, Any] = {}
    if tasks:
        clauses.append("task_type = ANY(:tasks)")
        params["tasks"] = list(tasks)
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT :limit"
        params["limit"] = limit
    sql = text(
        f"""
        SELECT id, task_type, question, answer, evidence, source_text, metadata_json
        FROM corpus_items
        WHERE {' AND '.join(clauses)}
        ORDER BY
          CASE WHEN task_type IN ('terminology_pair','terminology_translation','terminology_explanation') THEN 0 ELSE 1 END,
          id
        {limit_sql}
        """
    )
    with engine.connect() as conn:
        return [
            CorpusRow(
                id=row.id,
                task_type=row.task_type,
                question=row.question or "",
                answer=row.answer or "",
                evidence=row.evidence or "",
                source_text=row.source_text or "",
                metadata_json=row.metadata_json or {},
            )
            for row in conn.execute(sql, params).mappings()
        ]


def update_row(row: CorpusRow, question_en: str, answer_en: str, metadata: dict[str, Any]) -> None:
    sql = text(
        """
        UPDATE corpus_items
        SET question_en = :question_en,
            answer_en = :answer_en,
            metadata_json = CAST(:metadata_json AS JSONB),
            updated_at = now()
        WHERE id = :id
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "id": row.id,
                "question_en": question_en,
                "answer_en": answer_en,
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
            },
        )


def translate_row(row: CorpusRow, term_pairs: list[tuple[str, str]]) -> tuple[int, str, str, dict[str, Any]]:
    terms = matched_terms(row, term_pairs)
    metadata = dict(row.metadata_json or {})
    translation_meta = {
        "method": "deepseek_translation_en_v1",
        "model": os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
        "matched_terms": terms[:30],
    }

    if row.task_type in TERM_TASKS:
        deterministic = deterministic_terminology_translation(row)
        if deterministic:
            question_en, answer_en = deterministic
            warnings = validate_translation(row, question_en, answer_en, terms)
            translation_meta["method"] = "terminology_rule_translation_en_v1"
            translation_meta["warnings"] = warnings
            metadata["translation_en_v1"] = translation_meta
            return row.id, question_en, answer_en, metadata

    prompt = build_prompt(row, terms)
    last_parse_error: Exception | None = None
    for _ in range(3):
        raw = call_deepseek(prompt)
        try:
            parsed = parse_json_object(raw)
            break
        except ValueError as exc:
            last_parse_error = exc
            time.sleep(2)
    else:
        raise RuntimeError(f"Unable to parse model JSON after retries: {last_parse_error}")
    question_en = parsed["question_en"]
    answer_en = parsed["answer_en"]
    warnings = validate_translation(row, question_en, answer_en, terms)
    translation_meta["warnings"] = warnings
    metadata["translation_en_v1"] = translation_meta
    return row.id, question_en, answer_en, metadata


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_tasks(raw_tasks: str) -> set[str] | None:
    if not raw_tasks:
        return None
    tasks = {item.strip() for item in raw_tasks.split(",") if item.strip()}
    return tasks or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.jsonl"
    summary_path = OUTPUT_DIR / "summary.json"

    term_pairs = load_term_pairs()
    rows = fetch_missing_rows(limit=args.limit or None, tasks=parse_tasks(args.tasks))
    total = len(rows)
    done = 0
    failed = 0
    started = time.time()

    deterministic_rows = [row for row in rows if row.task_type in TERM_TASKS]
    api_rows = [row for row in rows if row.task_type not in TERM_TASKS]
    print(
        json.dumps(
            {
                "rows": total,
                "deterministic": len(deterministic_rows),
                "api": len(api_rows),
                "term_pairs": len(term_pairs),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for row in deterministic_rows:
        try:
            row_id, question_en, answer_en, metadata = translate_row(row, term_pairs)
            if not args.dry_run:
                update_row(row, question_en, answer_en, metadata)
            append_jsonl(
                results_path,
                {
                    "id": row_id,
                    "task_type": row.task_type,
                    "question_en": question_en,
                    "answer_en": answer_en,
                    "method": metadata.get("translation_en_v1", {}).get("method"),
                    "warnings": metadata.get("translation_en_v1", {}).get("warnings", []),
                },
            )
            done += 1
        except Exception as exc:
            failed += 1
            append_jsonl(results_path, {"id": row.id, "task_type": row.task_type, "error": str(exc)})
        if done % 1000 == 0 and done:
            print(json.dumps({"done": done, "failed": failed, "elapsed_sec": round(time.time() - started, 1)}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(translate_row, row, term_pairs): row for row in api_rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            try:
                row_id, question_en, answer_en, metadata = future.result()
                if not args.dry_run:
                    update_row(row, question_en, answer_en, metadata)
                append_jsonl(
                    results_path,
                    {
                        "id": row_id,
                        "task_type": row.task_type,
                        "question_en": question_en,
                        "answer_en": answer_en,
                        "method": metadata.get("translation_en_v1", {}).get("method"),
                        "warnings": metadata.get("translation_en_v1", {}).get("warnings", []),
                    },
                )
                done += 1
            except Exception as exc:
                failed += 1
                append_jsonl(results_path, {"id": row.id, "task_type": row.task_type, "error": str(exc)})
            if done % 50 == 0 or done + failed == total:
                print(
                    json.dumps(
                        {
                            "done": done,
                            "failed": failed,
                            "total": total,
                            "elapsed_sec": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {"done": done, "failed": failed, "total": total, "elapsed_sec": round(time.time() - started, 1)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
