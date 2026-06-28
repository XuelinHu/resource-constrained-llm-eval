from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

from sqlalchemy import select

from context_evidence import clean_semantic_text, context_window_from_file


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = REPO_ROOT / "annotation_system" / "backend"
sys.path.insert(0, str(BACKEND_APP))

from app.config import load_env  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem, ReviewEvent  # noqa: E402


DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_CODEX_MODEL = os.getenv("CODEX_MODEL", "codex")
DEFAULT_CODEX_COMMAND = (
    "codex -s read-only -a never exec --ephemeral --color never -"
    if shutil.which("codex")
    else ""
)
DEFAULT_EXCLUDED_TASK_TYPES = {
    "terminology_pair",
    "terminology_translation",
    "textbook_source",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else REPO_ROOT / raw


def markdown_path_for_item(item: CorpusItem) -> Path | None:
    metadata = item.metadata_json or {}
    markdown_path = metadata.get("ocr_page_path") or metadata.get("markdown")
    if not markdown_path:
        return None
    path = repo_path(str(markdown_path))
    return path if path.is_file() else None


def book_context_path_for_page(markdown_path: Path) -> Path | None:
    normalized = markdown_path.as_posix()
    if "/pages/" not in normalized:
        return None
    book_dir = markdown_path.parent.parent.name
    context_path = REPO_ROOT / "data" / "ocr" / "railway_context" / book_dir / "book_context.md"
    return context_path if context_path.is_file() else None


def source_context_for_item(
    item: CorpusItem,
    *,
    before: int,
    after: int,
    include_adjacent_pages: bool,
    adjacent_paragraphs: int,
    min_before_chars: int,
    min_after_chars: int,
) -> tuple[str, str]:
    question = clean_semantic_text(item.question or "")
    answer = clean_semantic_text(item.answer or "")
    evidence = clean_semantic_text(item.evidence or item.source_text or "")
    markdown_path = markdown_path_for_item(item)

    if markdown_path:
        context_path = book_context_path_for_page(markdown_path) or markdown_path
        context = context_window_from_file(
            context_path,
            answer=answer,
            before=before,
            after=after,
            question=question,
            evidence=clean_semantic_text(item.source_text or evidence),
            include_adjacent_pages=include_adjacent_pages,
            adjacent_paragraphs=adjacent_paragraphs,
            min_before_chars=min_before_chars,
            min_after_chars=min_after_chars,
        )
        return context, str(context_path.relative_to(REPO_ROOT))

    context = clean_semantic_text(item.source_text or item.answer or "")
    return context, ""


def source_label_for_item(item: CorpusItem, context_path: str) -> str:
    if context_path:
        return f"OCR page: {context_path}"
    parts = [part for part in [item.source_type, item.source_document, item.source_path] if part]
    if item.page_number is not None:
        parts.append(f"page {item.page_number}")
    return "Non-OCR source: " + " | ".join(parts) if parts else "Non-OCR source"


def evidence_prompt(*, provider: str, question: str, answer: str, context: str) -> str:
    return f"""请只基于给定上下文，为题目重新生成证据摘要。

要求：
1. 不得编造，不得加入上下文没有的信息。
2. 先在语义上忽略题目和答案里的 Markdown 标记、井号、列表符号和 OCR 标签。
3. 证据必须覆盖答案所在段落的前后语境，不能只摘一句话。
4. 若上下文包含【上一页相关内容】或【下一页相关内容】，且语义相关，应纳入证据摘要。
5. 保留关键专业名词、数值、条件、图号、限制和结论。
6. 输出 2 到 5 段中文证据摘要，不要输出分析过程。

证据类型：{provider}

题目：
{question}

标准答案：
{answer}

上下文：
{context}
"""


def chat_completion(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是铁路教材审核助手。只基于用户提供的原文上下文抽取证据摘要，"
                    "不得编造，不得加入原文没有的信息。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def summarize_with_deepseek(
    *,
    question: str,
    answer: str,
    context: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> str:
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    prompt = evidence_prompt(provider="DeepSeek", question=question, answer=answer, context=context)
    return chat_completion(prompt, model=model, api_key=api_key, base_url=base_url, timeout=timeout)


def summarize_with_codex(
    *,
    question: str,
    answer: str,
    context: str,
    command: str,
    timeout: int,
) -> str:
    if not command:
        raise RuntimeError("Codex command is not configured; set --codex-command or CODEX_EVIDENCE_COMMAND")
    prompt = evidence_prompt(provider="Codex", question=question, answer=answer, context=context)
    result = subprocess.run(
        shlex.split(command),
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Codex command failed with exit code {result.returncode}: {detail}")
    return result.stdout.strip()


def provider_payload(
    *,
    provider: str,
    evidence: str = "",
    error: str = "",
    model: str = "",
    context: str,
    context_path: str,
    source_label: str = "",
) -> dict[str, str]:
    return {
        "provider": provider,
        "evidence": evidence,
        "error": error,
        "model": model,
        "context": context,
        "context_path": context_path,
        "source_label": source_label,
        "generated_at": utc_now(),
    }


def combined_evidence(evidence_sources: dict[str, dict]) -> str:
    parts: list[str] = []
    labels = {"codex": "Codex 证据", "deepseek": "DeepSeek 证据"}
    for key in ("codex", "deepseek"):
        payload = evidence_sources.get(key) or {}
        text = (payload.get("evidence") or "").strip()
        error = (payload.get("error") or "").strip()
        if text:
            parts.append(f"【{labels[key]}】\n{text}")
        elif error:
            parts.append(f"【{labels[key]}生成失败】\n{error}")
    return "\n\n".join(parts)


def snapshot(item: CorpusItem) -> dict:
    return {
        "question": item.question,
        "answer": item.answer,
        "evidence": item.evidence,
        "metadata_json": item.metadata_json,
        "version": item.version,
    }


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Regenerate Codex and DeepSeek evidence for annotation items.")
    parser.add_argument("--status", default="", help="Only process review_status; empty means all non-deleted.")
    parser.add_argument("--task-type", default="", help="Only process one task_type.")
    parser.add_argument(
        "--exclude-task-types",
        default=",".join(sorted(DEFAULT_EXCLUDED_TASK_TYPES)),
        help="Comma-separated task types excluded from regeneration.",
    )
    parser.add_argument(
        "--providers",
        default="codex,deepseek",
        help="Comma-separated providers to run: codex,deepseek.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=["context", "model"],
        default="context",
        help="context stores related source context verbatim; model asks each provider to rewrite evidence.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to update; 0 means no limit.")
    parser.add_argument("--before", type=int, default=2, help="Paragraphs before answer paragraph.")
    parser.add_argument("--after", type=int, default=2, help="Paragraphs after answer paragraph.")
    parser.add_argument("--min-before-chars", type=int, default=200, help="Minimum source characters before target paragraph when available.")
    parser.add_argument("--min-after-chars", type=int, default=200, help="Minimum source characters after target paragraph when available.")
    parser.add_argument("--adjacent-pages", action="store_true", help="Include related previous/next OCR page snippets.")
    parser.add_argument("--adjacent-paragraphs", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true", help="Skip items that already have all requested providers.")
    parser.add_argument("--clean-fields", action="store_true", help="Also overwrite item.question and item.answer with cleaned text.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deepseek-model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-timeout", type=int, default=int(os.getenv("DEEPSEEK_TIMEOUT", "120")))
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--codex-command", default=os.getenv("CODEX_EVIDENCE_COMMAND", DEFAULT_CODEX_COMMAND))
    parser.add_argument("--codex-timeout", type=int, default=int(os.getenv("CODEX_EVIDENCE_TIMEOUT", "300")))
    args = parser.parse_args()

    requested_providers = parse_csv(args.providers)
    unknown = sorted(set(requested_providers) - {"codex", "deepseek"})
    if unknown:
        raise SystemExit(f"unknown providers: {', '.join(unknown)}")

    excluded_task_types = set(parse_csv(args.exclude_task_types))
    updated = 0
    skipped = 0
    failed = 0

    with SessionLocal() as db:
        filters = [CorpusItem.review_status != "deleted"]
        if args.status:
            filters.append(CorpusItem.review_status == args.status)
        if args.task_type:
            filters.append(CorpusItem.task_type == args.task_type)
        if excluded_task_types:
            filters.append(CorpusItem.task_type.not_in(excluded_task_types))

        statement = select(CorpusItem).where(*filters).order_by(CorpusItem.id)
        if args.limit:
            statement = statement.limit(args.limit)

        for item in db.scalars(statement):
            metadata = dict(item.metadata_json or {})
            evidence_sources = dict(metadata.get("evidence_sources") or {})
            if args.skip_existing and all(
                (evidence_sources.get(provider) or {}).get("evidence") for provider in requested_providers
            ):
                skipped += 1
                continue

            question = clean_semantic_text(item.question or "")
            answer = clean_semantic_text(item.answer or "")
            if not question or not answer:
                skipped += 1
                continue

            context, context_path = source_context_for_item(
                item,
                before=args.before,
                after=args.after,
                include_adjacent_pages=args.adjacent_pages,
                adjacent_paragraphs=args.adjacent_paragraphs,
                min_before_chars=args.min_before_chars,
                min_after_chars=args.min_after_chars,
            )
            if not context:
                skipped += 1
                continue

            source_label = source_label_for_item(item, context_path)
            before_snapshot = snapshot(item)
            generated_any = False

            for provider in requested_providers:
                try:
                    if args.evidence_mode == "context":
                        evidence_sources[provider] = provider_payload(
                            provider="Codex" if provider == "codex" else "DeepSeek",
                            evidence=context,
                            model="source-context",
                            context=context,
                            context_path=context_path,
                            source_label=source_label,
                        )
                    elif provider == "deepseek":
                        evidence = summarize_with_deepseek(
                            question=question,
                            answer=answer,
                            context=context,
                            model=args.deepseek_model,
                            api_key=args.deepseek_api_key,
                            base_url=args.deepseek_base_url,
                            timeout=args.deepseek_timeout,
                        )
                        evidence_sources[provider] = provider_payload(
                            provider="DeepSeek",
                            evidence=evidence,
                            model=args.deepseek_model,
                            context=context,
                            context_path=context_path,
                            source_label=source_label,
                        )
                    elif provider == "codex":
                        evidence = summarize_with_codex(
                            question=question,
                            answer=answer,
                            context=context,
                            command=args.codex_command,
                            timeout=args.codex_timeout,
                        )
                        evidence_sources[provider] = provider_payload(
                            provider="Codex",
                            evidence=evidence,
                            model=args.codex_model,
                            context=context,
                            context_path=context_path,
                            source_label=source_label,
                        )
                    generated_any = True
                except Exception as exc:  # noqa: BLE001 - batch job must keep processing other rows.
                    failed += 1
                    evidence_sources[provider] = provider_payload(
                        provider="Codex" if provider == "codex" else "DeepSeek",
                        error=str(exc),
                        model=args.codex_model if provider == "codex" else args.deepseek_model,
                        context=context,
                        context_path=context_path,
                        source_label=source_label,
                    )

            metadata["evidence_sources"] = evidence_sources
            metadata["evidence_context"] = {
                "before": args.before,
                "after": args.after,
                "min_before_chars": args.min_before_chars,
                "min_after_chars": args.min_after_chars,
                "adjacent_pages": args.adjacent_pages,
                "adjacent_paragraphs": args.adjacent_paragraphs,
                        "context_path": context_path,
                        "source_label": source_label,
                        "generated_at": utc_now(),
                    }
            metadata["cleaned_question"] = question
            metadata["cleaned_answer"] = answer

            merged = combined_evidence(evidence_sources)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "id": item.id,
                            "external_id": item.external_id,
                            "task_type": item.task_type,
                            "context_path": context_path,
                            "cleaned_question": question,
                            "cleaned_answer": answer,
                            "evidence_preview": merged[:500],
                        },
                        ensure_ascii=False,
                    )
                )
                updated += 1
                continue

            if args.clean_fields:
                item.question = question
                item.answer = answer
            if merged:
                item.evidence = merged
            item.metadata_json = metadata
            item.version += 1
            db.add(
                ReviewEvent(
                    item=item,
                    action="dual_evidence_enrich",
                    reviewer="system",
                    comment=f"providers={','.join(requested_providers)}; adjacent_pages={args.adjacent_pages}",
                    snapshot={"before": before_snapshot},
                )
            )
            db.commit()
            updated += 1 if generated_any or evidence_sources else 0

    print(
        json.dumps(
            {
                "updated": updated,
                "skipped": skipped,
                "failed_provider_calls": failed,
                "providers": requested_providers,
                "excluded_task_types": sorted(excluded_task_types),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
