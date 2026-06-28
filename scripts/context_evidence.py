from __future__ import annotations

import re
from pathlib import Path


MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
OCR_REF_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.S)
OCR_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
MARKDOWN_PREFIX_RE = re.compile(r"^[#>\-\*\+\d\.\)\(、\s]+")
MARKDOWN_INLINE_RE = re.compile(r"[*_`~#]+")
PAGE_FILE_RE = re.compile(r"page_(\d+)\.md$", re.I)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_markdown_text(markdown: str) -> str:
    text = OCR_REF_RE.sub("", markdown)
    text = OCR_DET_RE.sub("", text)
    text = MARKDOWN_IMAGE_RE.sub("", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    text = MARKDOWN_INLINE_RE.sub("", text)
    return text


def clean_semantic_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in clean_markdown_text(text).splitlines():
        line = normalize_text(MARKDOWN_PREFIX_RE.sub("", raw_line))
        if line:
            lines.append(line)
    return normalize_text(" ".join(lines))


def split_paragraphs(markdown: str) -> list[str]:
    text = clean_markdown_text(markdown)
    paragraphs: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = normalize_text(MARKDOWN_PREFIX_RE.sub("", raw_line))
        if not line:
            if current:
                paragraphs.append(normalize_text(" ".join(current)))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(normalize_text(" ".join(current)))

    return [paragraph for paragraph in paragraphs if len(paragraph) >= 8]


def _keywords(*texts: str) -> set[str]:
    joined = clean_semantic_text(" ".join(texts))
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?", joined))
    stopwords = {
        "什么",
        "哪些",
        "如何",
        "进行",
        "要求",
        "答案",
        "问题",
        "标准",
        "应当",
        "可以",
        "不得",
    }
    return {token for token in tokens if token not in stopwords}


def _target_index(paragraphs: list[str], question: str = "", answer: str = "", evidence: str = "") -> int:
    candidates = [
        clean_semantic_text(answer),
        clean_semantic_text(evidence),
        clean_semantic_text(question),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for index, paragraph in enumerate(paragraphs):
            if candidate in paragraph or paragraph in candidate:
                return index

    keywords = _keywords(question, answer, evidence)
    if not keywords:
        return 0

    best_index = 0
    best_score = -1
    for index, paragraph in enumerate(paragraphs):
        paragraph_keywords = _keywords(paragraph)
        score = len(keywords & paragraph_keywords)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _window_bounds_with_min_chars(
    paragraphs: list[str],
    target_index: int,
    *,
    before: int,
    after: int,
    min_before_chars: int,
    min_after_chars: int,
) -> tuple[int, int]:
    start = max(0, target_index - before)
    end = min(len(paragraphs), target_index + after + 1)

    before_chars = sum(len(paragraph) for paragraph in paragraphs[start:target_index])
    while start > 0 and before_chars < min_before_chars:
        start -= 1
        before_chars += len(paragraphs[start])

    after_chars = sum(len(paragraph) for paragraph in paragraphs[target_index + 1 : end])
    while end < len(paragraphs) and after_chars < min_after_chars:
        after_chars += len(paragraphs[end])
        end += 1

    return start, end


def context_window_from_markdown(
    markdown: str,
    answer: str = "",
    before: int = 2,
    after: int = 2,
    *,
    question: str = "",
    evidence: str = "",
    min_before_chars: int = 200,
    min_after_chars: int = 200,
) -> str:
    paragraphs = split_paragraphs(markdown)
    if not paragraphs:
        return normalize_text(clean_markdown_text(markdown))

    target_index = _target_index(paragraphs, question=question, answer=answer, evidence=evidence)
    start, end = _window_bounds_with_min_chars(
        paragraphs,
        target_index,
        before=before,
        after=after,
        min_before_chars=min_before_chars,
        min_after_chars=min_after_chars,
    )
    return "\n\n".join(paragraphs[start:end])


def page_number_from_path(path: Path) -> int | None:
    match = PAGE_FILE_RE.search(path.name)
    return int(match.group(1)) if match else None


def sibling_page_path(path: Path, offset: int) -> Path | None:
    page_number = page_number_from_path(path)
    if page_number is None:
        return None
    candidate = path.with_name(f"page_{page_number + offset:04d}.md")
    return candidate if candidate.is_file() else None


def _has_hard_boundary(paragraphs: list[str], *, from_previous: bool) -> bool:
    if not paragraphs:
        return False
    probe = paragraphs[-1] if from_previous else paragraphs[0]
    return bool(re.match(r"^(第[一二三四五六七八九十百\d]+[章节]|[一二三四五六七八九十]+、)", probe))


def _semantically_related(paragraphs: list[str], question: str, answer: str, evidence: str) -> bool:
    if not paragraphs:
        return False
    query_keywords = _keywords(question, answer, evidence)
    if not query_keywords:
        return False
    page_keywords = _keywords(" ".join(paragraphs))
    return len(query_keywords & page_keywords) >= 2


def context_window_from_file(
    path: Path,
    answer: str = "",
    before: int = 2,
    after: int = 2,
    *,
    question: str = "",
    evidence: str = "",
    include_adjacent_pages: bool = False,
    adjacent_paragraphs: int = 2,
    min_before_chars: int = 200,
    min_after_chars: int = 200,
) -> str:
    markdown = path.read_text(encoding="utf-8")
    current_context = context_window_from_markdown(
        markdown,
        answer=answer,
        before=before,
        after=after,
        question=question,
        evidence=evidence,
        min_before_chars=min_before_chars,
        min_after_chars=min_after_chars,
    )
    if not include_adjacent_pages:
        return current_context

    context_parts: list[str] = []
    previous_path = sibling_page_path(path, -1)
    if previous_path:
        previous_paragraphs = split_paragraphs(previous_path.read_text(encoding="utf-8"))
        tail = previous_paragraphs[-adjacent_paragraphs:]
        if not _has_hard_boundary(previous_paragraphs, from_previous=True) and _semantically_related(
            tail, question, answer, evidence
        ):
            context_parts.append("【上一页相关内容】\n" + "\n\n".join(tail))

    context_parts.append("【当前页相关内容】\n" + current_context)

    next_path = sibling_page_path(path, 1)
    if next_path:
        next_paragraphs = split_paragraphs(next_path.read_text(encoding="utf-8"))
        head = next_paragraphs[:adjacent_paragraphs]
        if not _has_hard_boundary(next_paragraphs, from_previous=False) and _semantically_related(
            head, question, answer, evidence
        ):
            context_parts.append("【下一页相关内容】\n" + "\n\n".join(head))

    return "\n\n".join(context_parts)
