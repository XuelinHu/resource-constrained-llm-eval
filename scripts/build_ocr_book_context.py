from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OCR_ROOT = REPO_ROOT / "data" / "ocr" / "railway"
OUTPUT_ROOT = REPO_ROOT / "data" / "ocr" / "railway_context"

OCR_REF_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.S)
OCR_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_line(line: str) -> str:
    line = OCR_REF_RE.sub("", line)
    line = OCR_DET_RE.sub("", line)
    line = MARKDOWN_IMAGE_RE.sub("", line)
    line = MARKDOWN_LINK_RE.sub(r"\1", line)
    line = HTML_TAG_RE.sub("", line)
    line = line.replace("\u3000", " ")
    line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
    line = re.sub(r"#{2,}", "", line)
    line = re.sub(r"[*_`]+", "", line)
    line = SPACE_RE.sub(" ", line).strip()
    line = re.sub(r"([：:，,。；;])\s+(?=[\u4e00-\u9fff一二三四五六七八九十百\d(（])", r"\1", line)
    line = line.replace("关于 ", "关于")
    return line


def clean_page_text(markdown: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in markdown.splitlines():
        line = normalize_line(raw_line)
        if not line:
            if lines and not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def page_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def build_book(book_dir: Path) -> dict:
    pages_dir = book_dir / "pages"
    page_paths = sorted(pages_dir.glob("page_*.md"), key=page_number)
    if not page_paths:
        return {}

    output_dir = OUTPUT_ROOT / book_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    book_context_path = output_dir / "book_context.md"
    page_index_path = output_dir / "page_index.json"

    config_path = book_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    source_pdf = str(config.get("pdf") or book_dir.name)

    chunks = [
        f"**教材：{book_dir.name}**",
        f"**来源 PDF：{source_pdf}**",
        "",
        "<!-- 本文件由 scripts/build_ocr_book_context.py 生成；OCR 原文已清理 Markdown 井号、图片标记和检测标签。 -->",
    ]
    page_index: list[dict] = []

    for page_path in page_paths:
        number = page_number(page_path)
        rel_page = page_path.relative_to(REPO_ROOT).as_posix()
        rel_image = rel_page.replace("/pages/", "/images/").removesuffix(".md") + ".png"
        cleaned = clean_page_text(page_path.read_text(encoding="utf-8"))
        anchor = f"page-{number:04d}"
        chunks.extend(
            [
                "",
                f'<a id="{anchor}"></a>',
                "",
                f"**第 {number} 页**",
                "",
                f"来源页：`{rel_page}`",
                "",
                f"原图：`{rel_image}`",
                "",
                cleaned,
            ]
        )
        page_index.append(
            {
                "page": number,
                "anchor": anchor,
                "page_path": rel_page,
                "image_path": rel_image,
                "book_context_path": book_context_path.relative_to(REPO_ROOT).as_posix(),
                "chars": len(cleaned),
            }
        )

    book_context_path.write_text("\n".join(chunks).strip() + "\n", encoding="utf-8")
    page_index_path.write_text(json.dumps(page_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "book": book_dir.name,
        "source_pdf": source_pdf,
        "pages": len(page_index),
        "book_context_path": book_context_path.relative_to(REPO_ROOT).as_posix(),
        "page_index_path": page_index_path.relative_to(REPO_ROOT).as_posix(),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    books = []
    all_chunks = ["**铁路 OCR 教材全书上下文汇总**", ""]
    for book_dir in sorted(path for path in OCR_ROOT.iterdir() if path.is_dir()):
        result = build_book(book_dir)
        if not result:
            continue
        books.append(result)
        book_context = REPO_ROOT / result["book_context_path"]
        all_chunks.extend(["", f"**教材：{result['book']}**", "", book_context.read_text(encoding="utf-8")])

    all_context_path = OUTPUT_ROOT / "all_books_context.md"
    manifest_path = OUTPUT_ROOT / "manifest.json"
    all_context_path.write_text("\n".join(all_chunks).strip() + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"books": books}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"books": len(books), "output": all_context_path.relative_to(REPO_ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
