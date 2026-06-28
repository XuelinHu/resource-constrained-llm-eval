"""Run local DeepSeek-OCR-2 over a PDF, page by page.

The script keeps outputs resumable: completed pages with non-empty markdown are
skipped unless --force is passed. It renders pages with pdftoppm, loads the OCR
model once, and records per-page timing and CUDA memory in manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path("/ds1/workspace/ai/DS-OCR2-3090-Runner/models/DeepSeek-OCR-2")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "ocr" / "railway"
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 PDF batch runner.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--base-size", type=int, default=1536)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--no-crop-mode", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--drop-images-after-ocr", action="store_true")
    return parser.parse_args()


def slugify(path: Path) -> str:
    text = path.stem
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:96] or "pdf"


def run_text(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout


def pdf_pages(pdf: Path) -> int:
    output = run_text(["pdfinfo", str(pdf)])
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Cannot detect PDF page count: {pdf}")


def render_page(pdf: Path, page: int, dpi: int, image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_dir / f"page_{page:04d}"
    expected = image_dir / f"page_{page:04d}-{page}.png"
    final = image_dir / f"page_{page:04d}.png"
    if final.is_file() and final.stat().st_size > 0:
        return final
    for stale in image_dir.glob(f"page_{page:04d}*.png"):
        stale.unlink()
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-png",
            str(pdf),
            str(prefix),
        ],
        check=True,
    )
    candidates = sorted(image_dir.glob(f"page_{page:04d}*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm did not create an image for page {page}")
    produced = expected if expected in candidates else candidates[0]
    produced.rename(final)
    return final


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def read_page_markdown(page_output_dir: Path, fallback: str | None) -> str:
    result = page_output_dir / "result.mmd"
    if result.is_file():
        return result.read_text(encoding="utf-8")
    return fallback or ""


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    pdf = args.pdf.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    total_pages = pdf_pages(pdf)
    end_page = args.end_page or total_pages
    end_page = min(end_page, total_pages)
    pages = list(range(max(1, args.start_page), end_page + 1))
    if args.max_pages:
        pages = pages[: args.max_pages]

    book_dir = output_root / slugify(pdf)
    page_dir = book_dir / "pages"
    image_dir = book_dir / "images"
    backend_dir = book_dir / "deepseek_outputs"
    for directory in (page_dir, image_dir, backend_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = book_dir / "manifest.jsonl"
    run_meta = {
        "pdf": str(pdf),
        "model_dir": str(model_dir),
        "total_pages": total_pages,
        "selected_pages": pages,
        "dpi": args.dpi,
        "base_size": args.base_size,
        "image_size": args.image_size,
        "crop_mode": not args.no_crop_mode,
        "dtype": args.dtype,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (book_dir / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir,
        _attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model = model.eval().cuda().to(dtype)

    try:
        for page in pages:
            markdown_path = page_dir / f"page_{page:04d}.md"
            if markdown_path.is_file() and markdown_path.stat().st_size > 0 and not args.force:
                print(f"[skip] page {page}/{total_pages}: {markdown_path}", flush=True)
                continue

            started = time.time()
            image_path = render_page(pdf, page, args.dpi, image_dir)
            page_backend_dir = backend_dir / f"page_{page:04d}"
            if page_backend_dir.exists() and args.force:
                shutil.rmtree(page_backend_dir)
            page_backend_dir.mkdir(parents=True, exist_ok=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            returned = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=str(image_path),
                output_path=str(page_backend_dir),
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=not args.no_crop_mode,
                save_results=True,
            )
            markdown = read_page_markdown(page_backend_dir, returned)
            markdown_path.write_text(markdown, encoding="utf-8")

            peak_allocated = None
            peak_reserved = None
            if torch.cuda.is_available():
                peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
                peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 4)
            record = {
                "page": page,
                "total_pages": total_pages,
                "pdf": str(pdf),
                "image": "" if args.drop_images_after_ocr else str(image_path),
                "markdown": str(markdown_path),
                "chars": len(markdown),
                "elapsed_s": round(time.time() - started, 3),
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_jsonl(manifest, record)
            print(
                f"[done] page {page}/{total_pages} chars={len(markdown)} "
                f"elapsed={record['elapsed_s']}s peak_reserved={peak_reserved}GB",
                flush=True,
            )
            if args.drop_images_after_ocr and image_path.exists():
                image_path.unlink()

        combined = book_dir / "combined.md"
        with combined.open("w", encoding="utf-8") as handle:
            handle.write(f"# {pdf.stem}\n\n")
            for page_file in sorted(page_dir.glob("page_*.md")):
                page_no = int(page_file.stem.split("_")[-1])
                handle.write(f"\n\n## Page {page_no}\n\n")
                handle.write(page_file.read_text(encoding="utf-8").strip())
                handle.write("\n")
        print(f"[complete] {pdf.name} -> {combined}", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
