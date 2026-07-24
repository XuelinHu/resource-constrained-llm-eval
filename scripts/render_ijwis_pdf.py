"""Render the IJWIS Markdown manuscript as a publication-style PDF."""

from __future__ import annotations

import argparse
import html
import subprocess
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/ijwis/manuscript.md"
OUTPUT = ROOT / "output/pdf/ijwis_bilingual_railway_manuscript.pdf"
HTML_OUTPUT = ROOT / "tmp/pdfs/ijwis_bilingual_railway_manuscript.html"
PLAYWRIGHT_CHROMIUM = Path.home() / ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"

FIGURES = {
    "## 4. Results": (
        "system_architecture.png",
        "Figure 1. System architecture and knowledge-governance workflow.",
    ),
    "### 4.4 Regulation-only pilot answer generation": (
        "top_k_quality_latency.png",
        "Figure 2. Approved-hybrid retrieval quality and latency across top-k settings.",
    ),
    "### 4.6 Multi-generator RAG and interaction effects": (
        "training_validation_loss.png",
        "Figure 3. Completion-only QLoRA training and validation loss.",
    ),
    "### 4.7 Directional translation": (
        "quality_latency_pareto.png",
        "Figure 4. Bilingual QA quality against generation latency and peak GPU memory.",
    ),
    "### 4.8 Resource use and automated error analysis": (
        "translation_before_after.png",
        "Figure 5. Direction- and task-separated COMET before and after QLoRA.",
    ),
    "## 5. Discussion": (
        "error_type_distribution.png",
        "Figure 6. Mean prevalence of automatically flagged output errors.",
    ),
}


def figure_markup(filename: str, caption: str) -> str:
    uri = (ROOT / "paper/ijwis/figures" / filename).resolve().as_uri()
    return (
        '<figure class="paper-figure">'
        f'<img src="{uri}" alt="{html.escape(caption)}">'
        f'<figcaption>{html.escape(caption)}</figcaption>'
        "</figure>"
    )


def inject_figures(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    for line in lines:
        if line in FIGURES:
            filename, caption = FIGURES[line]
            output.extend(["", figure_markup(filename, caption), ""])
        output.append(line)
    return "\n".join(output)


def build_html() -> str:
    source = inject_figures(MANUSCRIPT.read_text(encoding="utf-8"))
    body = markdown.markdown(
        source,
        extensions=["extra", "tables", "sane_lists", "smarty", "toc"],
        output_format="html5",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IJWIS - Bilingual Railway Vocational Education</title>
<style>
@page {{
  size: A4;
  margin: 19mm 18mm 21mm;
  @bottom-center {{
    content: counter(page);
    font-family: "Noto Sans", sans-serif;
    font-size: 8.5pt;
    color: #5b626a;
  }}
}}
* {{ box-sizing: border-box; }}
html {{ font-size: 10.4pt; }}
body {{
  margin: 0;
  color: #202428;
  font-family: "Noto Serif", "DejaVu Serif", serif;
  line-height: 1.5;
  text-rendering: optimizeLegibility;
}}
h1 {{
  margin: 14mm auto 7mm;
  max-width: 165mm;
  color: #172a3a;
  font-family: "Noto Sans", "DejaVu Sans", sans-serif;
  font-size: 22pt;
  line-height: 1.18;
  text-align: center;
}}
h1::after {{
  content: "Anonymous manuscript prepared for the International Journal of Web Information Systems";
  display: block;
  margin-top: 5mm;
  color: #68727c;
  font-size: 9pt;
  font-weight: 400;
}}
h2, h3 {{
  color: #19374a;
  font-family: "Noto Sans", "DejaVu Sans", sans-serif;
  break-after: avoid-page;
}}
h2 {{
  margin: 8mm 0 3mm;
  padding-bottom: 1.5mm;
  border-bottom: 0.5pt solid #92a7b4;
  font-size: 15pt;
}}
h3 {{ margin: 5mm 0 2mm; font-size: 11.5pt; }}
p {{ margin: 0 0 3.2mm; text-align: justify; orphans: 3; widows: 3; }}
ul, ol {{ margin: 2mm 0 3mm 6mm; padding-left: 5mm; }}
li {{ margin-bottom: 1.2mm; }}
strong {{ color: #132f40; }}
code {{
  padding: 0.2mm 0.7mm;
  border-radius: 1mm;
  background: #eef2f4;
  font-family: "Noto Sans Mono", monospace;
  font-size: 8.7pt;
}}
table {{
  width: 100%;
  margin: 3mm 0 5mm;
  border-collapse: collapse;
  font-family: "Noto Sans", "DejaVu Sans", sans-serif;
  font-size: 8.2pt;
  line-height: 1.3;
  break-inside: avoid-page;
}}
thead {{ display: table-header-group; }}
th {{
  padding: 1.5mm 1.2mm;
  border-top: 1.2pt solid #294b5f;
  border-bottom: 0.7pt solid #5d7888;
  background: #edf2f4;
  color: #173448;
  text-align: left;
}}
td {{ padding: 1.25mm 1.2mm; border-bottom: 0.35pt solid #ccd5da; }}
tbody tr:last-child td {{ border-bottom: 1.1pt solid #294b5f; }}
.paper-figure {{ margin: 4mm auto 6mm; break-inside: avoid-page; text-align: center; }}
.paper-figure img {{ display: block; width: 100%; max-height: 190mm; object-fit: contain; }}
.paper-figure figcaption {{
  margin-top: 2mm;
  color: #3e4d57;
  font-family: "Noto Sans", "DejaVu Sans", sans-serif;
  font-size: 8.8pt;
  text-align: left;
}}
h2:first-of-type + p {{ margin-top: 2mm; }}
h2:first-of-type ~ p strong:first-child {{ font-family: "Noto Sans", sans-serif; }}
a {{ color: #285d7a; text-decoration: none; }}
</style>
</head>
<body>{body}</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chromium",
        default=str(PLAYWRIGHT_CHROMIUM if PLAYWRIGHT_CHROMIUM.exists() else "/snap/bin/chromium"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(build_html(), encoding="utf-8")

    subprocess.run(
        [
            args.chromium,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output}",
            HTML_OUTPUT.resolve().as_uri(),
        ],
        check=True,
    )
    if not output.exists() or output.stat().st_size < 10_000:
        raise RuntimeError(f"Chromium did not create a valid PDF: {output}")
    print(output)


if __name__ == "__main__":
    main()
