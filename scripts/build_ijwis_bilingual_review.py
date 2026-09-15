"""Build a three-column bilingual paragraph review page for the IJWIS paper."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EN_SOURCE = ROOT / "paper/ijwis/submission/tex/anonymous_manuscript.tex"
ZH_SOURCE = ROOT / "paper/ijwis/submission/anonymous_manuscript_zh.pdf"
OUTPUT = ROOT / "paper/ijwis/bilingual_paragraph_review.html"


SKIP_ENVS = (
    "figure", "table", "table*", "tabular", "tabularx", "equation", "equation*",
    "align", "align*", "algorithm", "algorithmic", "lstlisting",
)


def clean_latex(text: str) -> str:
    text = re.sub(r"\\(?:cite|citep|citet|ref|cref|Cref|autoref)\s*\{([^}]*)\}", r"[\1]", text)
    text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\s*\{[^}]*\}\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|underline|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:label|includegraphics|input|include|bibliography|bibliographystyle)\s*(?:\[[^]]*\])?\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?\s*", "", text)
    text = re.sub(r"\\([^\s])", r"\1", text)
    for source, target in {
        "~": " ", "\\&": "&", "\\%": "%", "\\#": "#", "\\_": "_",
        "\\textendash": "-", "\\textemdash": "-", "---": "-", "--": "-",
        "``": '"', "''": '"', "\\,": " ", "\\;": " ", "\\!": "",
    }.items():
        text = text.replace(source, target)
    text = re.sub(r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_blocks(text: str) -> list[str]:
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if block:
            blocks.append(block)
    return blocks


def extract_english() -> tuple[str, list[tuple[str, list[str]]], list[tuple[str, str]]]:
    raw = EN_SOURCE.read_text(encoding="utf-8")
    title_match = re.search(r"\\title\s*\{([^{}]*)\}", raw)
    title = clean_latex(title_match.group(1)) if title_match else "IJWIS Manuscript"
    body = raw.split(r"\begin{document}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"(?<!\\)%.*", "", body)
    figures: list[tuple[str, str]] = []
    figure_matches = list(re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", body, re.DOTALL))
    for match in figure_matches:
        image = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", match.group(1))
        caption = re.search(r"\\caption\{([^{}]*)\}", match.group(1), re.DOTALL)
        if image:
            figures.append((image.group(1), clean_latex(caption.group(1)) if caption else ""))
    for env in SKIP_ENVS:
        if env == "figure":
            continue
        body = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", "\n", body, flags=re.DOTALL)
    figure_number = 0
    def figure_placeholder(_match: re.Match[str]) -> str:
        nonlocal figure_number
        value = f"\n[[FIGURE_{figure_number}]]\n"
        figure_number += 1
        return value
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", figure_placeholder, body, flags=re.DOTALL)
    heading_re = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{([^{}]*)\}")
    sections: list[tuple[str, list[str]]] = [("Abstract", [])]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", raw, re.DOTALL)
    if abstract:
        sections[0][1].extend(split_blocks(clean_latex(abstract.group(1))))
    matches = list(heading_re.finditer(body))
    for index, match in enumerate(matches):
        heading = clean_latex(match.group(2))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        segment = body[start:end]
        paragraphs = [clean_latex(block) for block in split_blocks(segment)]
        paragraphs = [p for p in paragraphs if p and not p.startswith(("documentclass", "usepackage"))]
        sections.append((heading, paragraphs))
    return title, sections, figures


def extract_chinese() -> list[str]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise RuntimeError("pdftotext is required to extract the Chinese PDF")
    result = subprocess.run(
        [pdftotext, "-layout", "-nopgbrk", str(ZH_SOURCE), "-"],
        check=True, capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*资源约束下.*?\n", "", text, count=1, flags=re.MULTILINE)
    blocks = []
    for block in split_blocks(text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        if re.fullmatch(r"(?:\d+(?:\.\d+)*|图\s*\d+|表\s*\d+)", block):
            continue
        blocks.append(block)
    return blocks


def render(title: str, sections: list[tuple[str, list[str]]], chinese: list[str], figures: list[tuple[str, str]]) -> str:
    zh_index = 0
    paragraph_id = 0
    rows: list[str] = []
    toc: list[str] = []
    for section_name, english_blocks in sections:
        anchor = f"section-{len(toc)}"
        toc.append(f'<a href="#{anchor}">{html.escape(section_name)}</a>')
        rows.append(f'<tr class="section"><th id="{anchor}" colspan="4">{html.escape(section_name)}</th></tr>')
        for english in english_blocks:
            figure_match = re.fullmatch(r"\[\[FIGURE_(\d+)\]\]", english)
            if figure_match:
                image_path, caption = figures[int(figure_match.group(1))]
                relative = image_path.replace(".pdf", ".png")
                image_file = ROOT / "paper/ijwis" / relative
                if image_file.exists():
                    media = f'<img src="{html.escape(relative)}" alt="Figure">'
                else:
                    media = f'<object data="{html.escape(image_path)}" type="application/pdf" aria-label="Figure"></object>'
                rows.append(
                    f'<tr class="figure"><td class="number">FIG</td><td colspan="2">{media}'
                    f'<div>{html.escape(caption)}</div></td><td class="editable" contenteditable="true" spellcheck="true"></td></tr>'
                )
                continue
            paragraph_id += 1
            chinese_text = chinese[zh_index] if zh_index < len(chinese) else ""
            if chinese_text and re.match(r"^(图|表)\s*\d+", chinese_text):
                chinese_text = ""
            else:
                zh_index += 1
            rows.append(
                f'<tr><td class="number">{paragraph_id:04d}</td>'
                f'<td>{html.escape(english)}</td><td>{html.escape(chinese_text)}</td>'
                f'<td class="editable" contenteditable="true" spellcheck="true"></td></tr>'
            )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>中英论文逐段核对 - {html.escape(title)}</title>
<style>
:root {{ --ink:#1f2937; --muted:#64748b; --line:#cbd5e1; --paper:#fff; --bg:#f1f5f9; --accent:#1d4ed8; --edit:#fffbe6; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ padding:24px max(16px,calc((100vw - 1500px)/2)); background:var(--paper); border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:24px; }} .meta {{ color:var(--muted); font-size:13px; }}
.layout {{ display:grid; grid-template-columns:230px minmax(0,1fr); gap:18px; max-width:1500px; margin:0 auto; padding:18px; }}
aside {{ position:sticky; top:12px; align-self:start; max-height:calc(100vh - 24px); overflow:auto; padding:14px; background:var(--paper); border:1px solid var(--line); border-radius:6px; font-size:13px; }}
aside strong {{ display:block; margin-bottom:8px; }} aside a {{ display:block; padding:4px 0; color:var(--accent); text-decoration:none; }}
main {{ min-width:0; background:var(--paper); border:1px solid var(--line); border-radius:6px; overflow:auto; }}
table {{ width:100%; min-width:1050px; border-collapse:collapse; table-layout:fixed; }}
th,td {{ border-bottom:1px solid #e2e8f0; vertical-align:top; padding:10px 12px; line-height:1.6; white-space:pre-wrap; overflow-wrap:anywhere; }}
thead th {{ position:sticky; top:0; z-index:2; background:#e2e8f0; text-align:left; font-size:14px; }}
.number {{ width:64px; color:var(--accent); font:12px ui-monospace,Consolas,monospace; text-align:center; }}
thead th:nth-child(2), tbody td:nth-child(2) {{ width:34%; }} thead th:nth-child(3), tbody td:nth-child(3) {{ width:34%; }}
thead th:nth-child(4), tbody td:nth-child(4) {{ width:27%; }}
.section th {{ background:#dbeafe; color:#1e3a8a; font-size:16px; text-align:left; scroll-margin-top:55px; }}
.editable {{ min-height:74px; background:var(--edit); outline:none; }} .editable:focus {{ box-shadow:inset 0 0 0 2px #f59e0b; }}
.figure td {{ background:#f8fafc; }} .figure img, .figure object {{ display:block; width:100%; max-height:420px; height:auto; margin-bottom:8px; }}
@media (max-width:800px) {{ .layout {{ display:block; padding:10px; }} aside {{ position:static; max-height:180px; margin-bottom:10px; }} h1 {{ font-size:20px; }} }}
</style></head><body>
<header><h1>中英论文逐段核对与修改</h1><div class="meta">英文来源：anonymous_manuscript.tex；中文来源：anonymous_manuscript_zh.pdf；第三列可直接点击编辑。图片使用当前 figures 目录中的导出文件。</div></header>
<div class="layout"><aside><strong>章节导航</strong>{''.join(toc)}</aside><main><table><thead><tr><th class="number">编号</th><th>英文原文</th><th>中文原文</th><th>人工修改</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></div>
</body></html>'''


def main() -> None:
    title, sections, figures = extract_english()
    chinese = extract_chinese()
    OUTPUT.write_text(render(title, sections, chinese, figures), encoding="utf-8")
    count = sum(len(items) for _, items in sections)
    print(f"Wrote {OUTPUT} with {count} English paragraphs and {len(chinese)} Chinese blocks")


if __name__ == "__main__":
    main()
