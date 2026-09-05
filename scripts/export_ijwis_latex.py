"""Export the authoritative IJWIS Markdown draft into the LaTeX submission template."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/ijwis/manuscript.md"
OUTPUT = ROOT / "IJWIS__Copy_/manuscript_body.tex"
BIBLIOGRAPHY = ROOT / "paper/ijwis/references.bib"

FIGURES: dict[str, list[tuple[str, str]]] = {
    "### 3.3 Retrieval and answer generation": [
        ("system_architecture.pdf", "Four-layer bounded bilingual agent workflow linking three role-access clusters to Web applications, governed retrieval, local generation and single-workstation infrastructure. Solid arrows denote runtime or data flow; dashed arrows denote governance or control. Source: Authors' own work."),
        ("knowledge_governance_lifecycle.pdf", "Expert-governed knowledge, production-index and evaluation lifecycle with exact held-out records excluded from indexing and training. Source: Authors' own work."),
    ],
    "### 4.3 QLoRA adaptation and held-out QA": [
        ("top_k_quality_latency.pdf", "Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work."),
    ],
    "### 4.4 Multi-generator RAG comparisons": [
        ("training_validation_loss.pdf", "Completion-only QLoRA training and validation loss for Qwen2.5-7B and GLM-4-9B. Source: Authors' own work."),
    ],
    "### 4.6 Resource use and automated error analysis": [
        ("translation_before_after.pdf", "Direction- and task-separated COMET before and after QLoRA. Left: Qwen2.5-7B; right: GLM-4-9B. Source: Authors' own work."),
    ],
    "### 4.7 Index, evidence-support and governance validation": [
        ("quality_latency_pareto.pdf", "Bilingual QA quality against generation latency and peak GPU memory. Left: mean generation latency; right: peak reserved GPU memory. Source: Authors' own work."),
        ("error_type_distribution.pdf", "Mean prevalence of automatically flagged output errors across the evaluated generator and retrieval conditions. Source: Authors' own work."),
        ("supplementary_system_validation.pdf", "Bilingual index, automated evidence support and governance-history validation. Panels A-C report field ablation, evidence support and governance audit results, respectively. Source: Authors' own work."),
    ],
}


# Keep the Markdown source readable for the Word submission while making the
# LaTeX PDF use a real citation graph checked by BibTeX/natbib.
CITATION_KEYS: dict[tuple[str, str], str] = {
    ("Barnett", "2024"): "barnett2024finetuning",
    ("Chang", "2024"): "chang2024survey",
    ("Chen", "2024"): "chen2024bgem3",
    ("Cormack", "2009"): "cormack2009rrf",
    ("Dettmers", "2023"): "dettmers2023qlora",
    ("Ding", "2023"): "ding2023peft",
    ("GLM Team", "2024"): "glm2024chatglm",
    ("Hu", "2024"): "hu2024translation",
    ("Huang", "2026"): "huang2026emergency",
    ("Hubscher", "2021"): "hubscher2021knowledge",
    ("Kasneci", "2023"): "kasneci2023chatgpt",
    ("Lahnalampi", "2024"): "lahnalampi2024rail",
    ("Lawrence", "2019"): "lawrence2019hsr",
    ("Lewis", "2020"): "lewis2020rag",
    ("Li", "2024"): "li2024rschat",
    ("Li", "2025"): "li2025training",
    ("Luo", "2025"): "luo2025driver",
    ("Lv", "2024"): "lv2024full",
    ("Qwen Team", "2024"): "qwen2024technical",
    ("Qwen Team", "2025"): "qwen2025technical",
    ("Robertson", "2009"): "robertson2009bm25",
    ("Schwartz", "2020"): "schwartz2020green",
    ("Song", "2025"): "song2025railway",
    ("Tian", "2024"): "tian2024factuality",
    ("UNESCO", "2023"): "unesco2023guidance",
    ("Vieira", "2024"): "vieira2024translation",
    ("Wandelt", "2024"): "wandelt2024transport",
    ("Wang", "2024"): "wang2024phm",
    ("Wang", "2025"): "wang2025peft",
    ("Wooldridge", "1995"): "wooldridge1995agents",
    ("Xi", "2025"): "xi2025agents",
    ("Xu", "2018"): "xu2018education",
    ("Yang", "2024"): "yang2024legalqa",
    ("Zheng", "2023"): "zheng2023trafficsafety",
    ("Zheng", "2024"): "zheng2024translation",
}

CITATION_GROUP = re.compile(r"\(([^()\n]*?(?:19|20)\d{2}[^()\n]*?)\)")
CITATION_ITEM = re.compile(
    r"([A-Z][A-Za-z-]*(?:\s+Team)?)(?:\s+\*?et al\.\*?|\s+and\s+[A-Z][A-Za-z-]+)?\s*,\s*(\d{4})"
)


def convert_citations(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        group = match.group(1)
        items = CITATION_ITEM.findall(group)
        if not items:
            return match.group(0)
        keys: list[str] = []
        for author, year in items:
            key = CITATION_KEYS.get((author, year))
            if key is None:
                raise ValueError(f"No BibTeX key configured for citation: {author}, {year}")
            keys.append(key)
        return r"\citep{" + ",".join(keys) + "}"

    return CITATION_GROUP.sub(replace, markdown)


def validate_citation_graph(markdown: str) -> None:
    cited = {
        key
        for group in re.findall(r"\\citep\{([^}]+)\}", markdown)
        for key in group.split(",")
    }
    bibliography = set(re.findall(r"^@\w+\{([^,]+),", BIBLIOGRAPHY.read_text(encoding="utf-8"), re.MULTILINE))
    missing = sorted(cited - bibliography)
    unused = sorted(bibliography - cited)
    if missing or unused:
        raise ValueError(f"Citation graph mismatch: missing={missing}, unused={unused}")


def prepare_markdown(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        if line == "## References":
            break
        if index == 0 and line.startswith("# "):
            continue
        if line == "## Structured abstract":
            output.extend(["```{=latex}", r"\begin{abstract}", "```"])
            continue
        if line == "## 1. Introduction":
            output.extend(["```{=latex}", r"\end{abstract}", "```", "## Introduction"])
            continue
        if line in FIGURES:
            for filename, caption in FIGURES[line]:
                path = f"../paper/ijwis/figures/{filename}"
                output.extend(["", f"![{caption}]({path}){{width=96%}}", ""])
        line = re.sub(r"^(#{2,3})\s+\d+(?:\.\d+)?\.?(?:\s+)", r"\1 ", line)
        if line.startswith('<div class="equation">'):
            output.extend(
                [
                    "```{=latex}",
                    r"\begin{equation}",
                    r"\operatorname{RRF}(d)=\sum_{r \in \{\mathrm{BM25},\mathrm{vector}\}}"
                    r"\frac{1}{k_0+\operatorname{rank}_r(d)}.",
                    r"\end{equation}",
                    "```",
                ]
            )
            continue
        output.append(line)
    return convert_citations("\n".join(output) + "\n")


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def inline_to_latex(text: str) -> str:
    """Convert the small inline-Markdown subset used by this manuscript."""
    protected: dict[str, str] = {}

    def token(value: str) -> str:
        key = f"@@LATEX{len(protected)}@@"
        protected[key] = value
        return key

    text = re.sub(r"\\citep\{[^}]+\}", lambda match: token(match.group(0)), text)
    text = re.sub(
        r"`([^`]+)`",
        lambda match: token(r"\texttt{" + escape_latex(match.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"<sub>([^<]+)</sub>",
        lambda match: token(r"\textsubscript{" + escape_latex(match.group(1)) + "}"),
        text,
    )
    text = escape_latex(text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda match: r"\textbf{" + match.group(1) + "}", text)
    text = re.sub(r"\*([^*]+)\*", lambda match: r"\emph{" + match.group(1) + "}", text)
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def table_to_latex(lines: list[str], number: str) -> list[str]:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    if len(rows) < 2:
        raise ValueError(f"Malformed Markdown table {number}")
    rows.pop(1)  # Markdown alignment/separator row.
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError(f"Inconsistent column count in Markdown table {number}")

    specifications = {
        "I": r"p{0.28\linewidth}p{0.64\linewidth}",
        "II": r">{\raggedright\arraybackslash}p{0.19\linewidth}lrrrrr",
        "III": r"p{0.38\linewidth}ccp{0.34\linewidth}",
    }
    specification = specifications.get(number, "l" * column_count)
    output: list[str] = []
    if number == "II":
        output.extend([r"\small", r"\begin{adjustbox}{max width=\linewidth}"])
    output.extend([rf"\begin{{tabular}}{{{specification}}}", r"\toprule"])
    for index, row in enumerate(rows):
        output.append(" & ".join(inline_to_latex(cell) for cell in row) + r" \\")
        if index == 0:
            output.append(r"\midrule")
    output.extend([r"\bottomrule", r"\end{tabular}"])
    if number == "II":
        output.append(r"\end{adjustbox}")
    return output


def markdown_to_latex(markdown: str) -> str:
    """Fallback exporter for the deliberately constrained manuscript syntax."""
    lines = markdown.splitlines()
    output: list[str] = []
    in_raw_latex = False
    in_itemize = False
    pending_table: str | None = None
    index = 0

    def close_itemize() -> None:
        nonlocal in_itemize
        if in_itemize:
            output.append(r"\end{itemize}")
            in_itemize = False

    while index < len(lines):
        line = lines[index]
        if line == "```{=latex}":
            close_itemize()
            in_raw_latex = True
            index += 1
            continue
        if line == "```" and in_raw_latex:
            in_raw_latex = False
            index += 1
            continue
        if in_raw_latex:
            output.append(line)
            index += 1
            continue

        if line.startswith("- "):
            if not in_itemize:
                output.append(r"\begin{itemize}")
                in_itemize = True
            output.append(r"\item " + inline_to_latex(line[2:]))
            index += 1
            continue
        close_itemize()

        table_caption = re.fullmatch(r"\*\*Table ([IVX]+)\. (.+)\*\*", line)
        if table_caption:
            pending_table = table_caption.group(1)
            output.extend(
                [
                    r"\begin{table}[ht]",
                    r"\centering",
                    r"\caption{" + inline_to_latex(table_caption.group(2)) + "}",
                ]
            )
            index += 1
            continue

        if line.startswith("|"):
            if pending_table is None:
                raise ValueError("Markdown table found without a preceding numbered caption")
            table_lines: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.extend(table_to_latex(table_lines, pending_table))
            output.append(r"\end{table}")
            pending_table = None
            continue

        image_match = re.fullmatch(r"!\[(.+)\]\((.+)\)\{width=(\d+)%\}", line)
        if image_match:
            width = int(image_match.group(3)) / 100
            output.extend(
                [
                    r"\begin{figure}[ht]",
                    r"\centering",
                    rf"\includegraphics[width={width:.2f}\linewidth,keepaspectratio]{{{image_match.group(2)}}}",
                    r"\caption{" + inline_to_latex(image_match.group(1)) + "}",
                    r"\end{figure}",
                ]
            )
            index += 1
            continue

        heading_match = re.fullmatch(r"(#{2,3}) (.+)", line)
        if heading_match:
            command = "section" if len(heading_match.group(1)) == 2 else "subsection"
            output.append(rf"\{command}{{{inline_to_latex(heading_match.group(2))}}}")
            index += 1
            continue

        output.append(inline_to_latex(line) if line else "")
        index += 1

    close_itemize()
    if in_raw_latex or pending_table is not None:
        raise ValueError("Unclosed raw-LaTeX block or Markdown table")
    return "\n".join(output).rstrip() + "\n"


def main() -> None:
    markdown_source = prepare_markdown(SOURCE.read_text(encoding="utf-8"))
    validate_citation_graph(markdown_source)
    pandoc = shutil.which("pandoc")
    if pandoc:
        result = subprocess.run(
            [
                pandoc,
                "--from=markdown+raw_tex",
                "--to=latex",
                "--top-level-division=section",
                "--shift-heading-level-by=-1",
            ],
            input=markdown_source,
            text=True,
            capture_output=True,
            check=True,
        )
        latex = re.sub(r",alt=\{[^{}]*\}", "", result.stdout)
    else:
        latex = markdown_to_latex(markdown_source)
    discussion_heading = r"\section{Discussion}"
    if latex.count(discussion_heading) != 1:
        raise ValueError("Expected exactly one Discussion section in the LaTeX export")
    latex = latex.replace(
        discussion_heading,
        r"\clearpage" + "\n\n" + discussion_heading,
        1,
    )
    OUTPUT.write_text(latex, encoding="utf-8", newline="\n")
    print(f"{OUTPUT} ({'pandoc' if pandoc else 'built-in fallback'})")


if __name__ == "__main__":
    main()
