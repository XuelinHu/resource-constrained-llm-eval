"""Export the authoritative IJWIS Markdown draft into the LaTeX submission template."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/ijwis/manuscript.md"
OUTPUT = ROOT / "IJWIS__Copy_/manuscript_body.tex"

FIGURES: dict[str, list[tuple[str, str]]] = {
    "### 3.3 Retrieval and answer generation": [
        ("system_architecture.pdf", "System architecture and knowledge-governance workflow."),
    ],
    "### 4.4 Regulation-only pilot answer generation": [
        ("top_k_quality_latency.pdf", "Approved-hybrid retrieval quality and latency across top-k settings."),
    ],
    "### 4.6 Multi-generator RAG and interaction effects": [
        ("training_validation_loss.pdf", "Completion-only QLoRA training and validation loss."),
    ],
    "### 4.8 Resource use and automated error analysis": [
        ("translation_before_after.pdf", "Direction- and task-separated COMET before and after QLoRA."),
    ],
    "## 5. Discussion": [
        ("quality_latency_pareto.pdf", "Bilingual QA quality against generation latency and peak GPU memory."),
        ("error_type_distribution.pdf", "Mean prevalence of automatically flagged output errors."),
    ],
}


def prepare_markdown(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
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
    return "\n".join(output) + "\n"


def main() -> None:
    markdown_source = prepare_markdown(SOURCE.read_text(encoding="utf-8"))
    result = subprocess.run(
        [
            "pandoc",
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
    OUTPUT.write_text(latex, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
