"""Build an Emerald-style IJWIS submission package from the Markdown draft."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/ijwis/manuscript.md"
OUTPUT = ROOT / "paper/ijwis/submission"

FIGURES = [
    ("system_architecture.pdf", "Icon-enhanced role-specific Web and online bilingual RAG architecture. Solid arrows denote runtime or data flow; dashed arrows denote governance or control. Source: Authors' own work."),
    ("knowledge_governance_lifecycle.pdf", "Expert-governed knowledge, production-index and leakage-controlled evaluation lifecycle. Source: Authors' own work."),
    ("top_k_quality_latency.pdf", "Approved-hybrid retrieval quality and latency across top-k settings. Source: Authors' own work."),
    ("training_validation_loss.pdf", "Completion-only QLoRA training and validation loss. Source: Authors' own work."),
    ("translation_before_after.pdf", "Direction- and task-separated COMET before and after QLoRA. Source: Authors' own work."),
    ("quality_latency_pareto.pdf", "Bilingual QA quality against generation latency and peak GPU memory. Source: Authors' own work."),
    ("error_type_distribution.pdf", "Mean prevalence of automatically flagged output errors. Source: Authors' own work."),
    ("supplementary_system_validation.pdf", "Bilingual index, evidence support, governance history and steady-state retrieval load validation. Source: Authors' own work."),
]

TABLE_CAPTION = re.compile(r"^\*\*Table ([IVX]+)\. (.+)\*\*$")


def split_tables(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    manuscript: list[str] = []
    tables: list[str] = ["# Tables", "", "Tables are numbered with Roman numerals and supplied separately in accordance with the IJWIS author guidelines.", ""]
    index = 0
    while index < len(lines):
        match = TABLE_CAPTION.match(lines[index])
        if not match:
            manuscript.append(lines[index])
            index += 1
            continue

        number, title = match.groups()
        tables.extend([f"## Table {number}", "", title, ""])
        manuscript.extend([f"**[Insert Table {number} here]**", ""])
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        while index < len(lines) and lines[index].startswith("|"):
            tables.append(lines[index])
            index += 1
        tables.append("")
    return "\n".join(manuscript).rstrip() + "\n", "\n".join(tables).rstrip() + "\n"


def run_pandoc(source: Path, target: Path) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to build the IJWIS Word files")
    subprocess.run(
        [pandoc, "--from=markdown", "--to=docx", "--standalone", str(source), "--output", str(target)],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_output = OUTPUT / "figures"
    figure_output.mkdir(exist_ok=True)

    manuscript, tables = split_tables(SOURCE.read_text(encoding="utf-8"))
    manuscript_path = OUTPUT / "anonymous_manuscript.md"
    tables_path = OUTPUT / "tables.md"
    captions_path = OUTPUT / "figure_captions.md"
    title_page_path = OUTPUT / "title_page_template.md"

    manuscript_path.write_text(manuscript, encoding="utf-8")
    tables_path.write_text(tables, encoding="utf-8")
    captions_path.write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"**Figure {index}.** {caption}" for index, (_, caption) in enumerate(FIGURES, start=1))
        + "\n",
        encoding="utf-8",
    )
    title_page_path.write_text(
        "# A Knowledge-Enhanced Large Language Model Web Information System for Bilingual Railway Vocational Education\n\n"
        "**Article type:** Research Paper\n\n"
        "**Authors in publication order:** [complete before submission]\n\n"
        "**Affiliations:** [complete before submission]\n\n"
        "**Corresponding author:** [name, institutional email and postal address]\n\n"
        "**ORCID identifiers:** [complete for each author]\n\n"
        "**Funding:** [funding body, grant number and funder's role, or confirmed no external funding statement]\n\n"
        "**Author contributions:** [complete using the CRediT taxonomy]\n\n"
        "**Acknowledgements:** [include non-author contributors only after consent]\n",
        encoding="utf-8",
    )

    for index, (filename, _) in enumerate(FIGURES, start=1):
        shutil.copy2(ROOT / "paper/ijwis/figures" / filename, figure_output / f"Figure_{index}.pdf")

    for source, target in (
        (manuscript_path, OUTPUT / "anonymous_manuscript.docx"),
        (tables_path, OUTPUT / "tables.docx"),
        (captions_path, OUTPUT / "figure_captions.docx"),
        (title_page_path, OUTPUT / "title_page_template.docx"),
    ):
        run_pandoc(source, target)

    preview = ROOT / "output/pdf/ijwis_bilingual_railway_manuscript_latex.pdf"
    if preview.exists():
        shutil.copy2(preview, OUTPUT / "anonymous_manuscript_preview.pdf")

    print(OUTPUT)


if __name__ == "__main__":
    main()
