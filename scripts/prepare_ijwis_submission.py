"""Build an Emerald-style IJWIS submission package from the Markdown draft."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/ijwis/manuscript.md"
OUTPUT = ROOT / "paper/ijwis/submission"

FIGURES = [
    ("figure_01_neural_retrieval.pdf", "Bilingual railway QA workflow from left to right: governed bilingual input, shared encoding, lexical and dense retrieval, fusion, evidence-conditioned generation and evaluation. The rightmost module shows the QLoRA adapter path alongside the frozen base model."),
    ("figure_03_top_k_quality_latency.pdf", "Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work."),
    ("figure_04_training_validation_loss.pdf", "Completion-only QLoRA optimisation for Qwen2.5-7B and GLM-4-9B. Lines show logged training loss; diamonds mark the end-of-epoch validation measurement for each model. Source: Authors' own work."),
    ("figure_06_quality_latency_pareto.pdf", "Mean bilingual standalone character-level F1 against generation latency and peak reserved GPU memory (GiB) for the four Qwen2.5/GLM original and QLoRA conditions. Left: mean generation latency; right: PyTorch reserved GPU memory. Quality and resources are measured on separate workloads. Source: Authors' own work."),
    ("figure_08_system_validation.pdf", "Three validation layers. Panel A compares source-only, Chinese-field, English-field and bilingual indexes; Panel B compares semantic support against retrieved and explicitly cited evidence; Panel C audits immutable review events and before-state snapshots. Together the panels show retrieval balance, evidence support and governance traceability without reducing them to one score. An em dash indicates that a measure is not language-specific."),
]

TABLE_CAPTION = re.compile(r"^\*\*Table ([IVX]+)\. (.+)\*\*$")

def write_utf8_lf(path: Path, content: str) -> None:

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

def split_table_files(tables: str) -> dict[str, str]:
    blocks = re.split(r"(?=^## Table [IVX]+$)", tables, flags=re.MULTILINE)
    return {
        re.search(r"^## Table ([IVX]+)$", block, re.MULTILINE).group(1): block.strip() + "\n"
        for block in blocks
        if re.search(r"^## Table ([IVX]+)$", block, re.MULTILINE)
    }

def split_tables(markdown: str) -> tuple[str, str]:

    lines = markdown.splitlines()
    manuscript: list[str] = []
    tables: list[str] = ["# Tables", "", "Tables are numbered with Roman numerals and supplied separately in accordance with the IJWIS author guidelines.", ""]
    index = 0
    while index < len(lines):
        if lines[index] == "## Declarations":
            while index < len(lines) and lines[index] != "## References":
                index += 1
            continue
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

def main(skip_docx: bool = False) -> None:

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_output = OUTPUT / "figures"
    figure_output.mkdir(exist_ok=True)

    manuscript, tables = split_tables(SOURCE.read_text(encoding="utf-8"))
    manuscript_path = OUTPUT / "anonymous_manuscript.md"
    tables_path = OUTPUT / "tables.md"
    table_output = OUTPUT / "tables"
    table_output.mkdir(exist_ok=True)
    captions_path = OUTPUT / "figure_captions.md"
    title_page_path = OUTPUT / "title_page_template.md"

    write_utf8_lf(manuscript_path, manuscript)
    write_utf8_lf(tables_path, tables)
    roman_to_number = {"I": "1", "II": "2", "III": "3", "IV": "4"}
    for roman, content in split_table_files(tables).items():
        write_utf8_lf(table_output / f"Table_{roman_to_number[roman]}.md", content)
    write_utf8_lf(
        captions_path,
        "# Figure captions\n\n"
        + "\n\n".join(f"**Figure {index}.** {caption}" for index, (_, caption) in enumerate(FIGURES, start=1))
        + "\n",
    )
    write_utf8_lf(
        title_page_path,
        "# Knowledge-Enhanced Large Language Models for Bilingual Railway Vocational Education under Resource Constraints\n\n"
        "**Article type:** Research Paper\n\n"
        "**Authors in publication order:** Xiaoqin Fu, Xuelin Hu, Youjing Fu, Shu Cai, Simeng Li\n\n"
        "## Author 1: Xiaoqin Fu\n\n"
        "**Affiliation:** Liuzhou Railway Vocational Technical College, Liuzhou, China\n\n"
        "**ORCID:** 0009-0003-5123-8393\n\n"
        "**Email:** xiaoqin.fu@qq.com; fuxiaoqin@ltzy.edu.cn\n\n"
        "**Biography:** Lecturer; master's degree in English education. Research interests include international vocational education and railway education.\n\n"
        "## Author 2 and Corresponding Author: Xuelin Hu\n\n"
        "**Affiliation:** Liuzhou Railway Vocational Technical College, Liuzhou, China\n\n"
        "**ORCID:** 0000-0002-4475-3034\n\n"
        "**Email:** huxl@ltzy.edu.cn\n\n"
        "**Biography:** Master's degree in software engineering; senior engineer. Research interests include graph neural networks, trustworthy software testing and development, computer vision and multimodal foundation models.\n\n"
        "## Author 3: Youjing Fu\n\n"
        "**Affiliation:** College of Earth and Environmental Sciences, Lanzhou University, Lanzhou, China\n\n"
        "**ORCID:** 0009-0004-1369-7879\n\n"
        "**Email:** fuyj2025@lzu.edu.cn\n\n"
        "## Author 4: Shu Cai\n\n"
        "**Affiliation:** Liuzhou Railway Vocational Technical College, Liuzhou, China\n\n"
        "**Email:** caishuqqcom@qq.com\n\n"
        "## Author 5: Simeng Li\n\n"
        "**Affiliation:** Liuzhou Railway Vocational Technical College, Liuzhou, China\n\n"
        "**Email:** li.simeng.ai@qq.com\n\n"
        "**Corresponding author:** Xuelin Hu, huxl@ltzy.edu.cn; huxuelinai@gmail.com\n\n"
        "**Funding:** This research received no external funding.\n\n"
        "**Author contributions (CRediT):** Xiaoqin Fu: Conceptualization, Methodology, Data curation, Investigation, Validation, Writing - original draft, Writing - review and editing. Youjing Fu: Data curation, Investigation, Formal analysis, Validation, Writing - review and editing. Xuelin Hu: Conceptualization, Methodology, Software, Data curation, Formal analysis, Validation, Resources, Supervision, Project administration, Writing - review and editing. Shu Cai and Simeng Li: Data curation, Investigation, Validation, Writing - review and editing. All five authors jointly constructed the training datasets and performed data checking and quality control.\n\n"
        "**Acknowledgements:** The authors have no acknowledgements to declare.\n",
    )

    for index, (filename, _) in enumerate(FIGURES, start=1):
        shutil.copy2(ROOT / "paper/ijwis/figures" / filename, figure_output / f"Figure_{index}.pdf")

    tex_output = OUTPUT / "tex"
    tex_output.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "IJWIS/Main.tex", tex_output / "Main.tex")
    shutil.copy2(ROOT / "IJWIS/manuscript_body.tex", tex_output / "manuscript_body.tex")
    shutil.copy2(ROOT / "paper/ijwis/references.bib", tex_output / "references.bib")
    tex_figures = tex_output / "figures"
    tex_figures.mkdir(exist_ok=True)
    for filename, _ in FIGURES:
        shutil.copy2(ROOT / "paper/ijwis/figures" / filename, tex_figures / filename)
    body_path = tex_output / "manuscript_body.tex"
    body = body_path.read_text(encoding="utf-8")
    body = body.replace("../paper/ijwis/figures/", "figures/")
    body_path.write_text(body, encoding="utf-8", newline="\n")
    main_path = tex_output / "Main.tex"
    main = main_path.read_text(encoding="utf-8")
    main = main.replace("\\input{manuscript_body}", "\\input{manuscript_body}")
    main = main.replace("\\bibliography{../paper/ijwis/references}", "\\bibliography{references}")
    main_path.write_text(main, encoding="utf-8", newline="\n")

    if not skip_docx:
        for source, target in (
            (manuscript_path, OUTPUT / "anonymous_manuscript.docx"),
            (tables_path, OUTPUT / "tables.docx"),
            (captions_path, OUTPUT / "figure_captions.docx"),
            (title_page_path, OUTPUT / "title_page_template.docx"),
        ):
            run_pandoc(source, target)

    print(OUTPUT)

if __name__ == "__main__":

    main(skip_docx="--skip-docx" in sys.argv[1:])
