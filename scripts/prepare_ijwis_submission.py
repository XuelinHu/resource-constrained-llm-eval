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
    ("system_architecture.pdf", "Four-layer bounded bilingual agent workflow linking three role-access clusters to Web applications, governed retrieval, local generation and single-workstation infrastructure. Solid arrows denote runtime or data flow; dashed arrows denote governance or control. Source: Authors' own work."),
    ("knowledge_governance_lifecycle.pdf", "Four-stage expert-governed lifecycle covering source acquisition and bilingual editing, review-state decisions, approved production indexing, and evaluation splits with exact held-out records excluded from indexing and training. Source: Authors' own work."),
    ("top_k_quality_latency.pdf", "Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work."),
    ("training_validation_loss.pdf", "Completion-only QLoRA training and validation loss for Qwen2.5-7B and GLM-4-9B. Source: Authors' own work."),
    ("translation_before_after.pdf", "Direction- and task-separated COMET before and after QLoRA. Left: Qwen2.5-7B; right: GLM-4-9B. Source: Authors' own work."),
    ("quality_latency_pareto.pdf", "Bilingual QA quality against generation latency and peak GPU memory. Left: mean generation latency; right: peak reserved GPU memory. Source: Authors' own work."),
    ("error_type_distribution.pdf", "Mean prevalence of automatically flagged output errors across the evaluated generator and retrieval conditions. Source: Authors' own work."),
    ("supplementary_system_validation.pdf", "Bilingual index, automated evidence support and governance-history validation. Panels A-C report field ablation, evidence support and governance audit results, respectively. Source: Authors' own work."),
]

TABLE_CAPTION = re.compile(r"^\*\*Table ([IVX]+)\. (.+)\*\*$")


def write_utf8_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


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


def main(skip_docx: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_output = OUTPUT / "figures"
    figure_output.mkdir(exist_ok=True)

    manuscript, tables = split_tables(SOURCE.read_text(encoding="utf-8"))
    manuscript_path = OUTPUT / "anonymous_manuscript.md"
    tables_path = OUTPUT / "tables.md"
    captions_path = OUTPUT / "figure_captions.md"
    title_page_path = OUTPUT / "title_page_template.md"

    write_utf8_lf(manuscript_path, manuscript)
    write_utf8_lf(tables_path, tables)
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
        "**Authors in publication order:** Xiaoqin Fu, Youjing Fu, Xuelin Hu\n\n"
        "## Author 1: Xiaoqin Fu\n\n"
        "**Affiliation:** School of Railway Locomotive and Rolling Stock, Liuzhou Railway Vocational Technical College, Liuzhou 545000, China\n\n"
        "**ORCID:** 0009-0003-5123-8393\n\n"
        "**Email:** xiaoqin.fu@qq.com; fuxiaoqin@ltzy.edu.cn\n\n"
        "**Biography:** Lecturer; master's degree in English education. Research interests include international vocational education and railway education.\n\n"
        "## Author 2: Youjing Fu\n\n"
        "**Affiliation:** College of Earth and Environmental Sciences, Lanzhou University, Lanzhou 730000, China\n\n"
        "**ORCID:** 0009-0004-1369-7879\n\n"
        "**Email:** fuyj2025@lzu.edu.cn\n\n"
        "## Author 3 and Corresponding Author: Xuelin Hu\n\n"
        "**Affiliation:** School of Railway Communication and Signaling Engineering, Liuzhou Railway Vocational Technical College, No. 2 Wenyuan Road, Yufeng District, Liuzhou City, Guangxi Zhuang Autonomous Region, 545000, China\n\n"
        "**ORCID:** 0000-0002-4475-3034\n\n"
        "**Email:** huxuelinai@gmail.com; huxl@ltzy.edu.cn\n\n"
        "**Biography:** Master's degree in software engineering; senior engineer. Research interests include graph neural networks, trustworthy software testing and development, computer vision and multimodal foundation models.\n\n"
        "**Corresponding author:** Xuelin Hu, huxl@ltzy.edu.cn; huxuelinai@gmail.com\n\n"
        "**Funding:** This research received no external funding.\n\n"
        "**Author contributions (CRediT):** Xiaoqin Fu: Conceptualization, Methodology, Data curation, Investigation, Validation, Writing - original draft, Writing - review and editing. Youjing Fu: Data curation, Investigation, Formal analysis, Validation, Writing - review and editing. Xuelin Hu: Conceptualization, Methodology, Software, Data curation, Formal analysis, Validation, Resources, Supervision, Project administration, Writing - review and editing. All three authors jointly constructed the training datasets and performed data checking and quality control.\n\n"
        "**Acknowledgements:** The authors have no acknowledgements to declare.\n",
    )

    for index, (filename, _) in enumerate(FIGURES, start=1):
        shutil.copy2(ROOT / "paper/ijwis/figures" / filename, figure_output / f"Figure_{index}.pdf")

    for index, filename in enumerate(("system_architecture.drawio", "knowledge_governance_lifecycle.drawio"), start=1):
        shutil.copy2(ROOT / "paper/ijwis/figures" / filename, figure_output / f"Figure_{index}.drawio")

    if not skip_docx:
        for source, target in (
            (manuscript_path, OUTPUT / "anonymous_manuscript.docx"),
            (tables_path, OUTPUT / "tables.docx"),
            (captions_path, OUTPUT / "figure_captions.docx"),
            (title_page_path, OUTPUT / "title_page_template.docx"),
        ):
            run_pandoc(source, target)

    preview = ROOT / "output/pdf/ijwis_manuscript_anonymous.pdf"
    if preview.exists():
        shutil.copy2(preview, OUTPUT / "anonymous_manuscript_preview.pdf")

    print(OUTPUT)


if __name__ == "__main__":
    main(skip_docx="--skip-docx" in sys.argv[1:])
