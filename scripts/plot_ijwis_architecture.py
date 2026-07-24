"""Render the paper's system architecture and knowledge-governance workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/ijwis/figures"


def box(ax, x: float, y: float, width: float, height: float, title: str, detail: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.5))
    ax.text(x + width / 2, y + height * 0.65, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", fontsize=8, color="#333333")


def arrow(ax, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#4D4D4D"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18, label, ha="center", fontsize=8)


def main() -> None:
    rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    blue, green, orange = "#0072B2", "#009E73", "#D55E00"

    box(ax, 0.3, 3.7, 2.0, 1.1, "Knowledge sources", "Terminology | regulations\ntextbooks | OCR pages", blue)
    box(ax, 3.0, 3.7, 2.0, 1.1, "Bilingual processing", "Chinese-English fields\nsource metadata", blue)
    box(ax, 5.7, 3.7, 2.0, 1.1, "Expert governance", "approved | revision\nrejected | history", orange)
    box(ax, 8.4, 3.7, 2.0, 1.1, "PostgreSQL", "governed records\nheld-out split controls", green)

    box(ax, 1.3, 1.1, 2.1, 1.1, "BM25 index", "lexical retrieval", blue)
    box(ax, 4.0, 1.1, 2.1, 1.1, "pgvector index", "BGE-M3 embeddings", green)
    box(ax, 6.8, 1.1, 2.1, 1.1, "Hybrid RRF", "approved-only filter\nranked evidence", orange)
    box(ax, 9.5, 1.1, 2.1, 1.1, "Local generator", "QLoRA / base model\ncited bilingual answer", blue)

    arrow(ax, (2.3, 4.25), (3.0, 4.25))
    arrow(ax, (5.0, 4.25), (5.7, 4.25))
    arrow(ax, (7.7, 4.25), (8.4, 4.25))
    arrow(ax, (9.4, 3.7), (2.35, 2.2), "approved corpus")
    arrow(ax, (9.4, 3.7), (5.05, 2.2))
    arrow(ax, (3.4, 1.65), (6.8, 1.65))
    arrow(ax, (6.1, 1.65), (6.8, 1.65))
    arrow(ax, (8.9, 1.65), (9.5, 1.65))
    ax.text(9.2, 2.36, "top-k evidence", ha="center", fontsize=8)
    ax.text(10.55, 0.55, "Answer + evidence provenance", ha="center", fontsize=9, color="#333333")
    arrow(ax, (10.55, 1.1), (10.55, 0.72))
    ax.text(0.3, 5.35, "Knowledge acquisition and governance", fontsize=11, weight="bold")
    ax.text(0.3, 2.65, "Retrieval and generation", fontsize=11, weight="bold")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT / f"system_architecture.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote={OUTPUT / 'system_architecture.pdf'}")


if __name__ == "__main__":
    main()
