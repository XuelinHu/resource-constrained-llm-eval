"""Combine supplementary IJWIS validation plots into one journal figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper/ijwis/figures"
PANELS = (
    ("bilingual_index_ablation.png", "A", "Bilingual index fields"),
    ("rag_faithfulness.png", "B", "Automated evidence support"),
    ("governance_history_audit.png", "C", "Governance history audit"),
    ("web_load_test.png", "D", "Steady-state retrieval load"),
)
OUTPUT_PNG = FIGURE_DIR / "supplementary_system_validation.png"
OUTPUT_PDF = FIGURE_DIR / "supplementary_system_validation.pdf"


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (filename, label, title) in zip(axes.flat, PANELS, strict=True):
        axis.imshow(mpimg.imread(FIGURE_DIR / filename))
        axis.set_axis_off()
        axis.text(
            0.01,
            0.99,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )
        axis.set_title(title, fontsize=11, pad=4)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
