"""Combine supplementary IJWIS validation plots into one journal figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper/ijwis/figures"
PANELS = (
    ("figure_08_panel_a_bilingual_index.png", "A", "Bilingual index fields"),
    ("figure_08_panel_b_evidence_support.png", "B", "Automated evidence support"),
    ("figure_08_panel_c_governance_audit.png", "C", "Governance history audit"),
)
OUTPUT_PNG = FIGURE_DIR / "figure_08_system_validation.png"
OUTPUT_PDF = FIGURE_DIR / "figure_08_system_validation.pdf"


def main() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    # Stack all three panels at the same scale for a compact portrait figure.
    fig = plt.figure(figsize=(8.0, 12.0), constrained_layout=True)
    grid = GridSpec(3, 1, figure=fig, height_ratios=(1, 1, 1))
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    ]
    for axis, (filename, label, title) in zip(axes, PANELS, strict=True):
        axis.imshow(mpimg.imread(FIGURE_DIR / filename))
        axis.set_axis_off()
        axis.text(
            0.01,
            0.99,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )
        axis.set_title(title, fontsize=11, pad=4)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
