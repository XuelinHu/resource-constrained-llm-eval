"""Combine supplementary IJWIS validation plots into one journal figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image, ImageChops


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
    # Stack the panels with a compact vertical layout.  The source panels are
    # wide plots; using a shorter canvas avoids the large blank bands that
    # appeared when three equal-height rows were placed on a full page.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 9.1),
        gridspec_kw={"hspace": 0.04},
    )
    for axis, (filename, label, title) in zip(axes, PANELS, strict=True):
        image = Image.open(FIGURE_DIR / filename).convert("RGB")
        background = Image.new("RGB", image.size, "white")
        bbox = ImageChops.difference(image, background).getbbox()
        if bbox:
            pad = 10
            bbox = (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(image.width, bbox[2] + pad),
                min(image.height, bbox[3] + pad),
            )
            image = image.crop(bbox)
        axis.imshow(image, aspect="auto")
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
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
