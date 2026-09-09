"""Export only the editable v3 model layer for the IJWIS Figure 1 artwork."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/ijwis/figures/figure_01_neural_retrieval.drawio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", help="X display for Draw.io on Linux, e.g. :1")
    args = parser.parse_args()
    drawio = shutil.which("drawio")
    poppler = shutil.which("pdftoppm")
    if not drawio or not poppler:
        parser.error("drawio and pdftoppm must be on PATH")

    # The reference layer is not reliably marked hidden in the editable source.
    # Select the vector layer explicitly; never export the reference or notes.
    tree = ET.parse(SOURCE)
    layers = [cell for cell in tree.iter("mxCell") if cell.get("parent") == "0"]
    layer_index = next(i for i, cell in enumerate(layers) if cell.get("id") == "vector_layer")
    env = os.environ.copy()
    if args.display:
        env["DISPLAY"] = args.display
    pdf = SOURCE.with_suffix(".pdf")
    subprocess.run([
        drawio, "--disable-gpu", "--export", "--format", "pdf", "--crop",
        "--border", "8", "--layers", str(layer_index), "--theme", "light",
        "--output", str(pdf), str(SOURCE),
    ], env=env, check=True)

    # Raster companion at 600 dpi for a 257 mm-wide landscape placement.
    # The PDF remains the primary, resolution-independent manuscript artwork.
    width = round(257 / 25.4 * 600)
    scratch = ROOT / "tmp/pdfs"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neural-v3-", dir=scratch) as temp:
        prefix = Path(temp) / "figure"
        subprocess.run([
            poppler, "-singlefile", "-scale-to-x", str(width), "-scale-to-y", "-1",
            "-png", str(pdf), str(prefix),
        ], check=True)
        with Image.open(prefix.with_suffix(".png")) as image:
            image.convert("RGB").save(SOURCE.with_suffix(".png"), dpi=(600, 600))
    print(f"Exported {pdf} and its 600-dpi PNG companion")


if __name__ == "__main__":
    main()
