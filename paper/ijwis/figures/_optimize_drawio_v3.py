from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SCALE = 1.10
ANCHOR_X = 20.0


def scaled_x(value: str) -> str:
    """Scale absolute x coordinates while keeping the left stage margin."""
    number = float(value)
    result = ANCHOR_X + (number - ANCHOR_X) * SCALE
    # Keep draw.io XML readable and avoid noisy binary floating point tails.
    if abs(result - round(result)) < 1e-8:
        return str(int(round(result)))
    return f"{result:.2f}".rstrip("0").rstrip(".")


def scaled_width(value: str) -> str:
    number = float(value) * SCALE
    if abs(number - round(number)) < 1e-8:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def is_number(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def optimize(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()

    # The editor page is widened to leave a clean right margin after scaling.
    for graph in root.findall(".//mxGraphModel"):
        graph.set("pageWidth", "2650")

    for cell in root.findall(".//mxCell"):
        cell_id = cell.get("id", "")
        geometry = cell.find("mxGeometry")

        # Scale all visible and hidden vertex boxes so the reference/tracing
        # layer remains approximately aligned with the editable reconstruction.
        # Stage circles keep their original size; only their horizontal position
        # follows the wider panel rhythm.
        if geometry is not None and cell.get("vertex") == "1":
            if is_number(geometry.get("x")):
                geometry.set("x", scaled_x(geometry.get("x", "0")))
            if is_number(geometry.get("width")) and cell_id not in {
                "stage1",
                "stage2",
                "stage3",
                "stage4",
                "stage5",
            }:
                geometry.set("width", scaled_width(geometry.get("width", "0")))

        # Edge geometry x/y values such as .50 are relative label offsets and
        # must not be scaled. Absolute routing points, however, need to follow
        # the widened layout.
        for point in cell.findall(".//mxPoint"):
            if is_number(point.get("x")):
                point.set("x", scaled_x(point.get("x", "0")))

    # A few labels are intentionally concise so they remain legible in the
    # diamond and the compact empirical comparison card.
    replacements = {
        "admissible": (
            "<b><font style=\"font-size:22px;color:#111827\">Admissibility filter</font></b>\n"
            "approved / test excluded"
        ),
        "records": (
            "<b><font style=\"font-size:22px;color:#111827\">Governed bilingual records</font></b>\n"
            "Chinese/English · source review · split"
        ),
        "topk": (
            "<b><font style=\"font-size:22px;color:#111827\">Top-k evidence passages</font></b>\n"
            "[E1]/[E2] passage + source"
        ),
        "trade_cite": "Citation compliance\ndown / task dependent",
    }
    for cell_id, value in replacements.items():
        cell = next((item for item in root.findall(".//mxCell") if item.get("id") == cell_id), None)
        if cell is not None:
            cell.set("value", value)

    cells = {cell.get("id"): cell for cell in root.findall(".//mxCell")}

    def set_geometry(cell_id: str, **attrs: str) -> None:
        cell = cells.get(cell_id)
        if cell is None:
            return
        geometry = cell.find("mxGeometry")
        if geometry is None:
            return
        for key, value in attrs.items():
            geometry.set(key, value)

    # Give the top-row panels and the few bottom-aligned labels a little
    # vertical breathing room as well.  This removes the last one-pixel
    # fallback-raster clipping reported by the export audit.
    for panel_id in ("panel1", "panel2", "panel3", "panel4"):
        set_geometry(panel_id, height="610")
    set_geometry("embedding_dim", y="692", height="38")
    set_geometry("query_en", y="337")
    set_geometry("or_label", y="315", height="22")
    set_geometry("model_choice", height="80")
    set_geometry("lora_inset", height="365")
    set_geometry("lora_formula", y="620", height="45")
    set_geometry("metric_resource", height="65")

    # Concise explanatory labels are easier to read at the requested subtitle
    # size and avoid a long italic line crowding the panel border.
    note = cells.get("p1_note")
    if note is not None:
        note.set("value", "Query-time approval only; not a trainable mask.")
        note.set("style", note.get("style", "").replace("fontSize=16", "fontSize=19"))

    formula = cells.get("lora_formula")
    if formula is not None:
        formula.set("style", formula.get("style", "").replace("fontSize=14", "fontSize=16"))

    # Keep the explicitly requested typography invariant for arrow labels.
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") != "1" or not cell.get("value"):
            continue
        style = cell.get("style", "")
        parts = [part for part in style.split(";") if part]
        replacements_by_key = {
            "fontSize": "14",
            "fontColor": "#111827",
            "labelBackgroundColor": "none",
            "labelBorderColor": "none",
        }
        seen: set[str] = set()
        normalized: list[str] = []
        for part in parts:
            if "=" not in part:
                normalized.append(part)
                continue
            key, _ = part.split("=", 1)
            if key in replacements_by_key:
                normalized.append(f"{key}={replacements_by_key[key]}")
                seen.add(key)
            else:
                normalized.append(part)
        for key, value in replacements_by_key.items():
            if key not in seen:
                normalized.append(f"{key}={value}")
        cell.set("style", ";".join(normalized) + ";")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: _optimize_drawio_v3.py INPUT OUTPUT")
    optimize(Path(sys.argv[1]), Path(sys.argv[2]))
