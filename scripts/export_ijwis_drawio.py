"""Export editable Draw.io architecture diagrams for the IJWIS manuscript."""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, register_namespace, tostring


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/ijwis/figures"
register_namespace("", "http://www.diagrams.net/ns/diagram")


def diagram(name: str) -> tuple[Element, Element]:
    mxfile = Element("mxfile", {"host": "app.diagrams.net", "version": "24.7.17"})
    diagram_node = SubElement(mxfile, "diagram", {"id": name, "name": name})
    model = SubElement(diagram_node, "mxGraphModel", {
        "dx": "1600", "dy": "1000", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1", "math": "0", "shadow": "0",
        "background": "#ffffff", "pageWidth": "1600", "pageHeight": "1000",
    })
    root = SubElement(model, "root")
    SubElement(root, "mxCell", {"id": "0"})
    SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    return mxfile, root


def style_box(stroke: str, fill: str, size: int = 14, bold: bool = True) -> str:
    return ";".join([
        "rounded=0", "whiteSpace=wrap", "html=1", "fontFamily=Arial",
        f"fontSize={size}", f"fontStyle={1 if bold else 0}",
        f"strokeColor={stroke}", f"fillColor={fill}", "spacing=8",
        "align=center", "verticalAlign=middle", "shadow=0",
    ])


def add_box(root: Element, ident: str, label: str, x: int, y: int, w: int, h: int,
            stroke: str, fill: str = "#ffffff", size: int = 14, bold: bool = True,
            parent: str = "1") -> None:
    if isinstance(size, str):
        parent, size = size, 14
    cell = SubElement(root, "mxCell", {
        "id": ident, "value": label, "style": style_box(stroke, fill, size, bold),
        "vertex": "1", "parent": parent,
    })
    SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


def add_layer(root: Element, ident: str, title: str, x: int, y: int, w: int, h: int,
              stroke: str, fill: str) -> None:
    add_box(root, ident, title, x, y, w, h, stroke, fill, 18, True)
    cell = root.find(f"mxCell[@id='{ident}']")
    assert cell is not None
    cell.set("style", ";".join([
        "rounded=0", "whiteSpace=wrap", "html=1", "fontFamily=Arial", "fontSize=18",
        "fontStyle=1", f"strokeColor={stroke}", f"fillColor={fill}", "align=left",
        "verticalAlign=top", "spacingTop=12", "spacingLeft=18", "container=1",
        "collapsible=0", "pointerEvents=0", "connectable=0",
    ]))


def add_edge(root: Element, ident: str, source: str, target: str, color: str = "#4d4d4d",
             dashed: bool = False, label: str = "") -> None:
    style = ";".join([
        "edgeStyle=orthogonalEdgeStyle", "rounded=0", "orthogonalLoop=1", "jettySize=auto",
        "html=1", "endArrow=block", f"strokeColor={color}", "strokeWidth=2",
        f"dashed={1 if dashed else 0}", "fontFamily=Arial", "fontSize=12",
    ])
    cell = SubElement(root, "mxCell", {
        "id": ident, "value": label, "style": style, "edge": "1", "parent": "1",
        "source": source, "target": target,
    })
    SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


def write(mxfile: Element, path: Path) -> None:
    path.write_bytes(tostring(mxfile, encoding="utf-8", xml_declaration=True))


def architecture() -> None:
    mxfile, root = diagram("system_architecture")
    add_layer(root, "layer_roles", "Role-specific access", 20, 20, 1540, 150, "#0072B2", "#EDF5FA")
    add_layer(root, "layer_apps", "Web applications", 20, 200, 1540, 150, "#009E73", "#ECF8F4")
    add_layer(root, "layer_rag", "Online bilingual RAG service", 20, 380, 1540, 330, "#4C7A34", "#F0F6EC")
    add_layer(root, "layer_data", "Governed data and local infrastructure", 20, 740, 1540, 230, "#D55E00", "#FCF1EB")
    boxes = [
        ("learner", "International learner\nChinese or English question", 80, 70, 300, 70, "#0072B2", "#ffffff", "layer_roles"),
        ("teacher", "Railway teacher\nAuthor and reviewer", 620, 70, 300, 70, "#0072B2", "#ffffff", "layer_roles"),
        ("admin", "System administrator\nService and resource control", 1160, 70, 300, 70, "#0072B2", "#ffffff", "layer_roles"),
        ("portal", "Bilingual Q&A portal\nAnswer and source evidence", 80, 245, 300, 70, "#009E73", "#ffffff", "layer_apps"),
        ("review", "Knowledge review console\nEdit, review and version", 620, 245, 300, 70, "#009E73", "#ffffff", "layer_apps"),
        ("ops", "Operations console\nLatency, memory and revisions", 1160, 245, 300, 70, "#009E73", "#ffffff", "layer_apps"),
        ("fastapi", "FastAPI\nOrchestration API", 60, 480, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("router", "Language router\nZH / EN fields", 280, 480, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("bm25", "BM25\nLexical retrieval", 500, 440, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("dense", "Dense search\nBGE-M3 / pgvector", 500, 550, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("rrf", "Hybrid RRF\nRank fusion", 730, 480, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("evidence", "Evidence\nTop-k with provenance", 960, 480, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("llm", "Local LLM\nBase or QLoRA", 1190, 480, 190, 80, "#4C7A34", "#ffffff", "layer_rag"),
        ("response", "Evidence-conditioned response\nAnswer with evidence IDs", 1190, 590, 280, 75, "#009E73", "#ffffff", "layer_rag"),
        ("postgres", "PostgreSQL + pgvector\nRecords, status, history and vectors", 70, 820, 340, 85, "#D55E00", "#ffffff", "layer_data"),
        ("approved_bm25", "Approved BM25 index\nTest records excluded", 470, 820, 280, 85, "#D55E00", "#ffffff", "layer_data"),
        ("embeddings", "BGE-M3 embeddings\nApproved vector index", 810, 820, 300, 85, "#D55E00", "#ffffff", "layer_data"),
        ("runtime", "RTX 3090 runtime\nLocal model and adapter", 1170, 820, 320, 85, "#7A5195", "#ffffff", "layer_data"),
    ]
    for args in boxes: add_box(root, *args)
    edges = [
        ("e1", "learner", "portal", "#0072B2"), ("e2", "teacher", "review", "#0072B2"), ("e3", "admin", "ops", "#0072B2"),
        ("e4", "portal", "fastapi", "#009E73", False, "question"), ("e5", "fastapi", "router"),
        ("e6", "router", "bm25"), ("e7", "router", "dense"), ("e8", "bm25", "rrf"), ("e9", "dense", "rrf"),
        ("e10", "rrf", "evidence"), ("e11", "evidence", "llm"), ("e12", "llm", "response"),
        ("e13", "response", "portal", "#009E73", False, "answer + evidence"),
        ("e14", "postgres", "bm25", "#D55E00", False, "approved text"), ("e15", "embeddings", "dense", "#D55E00", False, "approved vectors"),
        ("e16", "runtime", "llm", "#7A5195", False, "model service"), ("e17", "review", "postgres", "#D55E00", True, "review records"),
        ("e18", "ops", "fastapi", "#0072B2", True, "service control"),
    ]
    for args in edges: add_edge(root, *args)
    write(mxfile, OUTPUT / "system_architecture.drawio")


def governance() -> None:
    mxfile, root = diagram("knowledge_governance_lifecycle")
    add_layer(root, "layer_acquisition", "Knowledge acquisition and expert governance", 20, 20, 1540, 250, "#D55E00", "#FCF1EB")
    add_layer(root, "layer_indexing", "Approved production indexing", 20, 310, 1540, 250, "#4C7A34", "#F0F6EC")
    add_layer(root, "layer_eval", "Leakage-controlled training and evaluation", 20, 600, 1540, 350, "#0072B2", "#EDF5FA")
    boxes = [
        ("sources", "Railway sources\nTextbooks and regulations", 50, 100, 230, 90, "#D55E00", "#ffffff", "layer_acquisition"),
        ("ingestion", "Bilingual ingestion\nChunk, align and add metadata", 330, 100, 250, 90, "#D55E00", "#ffffff", "layer_acquisition"),
        ("records", "Governed records\nPostgreSQL system of record", 630, 100, 270, 90, "#D55E00", "#ffffff", "layer_acquisition"),
        ("gate", "Expert review gate\nApprove / revise / reject", 950, 100, 250, 90, "#D55E00", "#ffffff", "layer_acquisition"),
        ("approved", "Approved knowledge view\nQueryable state; version retained", 1250, 100, 260, 90, "#D55E00", "#ffffff", "layer_acquisition"),
        ("text", "Approved text fields\nChinese, English and provenance", 90, 395, 300, 95, "#4C7A34", "#ffffff", "layer_indexing"),
        ("bm25i", "BM25 index\nLexical production index", 480, 395, 270, 95, "#4C7A34", "#ffffff", "layer_indexing"),
        ("encoder", "BGE-M3 encoder\n1,024-dimensional vectors", 830, 395, 280, 95, "#4C7A34", "#ffffff", "layer_indexing"),
        ("pgvector", "pgvector index\nVectors linked to record IDs", 1190, 395, 300, 95, "#4C7A34", "#ffffff", "layer_indexing"),
        ("split", "Pair-grouped split\nTrain / validation / test", 60, 720, 280, 100, "#0072B2", "#ffffff", "layer_eval"),
        ("train", "Train + validation\nApproved pairs only", 410, 680, 280, 100, "#0072B2", "#ffffff", "layer_eval"),
        ("test", "Held-out test\nNever indexed or trained", 410, 810, 280, 90, "#D55E00", "#ffffff", "layer_eval"),
        ("qlora", "QLoRA training\nCompletion-only adapter", 760, 680, 280, 100, "#7A5195", "#ffffff", "layer_eval"),
        ("adapter", "Adapter\nLocal model revision", 1110, 680, 240, 100, "#7A5195", "#ffffff", "layer_eval"),
        ("evaluation", "Evaluation\nQA, RAG, translation and resources", 1400, 750, 120, 140, "#0072B2", "#ffffff", "layer_eval"),
    ]
    for args in boxes: add_box(root, *args)
    edges = [
        ("g1", "sources", "ingestion"), ("g2", "ingestion", "records"), ("g3", "records", "gate"), ("g4", "gate", "approved"),
        ("g5", "gate", "records", "#D55E00", True, "status + audit event"), ("g6", "gate", "ingestion", "#D55E00", True, "revision request"),
        ("g7", "approved", "text", "#4C7A34", False, "approved-only view"), ("g8", "text", "bm25i"), ("g9", "text", "encoder"), ("g10", "encoder", "pgvector"),
        ("g11", "text", "split", "#0072B2", True, "pair-grouped export"), ("g12", "split", "train"), ("g13", "split", "test"),
        ("g14", "train", "qlora"), ("g15", "qlora", "adapter"), ("g16", "adapter", "evaluation"),
        ("g17", "test", "evaluation", "#0072B2", False, "evaluation only"), ("g18", "pgvector", "evaluation", "#4C7A34", False, "retrieval snapshot"),
    ]
    for args in edges: add_edge(root, *args)
    write(mxfile, OUTPUT / "knowledge_governance_lifecycle.drawio")


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    architecture()
    governance()
    print(f"wrote={OUTPUT / 'system_architecture.drawio'}")
    print(f"wrote={OUTPUT / 'knowledge_governance_lifecycle.drawio'}")
