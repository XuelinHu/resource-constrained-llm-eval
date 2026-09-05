"""Render icon-enhanced vector architecture figures for the IJWIS paper."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/ijwis/figures"

BLUE = "#0072B2"
TEAL = "#009E73"
GREEN = "#4C7A34"
ORANGE = "#D55E00"
PURPLE = "#7A5195"
GREY = "#4D4D4D"


def configure() -> None:
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def layer(ax, y: float, height: float, title: str, color: str, fill: str) -> None:
    ax.add_patch(Rectangle((0.15, y), 15.7, height, facecolor=fill, edgecolor=color, linewidth=1.2, zorder=0))
    ax.text(0.35, y + height - 0.15, title, color=color, fontsize=13.2, weight="bold", va="top", zorder=8)


def _line(ax, xs: list[float], ys: list[float], color: str, lw: float = 1.55) -> None:
    ax.add_line(Line2D(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=7))


def draw_icon(ax, kind: str, x: float, y: float, size: float, color: str) -> None:
    """Draw compact line icons without raster assets or trademarked artwork."""
    l, r, b, t = x - size / 2, x + size / 2, y - size / 2, y + size / 2
    lw = 1.55
    if kind in {"user", "teacher", "admin"}:
        ax.add_patch(Circle((x, y + size * .18), size * .15, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        ax.add_patch(Arc((x, y - size * .18), size * .62, size * .44, theta1=15, theta2=165, edgecolor=color, linewidth=lw, zorder=7))
        if kind == "teacher":
            ax.add_patch(Polygon([(x - size*.27, y + size*.31), (x, y + size*.43), (x + size*.27, y + size*.31), (x, y + size*.20)], closed=True, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        elif kind == "admin":
            ax.add_patch(Circle((x + size*.26, y - size*.15), size*.11, fill=False, edgecolor=color, linewidth=lw, zorder=7))
            for angle in range(0, 360, 90):
                a = math.radians(angle)
                _line(ax, [x+size*.26+size*.12*math.cos(a), x+size*.26+size*.20*math.cos(a)], [y-size*.15+size*.12*math.sin(a), y-size*.15+size*.20*math.sin(a)], color)
        else:
            ax.add_patch(Circle((x + size*.25, y + size*.08), size*.16, fill=False, edgecolor=color, linewidth=lw, zorder=7))
            _line(ax, [x+size*.09, x+size*.41], [y+size*.08, y+size*.08], color)
            _line(ax, [x+size*.25, x+size*.25], [y-size*.08, y+size*.24], color)
    elif kind == "chat":
        ax.add_patch(Rectangle((l+size*.10, b+size*.20), size*.72, size*.50, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [l+size*.25, l+size*.16, l+size*.35], [b+size*.20, b+size*.03, b+size*.20], color)
        for offset in (.31, .47, .63):
            ax.add_patch(Circle((l+size*offset, y+size*.02), size*.025, color=color, zorder=7))
    elif kind in {"document", "review", "response"}:
        ax.add_patch(Rectangle((l+size*.18, b+size*.09), size*.58, size*.76, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [l+size*.29, l+size*.39, l+size*.56], [b+size*.47, b+size*.36, b+size*.60], color)
        _line(ax, [l+size*.30, l+size*.65], [b+size*.25, b+size*.25], color)
    elif kind in {"monitor", "chart"}:
        ax.add_patch(Rectangle((l+size*.10, b+size*.15), size*.78, size*.66, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [l+size*.20, l+size*.38, l+size*.53, l+size*.76], [b+size*.33, b+size*.52, b+size*.43, b+size*.68], color)
    elif kind == "api":
        _line(ax, [l+size*.31, l+size*.12, l+size*.31], [t-size*.12, y, b+size*.12], color)
        _line(ax, [r-size*.31, r-size*.12, r-size*.31], [t-size*.12, y, b+size*.12], color)
        _line(ax, [x-size*.08, x+size*.08], [b+size*.10, t-size*.10], color)
    elif kind == "language":
        ax.text(x, y+size*.14, "ZH", ha="center", va="center", fontsize=9, color=color, weight="bold", zorder=7)
        _line(ax, [l+size*.16, r-size*.16], [y, y], color, 1.1)
        ax.text(x, y-size*.23, "EN", ha="center", va="center", fontsize=9, color=color, weight="bold", zorder=7)
    elif kind == "search":
        ax.add_patch(Circle((x-size*.08, y+size*.07), size*.23, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [x+size*.09, x+size*.34], [y-size*.10, y-size*.35], color)
    elif kind in {"network", "merge"}:
        pts = [(l+size*.20, y+size*.22), (l+size*.20, y-size*.22), (r-size*.20, y)]
        for p in pts[:2]:
            _line(ax, [p[0], pts[2][0]], [p[1], pts[2][1]], color)
        for px, py in pts:
            ax.add_patch(Circle((px, py), size*.065, facecolor="white", edgecolor=color, linewidth=lw, zorder=7))
    elif kind in {"cpu", "adapter"}:
        ax.add_patch(Rectangle((l+size*.20, b+size*.20), size*.60, size*.60, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        ax.add_patch(Rectangle((l+size*.36, b+size*.36), size*.28, size*.28, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        for offset in (.30, .50, .70):
            _line(ax, [l+size*offset, l+size*offset], [b+size*.08, b+size*.20], color)
            _line(ax, [l+size*offset, l+size*offset], [t-size*.20, t-size*.08], color)
    elif kind in {"database", "index"}:
        ax.add_patch(Rectangle((l+size*.17, b+size*.25), size*.66, size*.45, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        ax.add_patch(Arc((x, t-size*.30), size*.66, size*.25, theta1=0, theta2=360, edgecolor=color, linewidth=lw, zorder=7))
        ax.add_patch(Arc((x, b+size*.25), size*.66, size*.25, theta1=180, theta2=360, edgecolor=color, linewidth=lw, zorder=7))
    elif kind == "book":
        _line(ax, [x, x, l+size*.10, l+size*.10, x], [b+size*.12, t-size*.12, t-size*.18, b+size*.22, b+size*.12], color)
        _line(ax, [x, r-size*.10, r-size*.10, x], [t-size*.12, t-size*.18, b+size*.22, b+size*.12], color)
    elif kind == "ingest":
        ax.add_patch(Rectangle((l+size*.12, b+size*.10), size*.50, size*.76, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [l+size*.48, r-size*.08], [y, y], color)
        _line(ax, [r-size*.22, r-size*.08, r-size*.22], [y+size*.14, y, y-size*.14], color)
    elif kind == "shield":
        pts = [(x, t-size*.08), (r-size*.14, t-size*.24), (r-size*.20, b+size*.30), (x, b+size*.08), (l+size*.20, b+size*.30), (l+size*.14, t-size*.24)]
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        _line(ax, [x-size*.16, x-size*.03, x+size*.20], [y, y-size*.13, y+size*.16], color)
    elif kind == "lock":
        ax.add_patch(Rectangle((l+size*.20, b+size*.12), size*.60, size*.48, fill=False, edgecolor=color, linewidth=lw, zorder=7))
        ax.add_patch(Arc((x, y+size*.12), size*.42, size*.50, theta1=0, theta2=180, edgecolor=color, linewidth=lw, zorder=7))
    else:
        ax.add_patch(Circle((x, y), size*.28, fill=False, edgecolor=color, linewidth=lw, zorder=7))


def node(ax, x: float, y: float, width: float, height: float, title: str, detail: str, color: str, symbol: str) -> None:
    box = Rectangle((x, y), width, height, facecolor="white", edgecolor=color, linewidth=1.35, zorder=3)
    ax.add_patch(box)
    draw_icon(ax, symbol, x + .43, y + height/2, min(height*.67, .60), color)
    tx = x + .83
    text_width = max(width - .94, .8)
    wrap_width = max(11, int(text_width * 12.5))
    title = "\n".join(textwrap.fill(part, width=wrap_width, break_long_words=False) for part in title.split("\n"))
    detail = "\n".join(textwrap.fill(part, width=wrap_width, break_long_words=False) for part in detail.split("\n"))
    title_longest = max(len(part) for part in title.split("\n"))
    detail_longest = max(len(part) for part in detail.split("\n"))
    title_size = min(12.2, max(8.4, text_width * 72 / max(title_longest, 1)))
    detail_size = min(9.8, max(7.5, text_width * 62 / max(detail_longest, 1)))
    title_artist = ax.text(tx, y + height - .16, title, ha="left", va="top", fontsize=title_size, weight="bold", linespacing=.95, zorder=8)
    detail_artist = ax.text(tx, y + .14, detail, ha="left", va="bottom", fontsize=detail_size, color=GREY, linespacing=.95, zorder=8)
    title_artist.set_clip_path(box)
    detail_artist.set_clip_path(box)


def arrow(ax, points: list[tuple[float, float]], label: str = "", color: str = GREY, dashed: bool = False, label_xy: tuple[float, float] | None = None) -> None:
    for start, end in zip(points, points[1:]):
        if not (math.isclose(start[0], end[0]) or math.isclose(start[1], end[1])):
            raise ValueError(f"Non-orthogonal arrow segment: {start} -> {end}")
    for i, (start, end) in enumerate(zip(points, points[1:])):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>" if i == len(points)-2 else "-", mutation_scale=13, linewidth=1.3, linestyle="--" if dashed else "-", color=color, shrinkA=0, shrinkB=0, zorder=2))
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=9.6, color=color, bbox={"facecolor":"white", "edgecolor":"none", "pad":1}, zorder=9)


def save(fig, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.7))
    ax.set(xlim=(0,16), ylim=(0,10.8)); ax.axis("off")
    layer(ax, 9.00, 1.55, "Role-specific access", BLUE, "#EDF5FA")
    layer(ax, 7.00, 1.55, "Web applications", TEAL, "#ECF8F4")
    layer(ax, 3.05, 3.30, "Bounded bilingual AI-agent workflow", GREEN, "#F0F6EC")
    layer(ax, .25, 2.45, "Governed data and reproducible local infrastructure", ORANGE, "#FCF1EB")

    node(ax,.75,9.18,3.45,.90,"Learners + practitioners","Bilingual questions and evidence",BLUE,"user")
    node(ax,6.28,9.18,3.45,.90,"Teachers + trainers + consultants","Author, review and advise",BLUE,"teacher")
    node(ax,11.80,9.18,3.45,.90,"Managers + administrators","Governance and resource control",BLUE,"admin")
    node(ax,.75,7.18,3.45,.90,"Bilingual Q&A portal","Answer and source evidence",TEAL,"chat")
    node(ax,6.28,7.18,3.45,.90,"Knowledge review console","Edit, review and version",TEAL,"review")
    node(ax,11.80,7.18,3.45,.90,"Operations console","Latency, memory and revisions",TEAL,"monitor")
    for x in (2.48,8.00,13.52): arrow(ax,[(x,9.18),(x,8.08)],color=BLUE)

    node(ax,.35,4.45,2.05,1.00,"Bounded policy","FastAPI orchestration",GREEN,"api")
    node(ax,2.65,4.45,2.10,1.00,"Language router","ZH / EN fields",GREEN,"language")
    node(ax,5.00,4.95,2.10,1.00,"BM25","Lexical retrieval",GREEN,"search")
    node(ax,5.00,3.65,2.10,1.00,"Dense search","BGE-M3 / pgvector",GREEN,"network")
    node(ax,7.40,4.45,2.10,1.00,"Hybrid RRF","Rank fusion",GREEN,"merge")
    node(ax,9.80,4.45,2.10,1.00,"Evidence","Top-k with provenance",GREEN,"document")
    node(ax,12.20,4.45,2.10,1.00,"Local generator","Base LLM or QLoRA",GREEN,"cpu")
    node(ax,12.20,3.25,3.10,.90,"Evidence-conditioned response","Answer with evidence IDs",TEAL,"response")

    arrow(ax,[(2.48,7.18),(2.48,6.82),(4.68,6.82),(4.68,5.65),(1.38,5.65),(1.38,5.45)],"question",TEAL,label_xy=(3.55,6.82))
    arrow(ax,[(2.40,4.95),(2.65,4.95)])
    arrow(ax,[(4.75,4.95),(4.88,4.95),(4.88,5.45),(5.00,5.45)])
    arrow(ax,[(4.75,4.95),(4.88,4.95),(4.88,4.15),(5.00,4.15)])
    arrow(ax,[(7.10,5.45),(7.25,5.45),(7.25,4.95),(7.40,4.95)])
    arrow(ax,[(7.10,4.15),(7.25,4.15),(7.25,4.95),(7.40,4.95)])
    arrow(ax,[(9.50,4.95),(9.80,4.95)])
    arrow(ax,[(11.90,4.95),(12.20,4.95)])
    arrow(ax,[(13.25,4.45),(13.25,4.15)])
    arrow(ax,[(12.20,3.70),(4.92,3.70),(4.92,7.63),(4.20,7.63)],"answer + evidence",TEAL,label_xy=(8.35,3.70))

    node(ax,.65,.80,3.80,1.10,"PostgreSQL + pgvector","Records, status, history and vectors",ORANGE,"database")
    node(ax,4.90,.80,3.05,1.10,"Approved BM25 index","Exact test records excluded",ORANGE,"index")
    node(ax,8.40,.80,3.10,1.10,"BGE-M3 embeddings","Approved vector index",ORANGE,"network")
    node(ax,11.95,.80,3.30,1.10,"RTX 3090 runtime","Local model and adapter",PURPLE,"cpu")
    arrow(ax,[(6.42,1.90),(6.42,4.95)],"approved text",ORANGE,label_xy=(6.95,2.95))
    arrow(ax,[(9.95,1.90),(9.95,2.82),(6.05,2.82),(6.05,3.65)],"approved vectors",ORANGE,label_xy=(8.05,2.82))
    arrow(ax,[(13.60,1.90),(13.60,4.45)],"model service",PURPLE,label_xy=(14.12,2.88))
    arrow(ax,[(8.00,7.18),(8.00,6.62),(.08,6.62),(.08,1.35),(.65,1.35)],"review records",ORANGE,True,(5.80,6.62))
    arrow(ax,[(13.52,7.18),(13.52,6.42),(.25,6.42),(.25,4.95),(.35,4.95)],"service control",BLUE,True,(10.65,6.42))
    ax.text(15.70,.34,"Solid: runtime/data flow   Dashed: governance/control",ha="right",va="bottom",fontsize=9.3,color=GREY)
    save(fig,"system_architecture")


def governance() -> None:
    fig, ax = plt.subplots(figsize=(12.4,7.5))
    ax.set(xlim=(0,16), ylim=(0,10)); ax.axis("off")
    layer(ax,7.00,2.70,"Knowledge acquisition and expert governance",ORANGE,"#FCF1EB")
    layer(ax,3.95,2.70,"Approved production indexing",GREEN,"#F0F6EC")
    layer(ax,.25,3.35,"Leakage-controlled training and evaluation",BLUE,"#EDF5FA")

    node(ax,.35,7.45,2.40,1.15,"Railway sources","Textbooks and regulations",ORANGE,"book")
    node(ax,3.05,7.45,2.50,1.15,"Bilingual ingestion","Chunk, align and add metadata",ORANGE,"ingest")
    node(ax,5.85,7.45,2.65,1.15,"Governed records","PostgreSQL system of record",ORANGE,"database")
    node(ax,8.80,7.45,2.50,1.15,"Expert review gate","Approve / revise / reject",ORANGE,"shield")
    node(ax,11.85,7.45,3.45,1.15,"Approved knowledge view","Queryable state; version retained",ORANGE,"review")
    for a,b in ((2.75,3.05),(5.55,5.85),(8.50,8.80),(11.30,11.85)): arrow(ax,[(a,8.03),(b,8.03)])
    arrow(ax,[(10.05,7.45),(10.05,7.28),(7.18,7.28),(7.18,7.45)],"status + audit event",ORANGE,True,(8.62,7.28))
    arrow(ax,[(10.05,7.45),(10.05,7.08),(4.30,7.08),(4.30,7.45)],"revision request",ORANGE,True,(7.18,7.08))

    node(ax,.80,4.50,3.05,1.15,"Approved text fields","Chinese, English and provenance",GREEN,"document")
    node(ax,4.55,4.50,2.70,1.15,"BM25 index","Lexical production index",GREEN,"search")
    node(ax,8.00,4.50,2.75,1.15,"BGE-M3 encoder","1,024-dimensional vectors",GREEN,"network")
    node(ax,11.50,4.50,3.15,1.15,"pgvector index","Vectors linked to record IDs",GREEN,"database")
    arrow(ax,[(13.58,7.45),(13.58,6.82),(3.98,6.82),(3.98,5.88),(2.32,5.88),(2.32,5.65)],"approved-only view",GREEN,label_xy=(10.35,6.82))
    arrow(ax,[(3.85,5.08),(4.55,5.08)])
    arrow(ax,[(3.85,5.08),(4.18,5.08),(4.18,4.22),(9.38,4.22),(9.38,4.50)])
    arrow(ax,[(10.75,5.08),(11.50,5.08)])

    node(ax,.45,1.25,2.80,1.15,"Pair-grouped split","Train / validation / test",BLUE,"database")
    node(ax,3.85,1.65,2.70,1.15,"Train + validation","Approved pairs only",BLUE,"review")
    node(ax,3.85,.50,2.70,.95,"Held-out QA records","Exact records not indexed or trained",ORANGE,"lock")
    node(ax,7.25,1.65,2.70,1.15,"QLoRA training","Completion-only adapter",PURPLE,"adapter")
    node(ax,10.65,1.65,2.40,1.15,"Adapter","Local model revision",PURPLE,"cpu")
    node(ax,13.40,.95,2.20,1.40,"Evaluation","QA, RAG, translation and resources",BLUE,"chart")
    arrow(ax,[(7.75,7.45),(7.75,6.95),(7.45,6.95),(7.45,3.76),(.20,3.76),(.20,1.83),(.45,1.83)],"pair-grouped export",BLUE,True,(4.70,3.76))
    arrow(ax,[(3.25,1.83),(3.55,1.83),(3.55,2.23),(3.85,2.23)])
    arrow(ax,[(3.25,1.83),(3.55,1.83),(3.55,.98),(3.85,.98)])
    arrow(ax,[(6.55,2.23),(7.25,2.23)])
    arrow(ax,[(9.95,2.23),(10.65,2.23)])
    arrow(ax,[(13.05,2.23),(13.22,2.23),(13.22,1.90),(13.40,1.90)])
    arrow(ax,[(6.55,.98),(6.90,.98),(6.90,.58),(13.10,.58),(13.10,1.30),(13.40,1.30)],"evaluation only",BLUE,label_xy=(10.00,.58))
    arrow(ax,[(13.08,4.50),(13.08,3.78),(14.50,3.78),(14.50,2.35)],"retrieval snapshot",GREEN,label_xy=(13.78,3.78))
    save(fig,"knowledge_governance_lifecycle")


def main() -> None:
    configure(); architecture(); governance()
    print(f"wrote={OUTPUT / 'system_architecture.pdf'}")
    print(f"wrote={OUTPUT / 'knowledge_governance_lifecycle.pdf'}")


if __name__ == "__main__":
    main()
