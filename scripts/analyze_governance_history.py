"""Audit real knowledge-review events without inferring unobserved quality gains."""

from __future__ import annotations

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import func, select


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import CorpusItem, ReviewEvent


RESULT_JSON = ROOT / "results/ijwis_single_gpu_3090/analysis/governance_history_audit.json"
TABLE_CSV = ROOT / "paper/ijwis/tables/table11_governance_audit.csv"
TABLE_TEX = ROOT / "paper/ijwis/tables/table11_governance_audit.tex"
FIGURE_PNG = ROOT / "paper/ijwis/figures/governance_history_audit.png"
FIGURE_PDF = ROOT / "paper/ijwis/figures/governance_history_audit.pdf"
COMPARABLE_FIELDS = (
    "question",
    "answer",
    "evidence",
    "task_type",
    "domain_category",
    "knowledge_category",
    "chapter",
    "page_number",
)


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    with SessionLocal() as db:
        status_counts = dict(
            db.execute(select(CorpusItem.review_status, func.count()).group_by(CorpusItem.review_status)).all()
        )
        action_counts = dict(
            db.execute(select(ReviewEvent.action, func.count()).group_by(ReviewEvent.action)).all()
        )
        distinct_reviewers = db.scalar(
            select(func.count(func.distinct(ReviewEvent.reviewer))).where(ReviewEvent.reviewer != "")
        )
        edit_pairs = db.execute(
            select(ReviewEvent, CorpusItem)
            .join(CorpusItem, CorpusItem.id == ReviewEvent.item_id)
            .where(ReviewEvent.action == "edit")
            .order_by(ReviewEvent.id)
        ).all()

    changed_fields: Counter[str] = Counter()
    similarities: dict[str, list[float]] = {field: [] for field in ("question", "answer", "evidence")}
    edit_rows: list[dict] = []
    for event, item in edit_pairs:
        before = (event.snapshot or {}).get("before") or {}
        changed = []
        for field in COMPARABLE_FIELDS:
            before_value = before.get(field)
            current_value = getattr(item, field)
            if normalized_text(before_value) != normalized_text(current_value):
                changed_fields[field] += 1
                changed.append(field)
            if field in similarities and before_value is not None:
                similarities[field].append(
                    SequenceMatcher(None, normalized_text(before_value), normalized_text(current_value)).ratio()
                )
        edit_rows.append(
            {
                "event_id": event.id,
                "item_id": item.id,
                "current_status": item.review_status,
                "current_version": item.version,
                "changed_fields": changed,
            }
        )

    metrics = [
        {"metric": "Corpus records", "value": sum(status_counts.values()), "note": "Current database snapshot"},
        {"metric": "Approved records", "value": status_counts.get("approved", 0), "note": "Eligible governance state"},
        {"metric": "Rejected records", "value": status_counts.get("rejected", 0), "note": "Insufficient for causal filtering analysis"},
        {"metric": "Recorded review events", "value": sum(action_counts.values()), "note": "Immutable event log"},
        {"metric": "Edit events", "value": action_counts.get("edit", 0), "note": "Events with before-state snapshots"},
        {"metric": "Distinct recorded reviewers", "value": int(distinct_reviewers or 0), "note": "Names not exported"},
        {"metric": "Approved actions", "value": action_counts.get("approved", 0) + action_counts.get("batch_approved", 0), "note": "Individual and batch actions"},
        {"metric": "Rejected actions", "value": action_counts.get("batch_rejected", 0), "note": "Recorded rejection actions"},
    ]
    payload = {
        "status_counts": status_counts,
        "action_counts": action_counts,
        "distinct_recorded_reviewers": distinct_reviewers,
        "changed_field_counts": dict(changed_fields),
        "mean_before_current_similarity": {
            field: sum(values) / len(values) if values else None for field, values in similarities.items()
        },
        "edit_events": edit_rows,
        "interpretation_boundary": (
            "This audit demonstrates executable provenance and revision activity. "
            "It does not estimate a causal quality benefit from expert approval because only three current records are rejected."
        ),
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(metrics)
    frame.to_csv(TABLE_CSV, index=False)
    TABLE_TEX.write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    actions = sorted(action_counts.items(), key=lambda item: item[1], reverse=True)
    axes[0].barh([name.replace("_", " ") for name, _ in reversed(actions)], [value for _, value in reversed(actions)], color="#0072B2")
    axes[0].set_title("Recorded governance actions")
    axes[0].set_xlabel("Events")
    fields = sorted(changed_fields.items(), key=lambda item: item[1], reverse=True)
    axes[1].barh([name.replace("_", " ") for name, _ in reversed(fields)], [value for _, value in reversed(fields)], color="#D55E00")
    axes[1].set_title("Fields changed after edit events")
    axes[1].set_xlabel("Edit events")
    for axis in axes:
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)
    print(frame.to_string(index=False))
    print(f"wrote={RESULT_JSON}")


if __name__ == "__main__":
    main()
