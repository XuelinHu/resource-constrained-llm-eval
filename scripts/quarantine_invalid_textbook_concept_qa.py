"""Quarantine the invalid long-context textbook concept QA batch."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "annotation_system" / "backend"))

from app.database import SessionLocal  # noqa: E402
from app.models import CorpusItem  # noqa: E402


TARGET_METHOD = "expanded_current_textbook_long_context"
FLAGS = {"human_review_required", "answer_boundary_error", "answer_incomplete", "question_answer_mismatch"}


def main() -> None:
    with SessionLocal() as db:
        rows = list(
            db.scalars(
                select(CorpusItem).where(
                    CorpusItem.source_type == "textbook_original_md",
                    CorpusItem.task_type == "concept_explanation_qa",
                    CorpusItem.metadata_json["generation_method"].astext == TARGET_METHOD,
                )
            )
        )
        status_counts: dict[str, int] = {}
        for item in rows:
            if item.review_status != "rejected":
                item.review_status = "needs_revision"
            item.quality_flags = sorted(set(item.quality_flags or []) | FLAGS)
            item.review_comment = item.review_comment or "整批隔离：长上下文生成存在跨段拼接、标题混入、图表引用或答案不完整问题，需重新生成。"
            status_counts[item.review_status] = status_counts.get(item.review_status, 0) + 1
        db.commit()
    print(f"quarantined={len(rows)} statuses={status_counts}")


if __name__ == "__main__":
    main()
