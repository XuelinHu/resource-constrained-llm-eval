from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/qlora_bilingual_approved/test.jsonl"
OUTPUT = ROOT / "data/efficiency/prompts.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    groups = {
        "zh_short": [row for row in rows if row["language"] == "zh" and not row["task_type"].startswith("regulation_")],
        "en_short": [row for row in rows if row["language"] == "en" and not row["task_type"].startswith("regulation_")],
        "regulation_long": [row for row in rows if row["task_type"].startswith("regulation_")],
    }
    selected = []
    for workload, candidates in groups.items():
        for row in sorted(candidates, key=lambda value: value["id"])[:30]:
            selected.append(
                {
                    "id": row["id"],
                    "workload": workload,
                    "language": row["language"],
                    "task_type": row["task_type"],
                    "prompt": row["prompt"],
                    "reference": row["answer"],
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(f"wrote={OUTPUT} rows={len(selected)}")


if __name__ == "__main__":
    main()
