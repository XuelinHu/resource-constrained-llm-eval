from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.evaluate_qa import summarize


def read_rows(path: Path, language: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    for row in rows:
        row["language"] = language
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently generated Chinese and English QA runs.")
    parser.add_argument("--chinese", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.chinese, "zh") + read_rows(args.english, "en")
    strategies = list(dict.fromkeys(row["strategy"] for row in rows))
    summaries = [summarize(rows, strategy, language) for strategy in strategies for language in ("zh", "en")]
    payload = {"summaries": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
