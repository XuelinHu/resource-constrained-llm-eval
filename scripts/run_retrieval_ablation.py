"""Run the controlled top-k retrieval ablation in one reusable model process."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.evaluate_retrieval import evaluate_mode, load_cases


def main() -> None:
    cases = load_cases(language="both", test_set="railway_bilingual_400")
    languages = sorted({case.language for case in cases})
    modes = (("bm25", False), ("vector", False), ("hybrid", False), ("hybrid", True))
    output_dir = Path("data/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    for top_k in (1, 3, 5, 8):
        results = [
            evaluate_mode(
                [case for case in cases if case.language == language],
                mode=mode,
                top_k=top_k,
                approved_only=approved_only,
            )
            for mode, approved_only in modes
            for language in languages
        ]
        payload = {"top_k": top_k, "cases": [asdict(case) for case in cases], "results": results}
        output = output_dir / f"retrieval_eval_top{top_k}.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        compact = [{key: value for key, value in result.items() if key != "examples"} for result in results]
        print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
        print(f"wrote={output}", flush=True)
    formal = output_dir / "retrieval_eval_railway_bilingual_400.json"
    formal.write_text((output_dir / "retrieval_eval_top5.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote={formal}")


if __name__ == "__main__":
    main()
