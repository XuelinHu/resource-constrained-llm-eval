"""Compute optional reference-based COMET metrics for bidirectional translation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


DIRECTIONS = {
    "zh_to_en": {"terminology_zh_to_en", "zh_to_en_translation"},
    "en_to_zh": {"terminology_en_to_zh", "en_to_zh_translation"},
}

SUBTASKS = {
    "terminology": lambda category: category.startswith("terminology_"),
    "sentence": lambda category: category.endswith("_translation") and not category.startswith("terminology_"),
}


def bootstrap_ci(scores: list[float], *, seed: int = 42, samples: int = 2000) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(samples))
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate translation generations with reference-based COMET.")
    parser.add_argument("generations", type=Path)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise SystemExit("Install optional COMET dependencies: pip install -r requirements-comet.txt") from exc

    payload = json.loads(args.generations.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload)
    checkpoint = str(args.checkpoint) if args.checkpoint else download_model(args.model)
    model = load_from_checkpoint(checkpoint)
    output_rows = []
    summaries = []
    for direction, categories in DIRECTIONS.items():
        for subtask, predicate in SUBTASKS.items():
            selected = [
                sample
                for sample in samples
                if sample.get("category") in categories and predicate(sample.get("category", ""))
            ]
            missing_source = [sample for sample in selected if not sample.get("source_text")]
            if missing_source:
                raise ValueError(f"{direction}: {len(missing_source)} samples are missing source_text")
            data = [
                {"src": sample["source_text"], "mt": sample["prediction"], "ref": sample["reference"]}
                for sample in selected
            ]
            if not data:
                continue
            prediction = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
            scores = [float(score) for score in prediction.scores]
            low, high = bootstrap_ci(scores)
            summaries.append(
                {
                    "direction": direction,
                    "subtask": subtask,
                    "model": args.model,
                    "checkpoint": str(checkpoint),
                    "num_examples": len(scores),
                    "mean": statistics.mean(scores),
                    "ci95_low": low,
                    "ci95_high": high,
                    "score_scale": "COMET model-native score; higher is better",
                }
            )
            for sample, score in zip(selected, scores, strict=True):
                output_rows.append(
                    {
                        "direction": direction,
                        "subtask": subtask,
                        "pair_id": sample.get("pair_id"),
                        "category": sample.get("category"),
                        "source_text": sample.get("source_text"),
                        "prediction": sample.get("prediction"),
                        "reference": sample.get("reference"),
                        "score": score,
                    }
                )

    output = args.output or args.generations.with_name(f"{args.generations.stem}_comet.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summaries": summaries, "samples": output_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
