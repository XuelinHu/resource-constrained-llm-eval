"""Build a stratified 1% smoke dataset from the full railway domain dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "domain"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "domain_smoke"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_sample(rows: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)

    sampled: list[dict[str, Any]] = []
    for category_rows in by_category.values():
        rows_copy = list(category_rows)
        rng.shuffle(rows_copy)
        sample_size = max(1, round(len(rows_copy) * ratio))
        sampled.extend(rows_copy[:sample_size])
    rng.shuffle(sampled)
    return sampled


def write_readme(output_dir: Path, counts: dict[str, int], ratio: float) -> None:
    lines = [
        "# Railway Domain Smoke Dataset",
        "",
        f"This dataset is a stratified {ratio:.2%} sample of `data/domain`.",
        "It is intended for foreground smoke tests only, not formal reporting.",
        "",
        "## Splits",
        "",
    ]
    for split, count in counts.items():
        lines.append(f"- {split}: {count}")
    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the railway domain smoke dataset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts: dict[str, int] = {}
    for split in ["train", "valid", "test"]:
        rows = read_jsonl(args.source_dir / f"{split}.jsonl")
        sampled = stratified_sample(rows, args.ratio, args.seed)
        write_jsonl(args.output_dir / f"{split}.jsonl", sampled)
        counts[split] = len(sampled)
        print(f"{split}: source={len(rows)} smoke={len(sampled)}")
    write_readme(args.output_dir, counts, args.ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
