"""Export classified railway terminology records for manual review.

The source inventory keeps the original subdomain labels extracted from the
classified vocabulary document. This exporter writes a stable JSONL and CSV
view where each term pair is explicitly tied to concrete category names.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "domain" / "terminology_inventory.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "railway_education"


FIELDS = [
    "domain_category",
    "domain_category_en",
    "domain_category_key",
    "term_zh",
    "term_en",
    "source",
    "source_block",
    "abbreviation",
    "full_name_en",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export railway terminology by concrete category names.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def validate(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    required = ["domain_category", "domain_category_en", "domain_category_key", "term_zh", "term_en"]
    seen = defaultdict(set)
    for row in rows:
        for key in required:
            if not row.get(key):
                issues.append(f"line {row['_line_no']}: missing {key}")
        seen[(row.get("term_zh"), row.get("term_en"))].add(row.get("domain_category"))
    for (term_zh, term_en), categories in sorted(seen.items()):
        if len(categories) > 1:
            issues.append(f"multi-category term pair: {term_zh} / {term_en} -> {', '.join(sorted(categories))}")
    return issues


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("domain_category", "")),
            str(row.get("term_zh", "")),
            str(row.get("term_en", "")),
            int(row.get("source_block", 0) or 0),
        ),
    )

    jsonl_path = args.output_dir / "terminology_by_category.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            clean = {field: row.get(field, "") for field in FIELDS}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")

    csv_path = args.output_dir / "terminology_by_category.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    issues = validate(rows)
    issue_path = args.output_dir / "terminology_category_issues.txt"
    issue_path.write_text("\n".join(issues) + ("\n" if issues else ""), encoding="utf-8")

    category_counts = Counter(row.get("domain_category", "") for row in rows)
    readme_lines = [
        "# Railway Education Terminology Export",
        "",
        "This folder contains a manual-reviewable export of railway terminology.",
        "Each row keeps concrete category names rather than relying on numeric category indexes.",
        "",
        f"- source inventory: `{args.input.relative_to(REPO_ROOT)}`",
        f"- total rows: {len(rows)}",
        f"- category issue lines: {len(issues)}",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in sorted(category_counts.items()):
        readme_lines.append(f"- {category}: {count}")
    readme_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `terminology_by_category.jsonl`: structured export for scripts.",
            "- `terminology_by_category.csv`: spreadsheet-friendly export for manual review.",
            "- `terminology_category_issues.txt`: duplicate cross-category pairs and missing-field issues.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"jsonl={jsonl_path}")
    print(f"csv={csv_path}")
    print(f"issues={len(issues)} {issue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
