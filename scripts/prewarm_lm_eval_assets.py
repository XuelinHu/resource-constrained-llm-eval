"""Preload configured lm-eval task assets without loading any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm_eval.tasks import TaskManager, get_task_dict

from src.rc_llm_eval.utils.config import load_all_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prewarm lm-eval and local JSONL task assets.")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/single_gpu_3090.yaml",
        help="Experiment config to inspect.",
    )
    return parser


def count_jsonl(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                rows += 1
    return rows


def main() -> int:
    args = build_parser().parse_args()
    configs = load_all_configs(args.experiment)
    tasks = configs["tasks"]
    baseline_tasks = configs["experiment"]["baseline"]["tasks"]

    lm_eval_tasks = [tasks[key]["task_name"] for key in baseline_tasks if tasks[key]["suite"] == "lm_eval"]
    local_tasks = [key for key in baseline_tasks if tasks[key]["suite"] == "local_jsonl"]

    if lm_eval_tasks:
        print(f"Prewarming lm-eval tasks: {', '.join(lm_eval_tasks)}", flush=True)
        task_manager = TaskManager(verbosity="ERROR")
        task_dict = get_task_dict(lm_eval_tasks, task_manager=task_manager)
        print(f"Loaded lm-eval task entries: {len(task_dict)}", flush=True)

    for task_key in local_tasks:
        task_cfg = tasks[task_key]
        print(f"Checking local task: {task_key}", flush=True)
        for split, field in [("train", "train_file"), ("valid", "valid_file"), ("test", "test_file")]:
            path = configs["root"] / task_cfg[field]
            rows = count_jsonl(path)
            print(f"  {split}: {rows} rows {path}", flush=True)

    print("Prewarm complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
