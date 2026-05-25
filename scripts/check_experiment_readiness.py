"""Check whether the configured experiment is ready to run.

This script is intentionally read-only: it does not download models, start
training, or run evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rc_llm_eval.utils.config import load_all_configs


HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
REQUIRED_DATA_FIELDS = {"prompt", "answer", "text", "category", "source"}


def model_cache_dir(hf_id: str) -> Path:
    return HF_HUB / f"models--{hf_id.replace('/', '--')}"


def latest_snapshot(cache_dir: Path) -> Path | None:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def model_status(hf_id: str) -> dict[str, Any]:
    cache_dir = model_cache_dir(hf_id)
    snapshot = latest_snapshot(cache_dir)
    status = {
        "hf_id": hf_id,
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "complete": False,
        "weight_files": 0,
        "missing_weight_files": 0,
        "incomplete_blobs": 0,
    }
    if not cache_dir.exists() or snapshot is None:
        return status

    index_files = list(snapshot.glob("*.safetensors.index.json")) + list(snapshot.glob("pytorch_model.bin.index.json"))
    weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
    incomplete_blobs = list(cache_dir.rglob("*.incomplete"))
    status["incomplete_blobs"] = len(incomplete_blobs)

    if index_files:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        needed = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in needed if not (snapshot / name).exists()]
        status["weight_files"] = len(needed) - len(missing)
        status["missing_weight_files"] = len(missing)
        status["complete"] = not missing
        return status

    status["weight_files"] = len(weights)
    status["complete"] = bool(weights)
    return status


def dataset_status(path: Path) -> dict[str, Any]:
    status = {"path": str(path), "exists": path.exists(), "rows": 0, "bad_rows": 0, "categories": {}}
    if not path.exists():
        return status

    categories: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            status["rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                status["bad_rows"] += 1
                continue
            if not REQUIRED_DATA_FIELDS <= record.keys() or not record.get("prompt") or not record.get("answer"):
                status["bad_rows"] += 1
            category = str(record.get("category", "unknown"))
            categories[category] = categories.get(category, 0) + 1
    status["categories"] = categories
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local experiment readiness without running experiments.")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/single_gpu_3090.yaml",
        help="Experiment config to inspect.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configs = load_all_configs(args.experiment)
    root = configs["root"]
    experiment_cfg = configs["experiment"]
    baseline_models = experiment_cfg["baseline"]["models"]
    qlora_models = experiment_cfg["qlora"]["candidate_models"]
    tasks = configs["tasks"]

    print("Experiment readiness report")
    print(f"Experiment: {args.experiment}")
    print(f"Output root: {experiment_cfg['experiment']['output_root']}")
    print()

    print("Baseline model cache")
    incomplete_models: list[str] = []
    for model_key in baseline_models:
        cfg = configs["models"][model_key]
        status = model_status(cfg["hf_id"])
        marker = "OK" if status["complete"] else "MISSING"
        if not status["complete"]:
            incomplete_models.append(model_key)
        print(
            f"- {marker} {model_key}: {cfg['hf_id']} "
            f"weights={status['weight_files']} missing={status['missing_weight_files']} "
            f"incomplete_blobs={status['incomplete_blobs']}"
        )
    print()

    print("QLoRA candidates")
    for model_key in qlora_models:
        cfg = configs["models"][model_key]
        status = model_status(cfg["hf_id"])
        marker = "OK" if status["complete"] else "MISSING"
        print(f"- {marker} {model_key}: {cfg['hf_id']}")
    print()

    print("Local JSONL datasets")
    local_task_keys = [
        task_key
        for task_key in experiment_cfg["baseline"]["tasks"] + experiment_cfg["qlora"]["candidate_models"][:0]
        if task_key in tasks and tasks[task_key]["suite"] == "local_jsonl"
    ]
    if not local_task_keys and "domain_qa" in tasks:
        local_task_keys = ["domain_qa"]
    for task_key in local_task_keys:
        domain_cfg = tasks[task_key]
        print(f"- task: {task_key}")
        for split, key in [("train", "train_file"), ("valid", "valid_file"), ("test", "test_file")]:
            status = dataset_status(root / domain_cfg[key])
            marker = "OK" if status["exists"] and status["rows"] > 0 and status["bad_rows"] == 0 else "BAD"
            print(f"  {marker} {split}: rows={status['rows']} bad_rows={status['bad_rows']} path={status['path']}")
            if status["categories"]:
                category_text = ", ".join(f"{name}={count}" for name, count in sorted(status["categories"].items()))
                print(f"    categories: {category_text}")
    print()

    print("Foreground commands to run after confirmation")
    print("1. python scripts/check_experiment_readiness.py --experiment configs/experiments/single_gpu_3090.yaml")
    print("2. bash scripts/run_baseline_pilot.sh")
    print("3. bash scripts/run_qlora_smoke.sh")
    print("4. bash scripts/run_baseline_all.sh")
    print("5. bash scripts/run_qlora_all.sh")
    print("6. bash scripts/run_qlora_eval_all.sh")
    print("7. python -m src.rc_llm_eval.cli export-paper-tables --experiment configs/experiments/single_gpu_3090.yaml")
    return 1 if incomplete_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
