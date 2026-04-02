from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_all_configs(experiment_path: str | Path) -> dict[str, Any]:
    experiment = load_yaml(experiment_path)
    models = load_yaml(ROOT / "configs" / "models" / "models.yaml")
    tasks = load_yaml(ROOT / "configs" / "datasets" / "tasks.yaml")
    return {
        "root": ROOT,
        "experiment": experiment,
        "models": models["models"],
        "tasks": tasks["tasks"],
    }
