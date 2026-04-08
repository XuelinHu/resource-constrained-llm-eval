"""配置加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# 统一以仓库根目录作为相对路径解析基准。
ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML；如果传入相对路径，则相对于仓库根目录解析。"""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_all_configs(experiment_path: str | Path) -> dict[str, Any]:
    """一次性加载实验、模型与任务配置，形成统一字典。"""
    experiment = load_yaml(experiment_path)
    models = load_yaml(ROOT / "configs" / "models" / "models.yaml")
    tasks = load_yaml(ROOT / "configs" / "datasets" / "tasks.yaml")
    return {
        "root": ROOT,
        "experiment": experiment,
        "models": models["models"],
        "tasks": tasks["tasks"],
    }
