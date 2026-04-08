"""系统级辅助函数。

这些函数负责目录创建、JSON 落盘以及命令执行等通用操作，
供多个流水线模块共享。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable


def run_command(command: list[str], cwd: Path | None = None) -> int:
    """执行外部命令，并将命令本身打印出来便于追踪。"""
    print(">>", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def ensure_directory(path: Path) -> Path:
    """确保目标目录存在，并返回目录路径本身。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    """按 UTF-8 编码输出 JSON，保留中文字符原样。"""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def flatten(items: Iterable[Iterable[str]]) -> list[str]:
    """将二维可迭代对象拍平成单层列表。"""
    return [value for group in items for value in group]
