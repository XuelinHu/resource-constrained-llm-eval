from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable


def run_command(command: list[str], cwd: Path | None = None) -> int:
    print(">>", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def flatten(items: Iterable[Iterable[str]]) -> list[str]:
    return [value for group in items for value in group]
