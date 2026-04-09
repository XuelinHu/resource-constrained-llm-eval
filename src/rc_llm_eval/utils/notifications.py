"""实验通知辅助工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_run_name(
    *,
    prefix_parts: list[object],
    batch_size: str | int,
    learning_rate: float | None = None,
    precision: str | None = None,
    seed: int | None = None,
    label: str | None = None,
    timestamp: datetime,
) -> str:
    fragments = [str(part) for part in prefix_parts]
    fragments.append(f"bs{batch_size}")
    if learning_rate is not None:
        fragments.append(f"lr{learning_rate:g}")
    if precision is not None:
        fragments.append(str(precision))
    if seed is not None:
        fragments.append(f"seed{seed}")
    if label:
        fragments.append(label)
    fragments.append(timestamp.strftime("%Y%m%d-%H%M%S"))
    return "_".join(fragments)


def send_dingtalk_notification(message: str, err: bool = False) -> None:
    try:
        from ..utils.dingtalk_util import send_to_dingtalk

        send_to_dingtalk(message, err=err)
    except Exception as exc:  # pragma: no cover - 通知失败不影响主流程
        print(f"Warning: failed to send DingTalk notification: {type(exc).__name__}: {exc}")


def build_markdown_message(title: str, fields: list[tuple[str, Any]]) -> str:
    lines = [f"## {title}"]
    for key, value in fields:
        lines.append(f"- {key}：{value}")
    return "\n".join(lines)
