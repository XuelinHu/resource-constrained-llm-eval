"""文本规范化工具。

这里的规范化主要用于本地域问答的精确匹配比较，
尽量去除大小写、空白和常见标点的干扰。
"""

from __future__ import annotations

import re


# 合并多余空白，避免换行或连续空格影响比对结果。
_SPACE_RE = re.compile(r"\s+")
# 保留字母、数字和中文字符，其他符号统一视为分隔符。
_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_answer(text: str) -> str:
    """将输出与参考答案映射到便于比较的统一格式。"""
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()
