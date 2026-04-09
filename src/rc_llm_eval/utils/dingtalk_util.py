"""钉钉消息通知工具。

支持两类消息：
- Markdown 通知
- ActionCard 正式通知
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from geomatric.logging_config import logger
except ImportError:  # pragma: no cover - 兼容当前仓库未提供该依赖的情况
    logger = logging.getLogger(__name__)


DEFAULT_ACCESS_TOKEN = "f38e4a0b83763311cff9aed9bfc1bb789dafa10c51ed8356649dcba8786feea2"


def _post_robot_payload(
    access_token: str,
    body: Dict[str, Any],
    timeout: int = 10,
) -> Dict[str, Any]:
    """发送原始钉钉机器人 payload。"""

    url = f"https://oapi.dingtalk.com/robot/send?access_token={access_token}"
    payload = json.dumps(body).encode("utf-8")
    http_request = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        logger.error("钉钉机器人消息发送失败：%s", response_text)
        raise

    logger.info("钉钉机器人消息响应：%s", response_text)
    return json.loads(response_text)


def send_custom_robot_group_message(
    access_token: str,
    msg: str,
    at_user_ids: Optional[Iterable[str]] = None,
    at_mobiles: Optional[Iterable[str]] = None,
    is_at_all: bool = False,
    title: str = "【通知】",
    timeout: int = 10,
) -> Dict[str, Any]:
    """发送钉钉 Markdown 群消息。

    默认参数：
    - `is_at_all=False`：默认不 @ 所有人
    - `title=\"【通知】\"`：默认通知标题
    - `timeout=10`：HTTP 请求超时 10 秒
    """

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = {
        "at": {
            "isAtAll": str(is_at_all).lower(),
            "atUserIds": list(at_user_ids or []),
            "atMobiles": list(at_mobiles or []),
        },
        "markdown": {
            "title": f"{formatted_time}{title}",
            "text": msg,
        },
        "msgtype": "markdown",
    }
    return _post_robot_payload(access_token=access_token, body=body, timeout=timeout)


def send_action_card_message(
    access_token: str,
    title: str,
    text: str,
    single_title: str,
    single_url: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """发送钉钉 ActionCard 正式通知。"""

    body = {
        "msgtype": "actionCard",
        "actionCard": {
            "title": title,
            "text": text,
            "singleTitle": single_title,
            "singleURL": single_url,
        },
    }
    return _post_robot_payload(access_token=access_token, body=body, timeout=timeout)


def send_release_success_notice(
    project_name: str,
    branch: str,
    deployed_at: str,
    result_text: str,
    detail_url: str,
    access_token: str = DEFAULT_ACCESS_TOKEN,
    notice_title: str = "发布成功通知",
    button_text: str = "查看详情",
    timeout: int = 10,
) -> Dict[str, Any]:
    """发送发布成功的正式通知卡片。

    该方法生成的 payload 结构与钉钉 `actionCard` 正式通知一致。
    """

    text = (
        "## 发布成功\n"
        f"**项目**：{project_name}\n\n"
        f"**分支**：{branch}\n\n"
        f"**时间**：{deployed_at}\n\n"
        f"**结果**：{result_text}\n\n"
        "请点击下方按钮查看详情。"
    )
    return send_action_card_message(
        access_token=access_token,
        title=notice_title,
        text=text,
        single_title=button_text,
        single_url=detail_url,
        timeout=timeout,
    )


def send_to_dingtalk(msg: str, err: bool = False) -> Dict[str, Any]:
    """发送一条带时间戳的钉钉通知。

    默认使用模块中固化的 access token；`err=True` 时标题切换为错误通知。
    """

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return send_custom_robot_group_message(
        DEFAULT_ACCESS_TOKEN,
        f"{formatted_time}通知\n{msg}",
        at_user_ids=[],
        at_mobiles=[],
        is_at_all=False,
        title="【错误】" if err else "【通知】",
    )
