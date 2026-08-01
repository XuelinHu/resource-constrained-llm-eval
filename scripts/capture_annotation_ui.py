"""Capture real review and RAG views from the running annotation Web client."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path.home() / ".cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
REVIEW_SCREENSHOT = ROOT / "paper/ijwis/figures/system_review_console_screenshot.png"
RAG_SCREENSHOT = ROOT / "paper/ijwis/figures/system_bilingual_qa_screenshot.png"


class CdpClient:
    def __init__(self, socket: websocket.WebSocket) -> None:
        self.socket = socket
        self.message_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.message_id += 1
        request_id = self.message_id
        self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self.socket.recv())
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result") or {}

    def evaluate(self, expression: str) -> dict:
        return self.call(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )

    def screenshot(self, path: Path) -> None:
        result = self.call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))


def page_target(port: int, timeout: float = 20) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1) as response:
                targets = json.load(response)
            pages = [target for target in targets if target.get("type") == "page"]
            if pages:
                return pages[0]
        except (OSError, json.JSONDecodeError):
            time.sleep(0.2)
    raise TimeoutError("Chromium debugging target did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--debug-port", type=int, default=9222)
    args = parser.parse_args()
    if not CHROME.is_file():
        raise FileNotFoundError(CHROME)

    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    with tempfile.TemporaryDirectory(prefix="ijwis-chrome-", ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--hide-scrollbars",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={args.debug_port}",
                f"--user-data-dir={profile}",
                "--window-size=1600,1000",
                args.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            target = page_target(args.debug_port)
            socket = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=20)
            client = CdpClient(socket)
            client.call("Page.enable")
            client.call("Runtime.enable")
            client.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
            )
            time.sleep(2)
            client.evaluate("localStorage.setItem('railway-ui-language','en'); location.reload();")
            time.sleep(3)
            client.evaluate(
                "(() => {"
                "const status = Array.from(document.querySelectorAll('.filter-grid select'))"
                ".find(select => Array.from(select.options).some(option => option.value === 'approved'));"
                "status.value = 'approved';"
                "status.dispatchEvent(new Event('change', {bubbles: true}));"
                "})()"
            )
            time.sleep(4)
            client.evaluate("document.querySelector('.queue-item-main')?.click()")
            time.sleep(2)
            client.screenshot(REVIEW_SCREENSHOT)

            if args.session_id:
                client.evaluate(
                    "localStorage.setItem('railway-ui-language','en');"
                    f"localStorage.setItem('railway-rag-session-id','{args.session_id}');"
                    "location.reload();"
                )
                time.sleep(4)
                client.evaluate(
                    "Array.from(document.querySelectorAll('button'))"
                    ".find(button => button.textContent.includes('RAG')).click()"
                )
                time.sleep(5)
                client.screenshot(RAG_SCREENSHOT)
            socket.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    print(REVIEW_SCREENSHOT)
    if args.session_id:
        print(RAG_SCREENSHOT)


if __name__ == "__main__":
    main()
