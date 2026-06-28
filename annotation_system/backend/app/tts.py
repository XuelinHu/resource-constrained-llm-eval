from __future__ import annotations

import hashlib
from pathlib import Path

import edge_tts


BACKEND_ROOT = Path(__file__).resolve().parents[1]
TTS_CACHE_DIR = BACKEND_ROOT / ".cache" / "tts"
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


def normalize_rate(rate: float) -> str:
    percent = int(round((rate - 1.0) * 100))
    sign = "+" if percent >= 0 else ""
    return f"{sign}{percent}%"


def tts_cache_key(text: str, voice: str, rate: float) -> str:
    payload = f"{voice}\x1f{rate:.2f}\x1f{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


async def synthesize_speech(text: str, voice: str = DEFAULT_VOICE, rate: float = 1.0) -> Path:
    content = text.strip()
    if not content:
        raise ValueError("empty text")

    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TTS_CACHE_DIR / f"{tts_cache_key(content, voice, rate)}.mp3"
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path

    communicate = edge_tts.Communicate(content, voice=voice, rate=normalize_rate(rate))
    await communicate.save(str(cache_path))
    return cache_path
