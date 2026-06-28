from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ASR_UPLOAD_DIR = BACKEND_ROOT / ".cache" / "asr"
DEFAULT_ASR_MODEL = os.getenv("RAILWAY_ASR_MODEL", "base")
DEFAULT_ASR_DEVICE = os.getenv("RAILWAY_ASR_DEVICE", "cpu")
DEFAULT_ASR_COMPUTE_TYPE = os.getenv("RAILWAY_ASR_COMPUTE_TYPE", "int8")


@lru_cache(maxsize=1)
def get_asr_model() -> WhisperModel:
    return WhisperModel(
        DEFAULT_ASR_MODEL,
        device=DEFAULT_ASR_DEVICE,
        compute_type=DEFAULT_ASR_COMPUTE_TYPE,
    )


def transcribe_audio(audio_path: Path, language: str = "zh") -> str:
    model = get_asr_model()
    segments, _info = model.transcribe(
        str(audio_path),
        language=language or "zh",
        vad_filter=True,
        beam_size=5,
    )
    return "".join(segment.text.strip() for segment in segments).strip()
