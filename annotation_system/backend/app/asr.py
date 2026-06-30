from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

try:
    from opencc import OpenCC
except Exception:  # pragma: no cover - optional runtime dependency fallback
    OpenCC = None  # type: ignore[assignment]


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ASR_UPLOAD_DIR = BACKEND_ROOT / ".cache" / "asr"
DEFAULT_ASR_MODEL = os.getenv("RAILWAY_ASR_MODEL", "base")
DEFAULT_ASR_DEVICE = os.getenv("RAILWAY_ASR_DEVICE", "cpu")
DEFAULT_ASR_COMPUTE_TYPE = os.getenv("RAILWAY_ASR_COMPUTE_TYPE", "int8")

_OPENCC = OpenCC("t2s") if OpenCC else None
_FALLBACK_T2S = str.maketrans(
    {
        "臺": "台",
        "台": "台",
        "鐵": "铁",
        "路": "路",
        "電": "电",
        "氣": "气",
        "網": "网",
        "觸": "触",
        "點": "点",
        "線": "线",
        "檢": "检",
        "修": "修",
        "運": "运",
        "行": "行",
        "規": "规",
        "章": "章",
        "標": "标",
        "準": "准",
        "應": "应",
        "該": "该",
        "時": "时",
        "與": "与",
        "為": "为",
        "後": "后",
        "將": "将",
        "對": "对",
        "進": "进",
        "過": "过",
        "開": "开",
        "關": "关",
        "閉": "闭",
        "斷": "断",
        "連": "连",
        "接": "接",
        "設": "设",
        "備": "备",
        "護": "护",
        "維": "维",
        "處": "处",
        "異": "异",
        "常": "常",
        "壓": "压",
        "絕": "绝",
        "緣": "缘",
        "測": "测",
        "試": "试",
        "責": "责",
        "實": "实",
        "現": "现",
        "務": "务",
        "員": "员",
        "區": "区",
        "間": "间",
        "號": "号",
        "車": "车",
        "輛": "辆",
        "牽": "牵",
        "供": "供",
        "營": "营",
        "學": "学",
        "習": "习",
        "題": "题",
        "問": "问",
        "答": "答",
    }
)


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
    return to_simplified_chinese("".join(segment.text.strip() for segment in segments).strip())


def to_simplified_chinese(text: str) -> str:
    if not text:
        return ""
    if _OPENCC:
        return _OPENCC.convert(text)
    return text.translate(_FALLBACK_T2S)
