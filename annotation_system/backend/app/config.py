from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = BACKEND_ROOT / ".env") -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    cors_origins: tuple[str, ...]
    ollama_url: str
    rag_model: str
    rag_timeout_seconds: int
    embedding_model: str
    embedding_dimension: int

    @property
    def database_url(self) -> str:
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return f"postgresql+psycopg://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


def get_settings() -> Settings:
    load_env()
    origins = os.getenv(
        "RAILWAY_CORS_ORIGINS",
        "http://localhost:4005,http://127.0.0.1:4005,http://localhost:5173,http://127.0.0.1:5173,http://192.168.1.9:4005,http://192.168.1.9:5173,http://47.120.48.245:14005",
    )
    return Settings(
        db_host=os.getenv("RAILWAY_DB_HOST", "localhost"),
        db_port=int(os.getenv("RAILWAY_DB_PORT", "5432")),
        db_user=os.getenv("RAILWAY_DB_USER", "deipss"),
        db_password=os.getenv("RAILWAY_DB_PASSWORD", ""),
        db_name=os.getenv("RAILWAY_DB_NAME", "railway_annotation"),
        cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        ollama_url=os.getenv("RAILWAY_OLLAMA_URL", "http://127.0.0.1:11434"),
        rag_model=os.getenv("RAILWAY_RAG_MODEL", "qwen3:14b"),
        rag_timeout_seconds=int(os.getenv("RAILWAY_RAG_TIMEOUT_SECONDS", "180")),
        embedding_model=os.getenv("RAILWAY_EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_dimension=int(os.getenv("RAILWAY_EMBEDDING_DIMENSION", "1024")),
    )


settings = get_settings()
