from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import bitsandbytes
import peft
import psycopg
import torch
import transformers
import yaml
from sqlalchemy import func, select, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_system.backend.app.config import settings
from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import CorpusItem, KnowledgeChunkEmbedding


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_ref(*paths: Path) -> str | None:
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def main() -> None:
    output_dir = ROOT / "results/ijwis_single_gpu_3090/manifest"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracked_inputs = [
        ROOT / "configs/experiments/ijwis_single_gpu_3090.yaml",
        ROOT / "configs/models/models.yaml",
        ROOT / "configs/datasets/tasks.yaml",
        ROOT / "data/qlora_bilingual_approved/train.jsonl",
        ROOT / "data/qlora_bilingual_approved/valid.jsonl",
        ROOT / "data/qlora_bilingual_approved/test.jsonl",
        ROOT / "data/domain/test.jsonl",
        ROOT / "data/rag_eval/railway_bilingual_400.jsonl",
    ]

    with SessionLocal() as db:
        db_stats = {
            "postgresql_version": db.execute(text("select version()")).scalar_one(),
            "pgvector_version": db.execute(text("select extversion from pg_extension where extname='vector'")).scalar_one(),
            "approved_items": db.scalar(select(func.count()).select_from(CorpusItem).where(CorpusItem.review_status == "approved")),
            "bilingual_approved_items": db.scalar(
                select(func.count()).select_from(CorpusItem).where(
                    CorpusItem.review_status == "approved",
                    CorpusItem.question_en != "",
                    CorpusItem.answer_en != "",
                )
            ),
            "embeddings": db.scalar(select(func.count()).select_from(KnowledgeChunkEmbedding)),
            "rag_test_pairs": db.scalar(
                select(func.count()).select_from(CorpusItem).where(
                    CorpusItem.metadata_json["rag_test_set"].as_string() == "railway_bilingual_400"
                )
            ),
        }

    experiment = yaml.safe_load(tracked_inputs[0].read_text(encoding="utf-8"))
    payload = {
        "frozen_at": datetime.now().astimezone().isoformat(),
        "git": {
            "commit": command("git", "rev-parse", "HEAD"),
            "branch": command("git", "branch", "--show-current"),
            "status": command("git", "status", "--short"),
            "tracked_diff_sha256": hashlib.sha256(command("git", "diff", "--binary").encode()).hexdigest(),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
            "cuda_runtime": torch.version.cuda,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "psycopg": psycopg.__version__,
            "ollama": command("ollama", "--version"),
        },
        "models": {
            "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2_5_revision": first_ref(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/refs/main"),
            "glm_4_9b_chat_hf": "THUDM/glm-4-9b-chat-hf",
            "glm_4_revision": first_ref(ROOT.parent / "hf_cache/hub/models--THUDM--glm-4-9b-chat-hf/refs/main"),
            "rag_reference": settings.rag_model,
            "rag_reference_inventory": command("ollama", "list"),
            "embedding": settings.embedding_model,
            "embedding_revision": first_ref(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/refs/main"),
            "comet": "Unbabel/wmt22-comet-da",
            "comet_revision": "2760a223ac957f30acfb18c8aa649b01cf1d75f2",
        },
        "experiment": experiment,
        "database": db_stats,
        "inputs": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in tracked_inputs
        },
    }
    output = output_dir / "experiment_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
