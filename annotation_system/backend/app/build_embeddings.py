from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable

from pgvector.sqlalchemy import Vector
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from .config import settings
from .database import SessionLocal
from .models import CorpusItem, KnowledgeChunkEmbedding


EXCLUDED_STATUSES = {"rejected", "needs_revision", "deleted"}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def item_to_chunk_text(item: CorpusItem) -> str:
    parts = [
        item.question,
        item.question_en,
        item.answer,
        item.answer_en,
        item.evidence,
        item.source_text,
        item.domain_category,
        item.knowledge_category,
        item.chapter,
    ]
    return normalize_text("\n".join(part for part in parts if part))


def iter_items(*, approved_only: bool, include_test: bool, limit: int | None) -> Iterable[CorpusItem]:
    with SessionLocal() as db:
        filters = [CorpusItem.review_status.notin_(EXCLUDED_STATUSES)]
        if approved_only:
            filters = [CorpusItem.review_status == "approved"]
        query = select(CorpusItem).where(*filters).order_by(CorpusItem.id)
        if limit:
            query = query.limit(limit)
        for item in db.scalars(query):
            metadata = item.metadata_json or {}
            if not include_test and (metadata.get("split") == "test" or metadata.get("rag_test_set")):
                continue
            yield item


def write_batch(model, model_name: str, batch: list[tuple[CorpusItem, str]]) -> int:
    texts = [chunk_text for _item, chunk_text in batch]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    rows = []
    for (item, chunk_text), vector in zip(batch, vectors, strict=True):
        rows.append(
            {
                "item_id": item.id,
                "chunk_index": 0,
                "chunk_text": chunk_text,
                "embedding_model": model_name,
                "embedding": vector.tolist(),
                "metadata_json": {
                    "source_type": item.source_type,
                    "task_type": item.task_type,
                    "review_status": item.review_status,
                    "source_document": item.source_document,
                    "page_number": item.page_number,
                },
            }
        )

    statement = (
        insert(KnowledgeChunkEmbedding)
        .values(rows)
        .on_conflict_do_update(
            constraint="uq_chunk_embedding_model",
            set_={
                "chunk_text": insert(KnowledgeChunkEmbedding).excluded.chunk_text,
                "embedding": insert(KnowledgeChunkEmbedding).excluded.embedding.cast(
                    Vector(settings.embedding_dimension)
                ),
                "metadata_json": insert(KnowledgeChunkEmbedding).excluded.metadata_json,
            },
        )
    )
    with SessionLocal() as db:
        db.execute(statement)
        db.commit()
    return len(rows)


def hash_vector(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}\x1f{text}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return values[:dimension]


def write_hash_batch(model_name: str, batch: list[tuple[CorpusItem, str]]) -> int:
    rows = []
    for item, chunk_text in batch:
        rows.append(
            {
                "item_id": item.id,
                "chunk_index": 0,
                "chunk_text": chunk_text,
                "embedding_model": model_name,
                "embedding": hash_vector(chunk_text, settings.embedding_dimension),
                "metadata_json": {
                    "source_type": item.source_type,
                    "task_type": item.task_type,
                    "review_status": item.review_status,
                    "source_document": item.source_document,
                    "page_number": item.page_number,
                    "debug_embedding": True,
                },
            }
        )

    statement = (
        insert(KnowledgeChunkEmbedding)
        .values(rows)
        .on_conflict_do_update(
            constraint="uq_chunk_embedding_model",
            set_={
                "chunk_text": insert(KnowledgeChunkEmbedding).excluded.chunk_text,
                "embedding": insert(KnowledgeChunkEmbedding).excluded.embedding.cast(
                    Vector(settings.embedding_dimension)
                ),
                "metadata_json": insert(KnowledgeChunkEmbedding).excluded.metadata_json,
            },
        )
    )
    with SessionLocal() as db:
        db.execute(statement)
        db.commit()
    return len(rows)


def build_embeddings(
    *,
    backend: str,
    model_name: str | None,
    batch_size: int,
    approved_only: bool,
    include_test: bool,
    limit: int | None,
    rebuild: bool,
) -> int:
    model_name = model_name or settings.embedding_model
    if backend == "hash":
        model = None
        model_name = f"debug-hash-{settings.embedding_dimension}"
        print(f"using_debug_embedding_model={model_name}")
    else:
        from sentence_transformers import SentenceTransformer

        print(f"loading_embedding_model={model_name}")
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        if dimension != settings.embedding_dimension:
            raise ValueError(
                f"Embedding model {model_name} returns {dimension} dimensions, "
                f"but RAILWAY_EMBEDDING_DIMENSION={settings.embedding_dimension}."
            )

    if rebuild:
        with SessionLocal() as db:
            db.execute(
                delete(KnowledgeChunkEmbedding).where(KnowledgeChunkEmbedding.embedding_model == model_name)
            )
            db.commit()

    pending: list[tuple[CorpusItem, str]] = []
    written = 0
    for item in iter_items(approved_only=approved_only, include_test=include_test, limit=limit):
        chunk_text = item_to_chunk_text(item)
        if len(chunk_text) < 4:
            continue
        pending.append((item, chunk_text))
        if len(pending) >= batch_size:
            written += write_hash_batch(model_name, pending) if backend == "hash" else write_batch(model, model_name, pending)
            pending.clear()
    if pending:
        written += write_hash_batch(model_name, pending) if backend == "hash" else write_batch(model, model_name, pending)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pgvector embeddings for railway education chunks.")
    parser.add_argument("--backend", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=None, help="Override RAILWAY_EMBEDDING_MODEL for this run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = build_embeddings(
        backend=args.backend,
        model_name=args.model,
        batch_size=args.batch_size,
        approved_only=args.approved_only,
        include_test=args.include_test,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    print(f"embedded_chunks={count}")


if __name__ == "__main__":
    main()
