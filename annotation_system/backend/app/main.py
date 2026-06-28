from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from .config import settings
from .asr import ASR_UPLOAD_DIR, transcribe_audio
from .database import Base, engine, get_db
from .models import CorpusItem, Document, ReviewEvent
from .rag import answer_question, retriever
from .tts import TTS_CACHE_DIR, synthesize_speech
from .schemas import (
    BatchReviewRequest,
    BatchReviewResponse,
    CorpusItemList,
    CorpusItemCreate,
    CorpusItemOut,
    CorpusItemUpdate,
    DocumentOut,
    ReviewRequest,
    RagAnswer,
    RagAskRequest,
    StatsOut,
    TtsRequest,
    TtsResponse,
    AsrResponse,
)


app = FastAPI(title="Railway Education Corpus Review API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REVIEW_TASK_TYPES = [
    "concept_explanation_qa",
    "regulation_clause_qa",
    "regulation_definition_qa",
    "regulation_extractive_qa",
    "regulation_inspection_qa",
    "regulation_judgment",
    "regulation_principle_qa",
    "regulation_prohibition_qa",
    "regulation_requirement_qa",
    "regulation_responsibility_qa",
    "regulation_standard_qa",
    "terminology_explanation",
    "terminology_pair",
    "terminology_translation",
    "textbook_definition_qa",
    "textbook_extractive_qa",
    "textbook_judgment",
    "textbook_multiple_choice",
    "textbook_operation_qa",
    "textbook_source",
]


@app.on_event("startup")
def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tts", response_model=TtsResponse)
async def text_to_speech(payload: TtsRequest) -> TtsResponse:
    try:
        audio_path = await synthesize_speech(payload.text, payload.voice, payload.rate)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"语音生成失败：{error}") from error
    return TtsResponse(audio_url=f"/api/tts/audio/{audio_path.name}")


@app.get("/api/tts/audio/{filename}")
def get_tts_audio(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or not filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    audio_path = TTS_CACHE_DIR / filename
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.post("/api/asr", response_model=AsrResponse)
async def speech_to_text(audio: UploadFile = File(...), language: str = "zh") -> AsrResponse:
    suffix = Path(audio.filename or "speech.webm").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".ogg"}:
        suffix = ".webm"

    ASR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = ASR_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with audio_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        text = transcribe_audio(audio_path, language)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"语音识别失败：{error}") from error
    finally:
        audio.file.close()
        audio_path.unlink(missing_ok=True)
    return AsrResponse(text=text)


@app.get("/api/rag/stats")
def rag_stats() -> dict:
    retriever.ensure_ready()
    return {
        "documents": len(retriever.documents),
        "built_at": retriever.built_at,
        "model": settings.rag_model,
        "excludes_test_split": True,
    }


@app.post("/api/rag/rebuild")
def rag_rebuild() -> dict:
    return {"documents": retriever.build(), "built_at": retriever.built_at}


@app.post("/api/rag/ask", response_model=RagAnswer)
async def rag_ask(payload: RagAskRequest) -> dict:
    return await answer_question(payload.question.strip(), payload.top_k, payload.generate)


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.title)))


@app.get("/api/files")
def get_local_file(path: str) -> FileResponse:
    repo_root = Path(__file__).resolve().parents[3]
    raw_path = Path(unquote(path))
    file_path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = file_path.resolve()
    if repo_root not in resolved.parents and resolved != repo_root:
        raise HTTPException(status_code=403, detail="File is outside repository")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(resolved)


@app.get("/api/ocr-assets")
def get_ocr_asset(src: str, markdown_path: str = "", page: int | None = None) -> FileResponse:
    repo_root = Path(__file__).resolve().parents[3]
    raw_src = Path(unquote(src))
    candidates: list[Path] = []

    if raw_src.is_absolute():
        candidates.append(raw_src)
    else:
        candidates.append(repo_root / raw_src)

    if markdown_path:
        raw_markdown_path = Path(unquote(markdown_path))
        markdown_file = raw_markdown_path if raw_markdown_path.is_absolute() else repo_root / raw_markdown_path
        markdown_dir = markdown_file.parent
        candidates.append(markdown_dir / raw_src)

        markdown_text = str(markdown_file)
        book_dir = markdown_dir.parent if markdown_dir.name == "pages" else markdown_dir
        if "/pages/" in markdown_text:
            book_dir = Path(markdown_text.split("/pages/", 1)[0])
        if page is not None and str(src).replace("\\", "/").startswith("images/"):
            candidates.append(book_dir / "deepseek_outputs" / f"page_{page:04d}" / raw_src)
        candidates.append(book_dir / raw_src)

    for candidate in candidates:
        resolved = candidate.resolve()
        if repo_root not in resolved.parents and resolved != repo_root:
            continue
        if resolved.is_file():
            return FileResponse(resolved)

    raise HTTPException(status_code=404, detail="OCR asset not found")


@app.get("/api/items", response_model=CorpusItemList)
def list_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=500),
    status: str = "",
    task_type: str = "",
    domain_category: str = "",
    document_id: int | None = None,
    search: str = "",
    db: Session = Depends(get_db),
) -> CorpusItemList:
    filters = item_filters(
        status=status,
        task_type=task_type,
        domain_category=domain_category,
        document_id=document_id,
        search=search,
    )

    total = db.scalar(select(func.count()).select_from(CorpusItem).where(*filters)) or 0
    statement = (
        select(CorpusItem)
        .options(selectinload(CorpusItem.document))
        .where(*filters)
        .order_by(CorpusItem.updated_at.desc(), CorpusItem.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list(db.scalars(statement))
    return CorpusItemList(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/items/{item_id}", response_model=CorpusItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)) -> CorpusItem:
    item = db.scalar(
        select(CorpusItem).options(selectinload(CorpusItem.document)).where(CorpusItem.id == item_id)
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


def snapshot(item: CorpusItem) -> dict:
    return {
        "review_status": item.review_status,
        "task_type": item.task_type,
        "domain_category": item.domain_category,
        "knowledge_category": item.knowledge_category,
        "question": item.question,
        "answer": item.answer,
        "question_en": item.question_en,
        "answer_en": item.answer_en,
        "evidence": item.evidence,
        "chapter": item.chapter,
        "page_number": item.page_number,
        "quality_flags": item.quality_flags,
        "review_comment": item.review_comment,
        "metadata_json": item.metadata_json,
        "version": item.version,
    }


@app.post("/api/items", response_model=CorpusItemOut)
def create_item(payload: CorpusItemCreate, db: Session = Depends(get_db)) -> CorpusItem:
    external_id = payload.external_id or f"manual_{uuid.uuid4().hex}"
    if db.scalar(select(CorpusItem.id).where(CorpusItem.external_id == external_id)):
        external_id = f"{external_id}_{uuid.uuid4().hex[:8]}"

    item = CorpusItem(
        external_id=external_id,
        source_type=payload.source_type or "manual",
        task_type=payload.task_type or "grounded_qa",
        review_status=payload.review_status or "pending",
        domain_category=payload.domain_category or "",
        knowledge_category=payload.knowledge_category or "",
        question=payload.question or "",
        answer=payload.answer or "",
        question_en=payload.question_en or "",
        answer_en=payload.answer_en or "",
        evidence=payload.evidence or "",
        source_text=payload.source_text or payload.evidence or "",
        original_question=payload.original_question or payload.question or "",
        original_answer=payload.original_answer or payload.answer or "",
        document_id=payload.document_id,
        source_document=payload.source_document or "manual",
        source_path=payload.source_path or "manual",
        chapter=payload.chapter or "",
        page_number=payload.page_number,
        quality_flags=payload.quality_flags or [],
        reviewer=payload.reviewer or "",
        review_comment=payload.review_comment or "",
        metadata_json=payload.metadata_json or {"created_from": "manual"},
    )
    db.add(item)
    db.flush()
    db.add(
        ReviewEvent(
            item=item,
            action="create",
            reviewer=payload.reviewer or "system",
            comment=payload.review_comment or "Created from review UI.",
            snapshot={"after": snapshot(item)},
        )
    )
    db.commit()
    return get_item(item.id, db)


@app.patch("/api/items/{item_id}", response_model=CorpusItemOut)
def update_item(item_id: int, payload: CorpusItemUpdate, db: Session = Depends(get_db)) -> CorpusItem:
    item = db.get(CorpusItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    before = snapshot(item)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    item.version += 1
    db.add(
        ReviewEvent(
            item=item,
            action="edit",
            reviewer=payload.reviewer or item.reviewer,
            comment=payload.review_comment or "",
            snapshot={"before": before},
        )
    )
    db.commit()
    return get_item(item_id, db)


@app.post("/api/items/batch/review", response_model=BatchReviewResponse)
def review_items_batch(payload: BatchReviewRequest, db: Session = Depends(get_db)) -> BatchReviewResponse:
    unique_ids = list(dict.fromkeys(payload.item_ids))
    items = db.scalars(select(CorpusItem).where(CorpusItem.id.in_(unique_ids))).all()
    found_ids = {item.id for item in items}
    missing_ids = [item_id for item_id in unique_ids if item_id not in found_ids]

    for item in items:
        before = snapshot(item)
        item.review_status = payload.status
        item.reviewer = payload.reviewer
        item.review_comment = payload.comment
        item.version += 1
        db.add(
            ReviewEvent(
                item=item,
                action=f"batch_{payload.status}",
                reviewer=payload.reviewer,
                comment=payload.comment,
                snapshot={"before": before, "after_status": payload.status},
            )
        )
    db.commit()
    return BatchReviewResponse(updated=len(items), missing_ids=missing_ids)



@app.post("/api/items/{item_id}/review", response_model=CorpusItemOut)
def review_item(item_id: int, payload: ReviewRequest, db: Session = Depends(get_db)) -> CorpusItem:
    item = db.get(CorpusItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    before = snapshot(item)
    item.review_status = payload.status
    item.reviewer = payload.reviewer
    item.review_comment = payload.comment
    item.version += 1
    db.add(
        ReviewEvent(
            item=item,
            action=payload.status,
            reviewer=payload.reviewer,
            comment=payload.comment,
            snapshot={"before": before, "after_status": payload.status},
        )
    )
    db.commit()
    return get_item(item_id, db)


@app.delete("/api/items/{item_id}", response_model=CorpusItemOut)
def delete_item(item_id: int, reviewer: str = "", comment: str = "", db: Session = Depends(get_db)) -> CorpusItem:
    item = db.get(CorpusItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    before = snapshot(item)
    item.review_status = "deleted"
    item.reviewer = reviewer or item.reviewer
    item.review_comment = comment or item.review_comment
    item.version += 1
    db.add(
        ReviewEvent(
            item=item,
            action="deleted",
            reviewer=item.reviewer,
            comment=item.review_comment,
            snapshot={"before": before, "after_status": "deleted"},
        )
    )
    db.commit()
    return get_item(item_id, db)


def item_filters(
    *,
    status: str = "",
    task_type: str = "",
    domain_category: str = "",
    document_id: int | None = None,
    search: str = "",
    exclude: str = "",
) -> list:
    filters = []
    if exclude != "status":
        if status:
            filters.append(CorpusItem.review_status == status)
        else:
            filters.append(CorpusItem.review_status != "deleted")
    else:
        filters.append(CorpusItem.review_status != "deleted")

    if task_type and exclude != "task_type":
        filters.append(CorpusItem.task_type == task_type)
    if domain_category and exclude != "domain_category":
        filters.append(CorpusItem.domain_category == domain_category)
    if document_id is not None and exclude != "document_id":
        filters.append(CorpusItem.document_id == document_id)
    if search and exclude != "search":
        pattern = f"%{search.strip()}%"
        filters.append(
            or_(
                CorpusItem.question.ilike(pattern),
                CorpusItem.answer.ilike(pattern),
                CorpusItem.evidence.ilike(pattern),
                CorpusItem.source_text.ilike(pattern),
                CorpusItem.source_document.ilike(pattern),
            )
        )
    return filters


def grouped_counts(db: Session, column, filters: list | None = None) -> dict[str, int]:
    rows = db.execute(
        select(column, func.count())
        .where(*(filters or [CorpusItem.review_status != "deleted"]))
        .group_by(column)
        .order_by(column)
    ).all()
    return {str(key or "未分类"): count for key, count in rows}


@app.get("/api/stats", response_model=StatsOut)
def stats(
    status: str = "",
    task_type: str = "",
    domain_category: str = "",
    document_id: int | None = None,
    search: str = "",
    db: Session = Depends(get_db),
) -> StatsOut:
    current_filters = item_filters(
        status=status,
        task_type=task_type,
        domain_category=domain_category,
        document_id=document_id,
        search=search,
    )
    return StatsOut(
        total=db.scalar(select(func.count()).select_from(CorpusItem).where(*current_filters)) or 0,
        by_status=grouped_counts(
            db,
            CorpusItem.review_status,
            item_filters(
                status=status,
                task_type=task_type,
                domain_category=domain_category,
                document_id=document_id,
                search=search,
                exclude="status",
            ),
        ),
        by_task_type=grouped_counts(
            db,
            CorpusItem.task_type,
            item_filters(
                status=status,
                task_type=task_type,
                domain_category=domain_category,
                document_id=document_id,
                search=search,
                exclude="task_type",
            ),
        ),
        by_domain_category=grouped_counts(
            db,
            CorpusItem.domain_category,
            item_filters(
                status=status,
                task_type=task_type,
                domain_category=domain_category,
                document_id=document_id,
                search=search,
                exclude="domain_category",
            ),
        ),
    )


@app.get("/api/options")
def options(db: Session = Depends(get_db)) -> dict[str, list[str]]:
    def values(column) -> list[str]:
        return [
            value
            for value in db.scalars(
                select(column)
                .where(CorpusItem.review_status != "deleted")
                .distinct()
                .order_by(column)
            )
            if value
        ]

    return {
        "task_types": list(dict.fromkeys([*REVIEW_TASK_TYPES, *values(CorpusItem.task_type)])),
        "domain_categories": values(CorpusItem.domain_category),
        "statuses": ["pending", "needs_revision", "approved", "rejected"],
        "quality_flags": [
            "question_underspecified",
            "answer_incomplete",
            "evidence_mismatch",
            "ocr_error",
            "duplicate",
            "category_error",
            "unsafe_or_uncertain",
        ],
    }


@app.post("/api/export")
def export_items(
    status: str = "approved",
    task_type: str = "",
    db: Session = Depends(get_db),
) -> FileResponse:
    filters = [CorpusItem.review_status != "deleted"]
    if status:
        filters.append(CorpusItem.review_status == status)
    if task_type:
        filters.append(CorpusItem.task_type == task_type)
    rows = db.scalars(select(CorpusItem).where(*filters).order_by(CorpusItem.id))
    output_dir = Path(__file__).resolve().parents[3] / "data" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = task_type or "all"
    path = output_dir / f"railway_corpus_{status}_{suffix}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            record = {
                "id": item.external_id,
                "task_type": item.task_type,
                "domain_category": item.domain_category,
                "knowledge_category": item.knowledge_category,
                "question": item.question,
                "answer": item.answer,
                "evidence": item.evidence,
                "source_document": item.source_document,
                "chapter": item.chapter,
                "page": item.page_number,
                "quality_flags": item.quality_flags,
                "reviewer": item.reviewer,
                "metadata": item.metadata_json,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)
