from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    document_type: Mapped[str] = mapped_column(String(40), nullable=False, default="unknown")
    domain_category: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    total_pages: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list["CorpusItem"]] = relationship(back_populates="document")


class CorpusItem(Base):
    __tablename__ = "corpus_items"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_corpus_items_external_id"),
        Index("ix_corpus_items_status_task", "review_status", "task_type"),
        Index("ix_corpus_items_document_page", "document_id", "page_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(180), nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    task_type: Mapped[str] = mapped_column(String(60), nullable=False)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    domain_category: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    knowledge_category: Mapped[str] = mapped_column(String(160), nullable=False, default="")

    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    question_en: Mapped[str] = mapped_column(Text, nullable=False, default="")
    answer_en: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    original_question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    original_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")

    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    source_document: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chapter: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    page_number: Mapped[int | None] = mapped_column(Integer)

    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    reviewer: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    review_comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document | None] = relationship(back_populates="items")
    review_events: Mapped[list["ReviewEvent"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )

    @property
    def source_image_path(self) -> str:
        metadata = self.metadata_json or {}
        image_path = metadata.get("image") or metadata.get("image_path")
        if image_path:
            return str(image_path)

        ocr_page_path = metadata.get("ocr_page_path") or metadata.get("markdown")
        if not ocr_page_path:
            return ""

        path = str(ocr_page_path)
        if "/pages/" not in path or not path.endswith(".md"):
            return ""
        return path.replace("/pages/", "/images/").removesuffix(".md") + ".png"


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("corpus_items.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    item: Mapped[CorpusItem] = relationship(back_populates="review_events")


class RagSession(Base):
    __tablename__ = "rag_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False, default="新会话")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["RagMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="RagMessage.id"
    )


class RagMessage(Base):
    __tablename__ = "rag_messages"
    __table_args__ = (Index("ix_rag_messages_session_id_id", "session_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("rag_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[RagSession] = relationship(back_populates="messages")
