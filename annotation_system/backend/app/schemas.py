from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


ReviewStatus = Literal["pending", "approved", "rejected", "needs_revision", "deleted"]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    source_path: str
    document_type: str
    domain_category: str
    total_pages: int | None


class CorpusItemBase(BaseModel):
    task_type: str
    domain_category: str = ""
    knowledge_category: str = ""
    question: str = ""
    answer: str = ""
    question_en: str = ""
    answer_en: str = ""
    evidence: str = ""
    source_text: str = ""
    source_document: str = ""
    source_path: str = ""
    chapter: str = ""
    page_number: int | None = None
    quality_flags: list[str] = Field(default_factory=list)
    reviewer: str = ""
    review_comment: str = ""
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class CorpusItemUpdate(BaseModel):
    task_type: str | None = None
    domain_category: str | None = None
    knowledge_category: str | None = None
    question: str | None = None
    answer: str | None = None
    question_en: str | None = None
    answer_en: str | None = None
    evidence: str | None = None
    source_text: str | None = None
    chapter: str | None = None
    page_number: int | None = None
    quality_flags: list[str] | None = None
    reviewer: str | None = None
    review_comment: str | None = None
    metadata_json: dict[str, Any] | None = None


class CorpusItemCreate(CorpusItemBase):
    source_type: str = "manual"
    review_status: ReviewStatus = "pending"
    source_document: str = "manual"
    source_path: str = "manual"
    external_id: str | None = None
    document_id: int | None = None
    original_question: str = ""
    original_answer: str = ""


class ReviewRequest(BaseModel):
    status: ReviewStatus
    reviewer: str = ""
    comment: str = ""


class BatchReviewRequest(ReviewRequest):
    item_ids: list[int] = Field(min_length=1, max_length=500)


class BatchReviewResponse(BaseModel):
    updated: int
    missing_ids: list[int] = Field(default_factory=list)


class CorpusItemOut(CorpusItemBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    source_type: str
    review_status: ReviewStatus
    original_question: str
    original_answer: str
    version: int
    document: DocumentOut | None = None
    created_at: datetime
    updated_at: datetime

    @computed_field
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


class CorpusItemList(BaseModel):
    items: list[CorpusItemOut]
    total: int
    page: int
    page_size: int


class StatsOut(BaseModel):
    total: int
    by_status: dict[str, int]
    by_task_type: dict[str, int]
    by_domain_category: dict[str, int]


class RagAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    generate: bool = True
    session_id: int | None = None


class RagSource(BaseModel):
    item_id: int
    score: float
    evidence: str
    source_document: str
    source_type: str
    task_type: str
    domain_category: str
    chapter: str
    page_number: int | None
    review_status: str
    generation_error: str | None = None


class RagAnswer(BaseModel):
    session_id: int | None = None
    user_message_id: int | None = None
    assistant_message_id: int | None = None
    answer: str
    mode: str
    model: str | None
    sources: list[RagSource]
    retrieval_ms: float
    generation_ms: float


class RagSessionCreate(BaseModel):
    title: str = Field(default="新会话", max_length=300)


class RagSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class RagMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: float = Field(default=1.0, ge=0.5, le=1.5)


class TtsResponse(BaseModel):
    audio_url: str


class AsrResponse(BaseModel):
    text: str
