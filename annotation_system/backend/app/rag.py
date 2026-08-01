from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock
from urllib.request import Request, urlopen

from sqlalchemy import select

from .config import settings
from .database import SessionLocal
from .models import CorpusItem, KnowledgeChunkEmbedding


SPACE_RE = re.compile(r"\s+")
LATIN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RagDocument:
    item_id: int
    text: str
    evidence: str
    source_document: str
    source_type: str
    task_type: str
    domain_category: str
    chapter: str
    page_number: int | None
    review_status: str


def tokenize(text: str) -> list[str]:
    normalized = SPACE_RE.sub(" ", text.lower()).strip()
    han = "".join(HAN_RE.findall(normalized))
    tokens = list(han)
    tokens.extend(han[index : index + 2] for index in range(max(0, len(han) - 1)))
    tokens.extend(LATIN_RE.findall(normalized))
    return tokens


class RailwayRetriever:
    def __init__(self) -> None:
        self.documents: list[RagDocument] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.average_length = 0.0
        self.built_at = 0.0
        self._lock = Lock()

    def build(self) -> int:
        with self._lock:
            documents: list[RagDocument] = []
            seen_evidence: set[str] = set()
            with SessionLocal() as db:
                rows = db.scalars(
                    select(CorpusItem)
                    .where(
                        CorpusItem.source_type != "generated_eval_review",
                        CorpusItem.review_status.notin_(["rejected", "needs_revision", "deleted"]),
                    )
                    .order_by(CorpusItem.id)
                )
                for item in rows:
                    metadata = item.metadata_json or {}
                    if metadata.get("split") == "test" or metadata.get("rag_test_set"):
                        continue
                    evidence = SPACE_RE.sub(" ", item.evidence or item.source_text or item.answer).strip()
                    if len(evidence) < 4:
                        continue
                    evidence_key = f"{item.source_document}\x1f{item.page_number}\x1f{evidence}"
                    if evidence_key in seen_evidence:
                        continue
                    seen_evidence.add(evidence_key)
                    searchable = " ".join(
                        value
                        for value in [
                            item.question,
                            item.question_en,
                            item.answer,
                            item.answer_en,
                            evidence,
                            item.domain_category,
                            item.knowledge_category,
                            item.chapter,
                        ]
                        if value
                    )
                    documents.append(
                        RagDocument(
                            item_id=item.id,
                            text=searchable,
                            evidence=evidence[:2400],
                            source_document=item.source_document,
                            source_type=item.source_type,
                            task_type=item.task_type,
                            domain_category=item.domain_category,
                            chapter=item.chapter,
                            page_number=item.page_number,
                            review_status=item.review_status,
                        )
                    )

            term_frequencies: list[Counter[str]] = []
            document_frequencies: Counter[str] = Counter()
            postings: dict[str, list[int]] = defaultdict(list)
            total_length = 0
            for index, document in enumerate(documents):
                frequencies = Counter(tokenize(document.text))
                term_frequencies.append(frequencies)
                total_length += sum(frequencies.values())
                for token in frequencies:
                    document_frequencies[token] += 1
                    postings[token].append(index)

            self.documents = documents
            self.term_frequencies = term_frequencies
            self.document_frequencies = document_frequencies
            self.postings = postings
            self.average_length = total_length / max(1, len(documents))
            self.built_at = time.time()
            return len(documents)

    def ensure_ready(self) -> None:
        if not self.documents:
            self.build()

    def search(self, query: str, top_k: int = 5, approved_only: bool = False) -> list[dict]:
        self.ensure_ready()
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(self.postings.get(token, []))

        total_documents = len(self.documents)
        k1 = 1.5
        b = 0.75
        scores: list[tuple[float, int]] = []
        for index in candidates:
            frequencies = self.term_frequencies[index]
            document_length = sum(frequencies.values())
            score = 0.0
            for token, query_frequency in query_tokens.items():
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies[token]
                inverse_document_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * document_length / max(1.0, self.average_length)
                )
                score += inverse_document_frequency * frequency * (k1 + 1) / denominator
                score *= 1 + min(query_frequency, 3) * 0.05
            if score > 0:
                scores.append((score, index))

        results = []
        for score, index in sorted(scores, reverse=True)[:top_k]:
            document = self.documents[index]
            if approved_only and document.review_status != "approved":
                continue
            results.append(
                {
                    "item_id": document.item_id,
                    "score": round(score, 4),
                    "evidence": document.evidence,
                    "source_document": document.source_document,
                    "source_type": document.source_type,
                    "task_type": document.task_type,
                    "domain_category": document.domain_category,
                    "chapter": document.chapter,
                    "page_number": document.page_number,
                    "review_status": document.review_status,
                    "retrieval_mode": "bm25",
                }
            )
        return results


retriever = RailwayRetriever()


_embedding_model = None
_embedding_lock = Lock()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                _embedding_model = SentenceTransformer(
                    settings.embedding_model,
                    local_files_only=settings.embedding_local_files_only,
                )
    return _embedding_model


def vector_search(query: str, top_k: int = 5, approved_only: bool = False) -> list[dict]:
    model = get_embedding_model()
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    distance = KnowledgeChunkEmbedding.embedding.cosine_distance(query_vector)
    with SessionLocal() as db:
        statement = (
            select(KnowledgeChunkEmbedding, CorpusItem, distance.label("distance"))
            .join(CorpusItem, CorpusItem.id == KnowledgeChunkEmbedding.item_id)
            .where(KnowledgeChunkEmbedding.embedding_model == settings.embedding_model)
            .where(CorpusItem.review_status.notin_(["rejected", "needs_revision", "deleted"]))
            .where(CorpusItem.metadata_json["rag_test_set"].as_string().is_(None))
            .order_by(distance)
            .limit(top_k)
        )
        if approved_only:
            statement = statement.where(CorpusItem.review_status == "approved")
        rows = db.execute(statement).all()

    results = []
    for chunk, item, score_distance in rows:
        evidence = SPACE_RE.sub(" ", item.evidence or item.source_text or item.answer or chunk.chunk_text).strip()
        similarity = 1.0 - float(score_distance)
        results.append(
            {
                "item_id": item.id,
                "score": round(similarity, 4),
                "evidence": evidence[:2400],
                "source_document": item.source_document,
                "source_type": item.source_type,
                "task_type": item.task_type,
                "domain_category": item.domain_category,
                "chapter": item.chapter,
                "page_number": item.page_number,
                "review_status": item.review_status,
                "retrieval_mode": "vector",
            }
        )
    return results


def hybrid_search(query: str, top_k: int = 5, approved_only: bool = False) -> list[dict]:
    # Keep the fusion pool fixed so top-k sensitivity changes only the final cutoff.
    candidate_k = max(top_k, 50)
    bm25_results = retriever.search(query, candidate_k, approved_only=approved_only)
    vector_results = vector_search(query, candidate_k, approved_only=approved_only)
    rrf_k = 60
    bm25_scores = {int(result["item_id"]): 1.0 / (rrf_k + rank) for rank, result in enumerate(bm25_results, 1)}
    vector_scores = {int(result["item_id"]): 1.0 / (rrf_k + rank) for rank, result in enumerate(vector_results, 1)}
    merged: dict[int, dict] = {}

    for result in bm25_results:
        item_id = int(result["item_id"])
        merged[item_id] = {**result, "bm25_score": result["score"], "vector_score": 0.0}
    for result in vector_results:
        item_id = int(result["item_id"])
        if item_id in merged:
            merged[item_id]["vector_score"] = result["score"]
        else:
            merged[item_id] = {**result, "bm25_score": 0.0, "vector_score": result["score"]}

    ranked = []
    for item_id, result in merged.items():
        fused = 0.5 * bm25_scores.get(item_id, 0.0) + 0.5 * vector_scores.get(item_id, 0.0)
        result["score"] = round(fused, 4)
        result["retrieval_mode"] = "hybrid"
        ranked.append(result)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]


def generate_with_ollama(question: str, sources: list[dict]) -> str:
    context_blocks = []
    for index, source in enumerate(sources, 1):
        location = source["source_document"]
        if source.get("page_number"):
            location += f"，第{source['page_number']}页"
        context_blocks.append(f"[证据{index}] 来源：{location}\n{source['evidence']}")
    context = "\n\n".join(context_blocks)
    payload = {
        "model": settings.rag_model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是铁道教育领域问答助手。只能依据提供的证据回答，不得补充证据中没有的事实。"
                    "证据不足时明确回答“现有语料无法支持该问题的确定答案”。"
                    "回答应简洁，并在相关句子后使用[证据1]这样的编号标注来源。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n可用证据：\n{context}",
            },
        ],
        "options": {"temperature": 0.1, "num_predict": 420},
    }
    request = Request(
        f"{settings.ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=settings.rag_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("message", {}).get("content", "").strip()


async def answer_question(
    question: str,
    top_k: int,
    generate: bool,
    retrieval_mode: str = "bm25",
    approved_only: bool = False,
) -> dict:
    retrieval_started = time.perf_counter()
    if retrieval_mode == "vector":
        sources = await asyncio.to_thread(vector_search, question, top_k, approved_only)
    elif retrieval_mode == "hybrid":
        sources = await asyncio.to_thread(hybrid_search, question, top_k, approved_only)
    else:
        sources = await asyncio.to_thread(retriever.search, question, top_k, approved_only)
    retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 1)
    if not sources:
        return {
            "answer": "现有语料中未检索到与该问题相关的证据。",
            "mode": "no_evidence",
            "model": None,
            "sources": [],
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0.0,
        }

    if not generate:
        return {
            "answer": sources[0]["evidence"],
            "mode": "retrieval_only",
            "model": None,
            "sources": sources,
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0.0,
        }

    generation_started = time.perf_counter()
    try:
        answer = await asyncio.to_thread(generate_with_ollama, question, sources)
        mode = "rag"
        model = settings.rag_model
    except Exception as error:
        answer = (
            "本地生成模型暂时不可用，以下为最相关的原始证据：\n"
            + sources[0]["evidence"]
        )
        mode = "retrieval_fallback"
        model = None
        sources[0]["generation_error"] = str(error)
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 1)
    return {
        "answer": answer,
        "mode": mode,
        "model": model,
        "sources": sources,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }
