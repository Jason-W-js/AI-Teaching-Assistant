from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PageDocument:
    text: str
    source: str
    page: int
    chapter: str
    section: str
    doc_type: str = "textbook"


@dataclass
class TextChunk:
    id: str
    text: str
    source: str
    chapter: str
    section: str
    page_start: int | None
    page_end: int | None
    doc_type: str
    knowledge_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk: TextChunk
    score: float
    vector_score: float
    bm25_score: float
    rerank_score: float

    def source_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "chapter": self.chunk.chapter,
            "section": self.chunk.section,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "score": round(self.score, 4),
            "doc_type": self.chunk.doc_type,
        }

