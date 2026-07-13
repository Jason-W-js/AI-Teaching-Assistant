from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.app.rag.models import RetrievalHit, TextChunk


DOMAIN_ANCHORS = (
    "PN结", "稳压二极管", "二极管", "晶体管", "三极管", "场效应管", "运算放大器",
    "节点电压法", "回路电流法", "网孔电流法", "戴维南", "诺顿", "叠加定理",
    "RC电路", "RL电路", "RLC", "相量", "功率因数", "传递函数", "频率响应",
)


def _query_anchors(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    selected = [anchor for anchor in DOMAIN_ANCHORS if anchor.lower() in lowered]
    return tuple(
        anchor
        for anchor in selected
        if not any(anchor.lower() in other.lower() for other in selected if other != anchor)
    )


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


class HybridRetriever:
    def __init__(self, index_dir: Path, embedding_model_path: Path) -> None:
        self.index_dir = index_dir
        self.embedding_model_path = embedding_model_path
        # Use Python file I/O so Unicode workspace paths work on Windows.
        serialized_index = np.frombuffer(
            (index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8
        )
        self.index = faiss.deserialize_index(serialized_index)
        self.chunks = [
            TextChunk(**json.loads(line))
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        self._tokenized = [tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self.question_chunks = self._load_question_chunks(index_dir / "question_bank.json")
        self._question_tokenized = [
            tokenize(self._search_text(chunk)) for chunk in self.question_chunks
        ]
        self._question_bm25 = (
            BM25Okapi(self._question_tokenized) if self._question_tokenized else None
        )
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._question_embeddings: np.ndarray | None = None
        self._question_embedding_lock = threading.Lock()

    @staticmethod
    def _load_question_chunks(path: Path) -> list[TextChunk]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            return []
        chunks: list[TextChunk] = []
        for item in payload.get("questions", []):
            if not isinstance(item, dict) or not str(item.get("question_text", "")).strip():
                continue
            tags = [str(value).strip() for value in item.get("knowledge_tags", []) if str(value).strip()]
            question_id = str(item.get("question_id") or len(chunks) + 1)
            chunks.append(
                TextChunk(
                    id=f"question-{question_id}",
                    text=(
                        f"题目：{item.get('question_text', '')}\n"
                        f"标准答案：{item.get('standard_answer', '')}\n"
                        f"解题步骤：{item.get('solution_steps', '')}\n"
                        f"易错点：{item.get('common_mistakes', '')}"
                    ).strip(),
                    source=str(item.get("source", "question_bank.json")),
                    chapter="示例题库",
                    section="、".join(tags) or str(item.get("question_type", "综合题")),
                    page_start=None,
                    page_end=None,
                    doc_type="question",
                    knowledge_tags=tags,
                )
            )
        return chunks

    @staticmethod
    def _search_text(chunk: TextChunk) -> str:
        return " ".join(
            [chunk.chapter, chunk.section, " ".join(chunk.knowledge_tags), chunk.text]
        )

    def _embedding_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(str(self.embedding_model_path), device="cpu")
        return self._model

    def _get_question_embeddings(self) -> np.ndarray:
        if self._question_embeddings is None:
            with self._question_embedding_lock:
                if self._question_embeddings is None:
                    if not self.question_chunks:
                        dimension = int(self.meta.get("dimension", 384))
                        self._question_embeddings = np.empty((0, dimension), dtype=np.float32)
                    else:
                        self._question_embeddings = self._embedding_model().encode(
                            [self._search_text(chunk) for chunk in self.question_chunks],
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        ).astype(np.float32)
        return self._question_embeddings

    @staticmethod
    def _normalize(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        minimum, maximum = min(values.values()), max(values.values())
        if math.isclose(minimum, maximum):
            return {key: 1.0 if maximum > 0 else 0.0 for key in values}
        return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}

    @staticmethod
    def _diversify_sources(hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        """Keep relevance first while ensuring credible books are not crowded out."""
        if len(hits) <= 1 or k <= 1:
            return hits[:k]
        selected = [hits[0]]
        used_ids = {hits[0].chunk.id}
        first_source = hits[0].chunk.source
        threshold = max(0.18, hits[0].score * 0.42)
        best_by_source: dict[str, RetrievalHit] = {}
        for hit in hits[1:]:
            if hit.chunk.source == first_source or hit.score < threshold:
                continue
            best_by_source.setdefault(hit.chunk.source, hit)
        for hit in sorted(best_by_source.values(), key=lambda item: item.score, reverse=True)[:2]:
            if len(selected) >= k:
                break
            selected.append(hit)
            used_ids.add(hit.chunk.id)
        source_counts = Counter(hit.chunk.source for hit in selected)
        for hit in hits:
            if len(selected) >= k:
                break
            if hit.score < threshold:
                continue
            if hit.chunk.id in used_ids:
                continue
            # Avoid one book occupying every visible citation when another
            # credible source exists, but never force a low-relevance result.
            if source_counts[hit.chunk.source] >= 3 and len(best_by_source) > 0:
                continue
            selected.append(hit)
            used_ids.add(hit.chunk.id)
            source_counts[hit.chunk.source] += 1
        return selected[:k]

    def _question_hits(
        self, query: str, query_embedding: np.ndarray, k: int
    ) -> list[RetrievalHit]:
        if not self.question_chunks or self._question_bm25 is None or k <= 0:
            return []
        vector_values = self._get_question_embeddings() @ query_embedding[0]
        bm25_values = self._question_bm25.get_scores(tokenize(query))
        vector_norm = self._normalize(
            {index: float(value) for index, value in enumerate(vector_values)}
        )
        bm25_norm = self._normalize(
            {index: float(value) for index, value in enumerate(bm25_values)}
        )
        query_tokens = set(tokenize(query))
        hits: list[RetrievalHit] = []
        for index, chunk in enumerate(self.question_chunks):
            chunk_tokens = set(self._question_tokenized[index])
            tag_tokens = set(tokenize(" ".join(chunk.knowledge_tags)))
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            tag_overlap = len(query_tokens & tag_tokens) / max(1, len(query_tokens))
            # The compact multilingual embedding model can over-associate broad
            # electronics concepts. A question-bank candidate therefore needs
            # lexical or knowledge-tag evidence before it may guide generation.
            if bm25_values[index] <= 0 and tag_overlap <= 0:
                continue
            score = min(
                1.0,
                0.35 * vector_norm.get(index, 0.0)
                + 0.40 * bm25_norm.get(index, 0.0)
                + 0.15 * overlap
                + 0.10 * tag_overlap,
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    vector_score=float(vector_values[index]),
                    bm25_score=float(bm25_values[index]),
                    rerank_score=score,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def search(self, query: str, k: int = 6, prefer_questions: bool = False) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        candidate_count = min(len(self.chunks), max(k * 12, 64))
        query_embedding = self._embedding_model().encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        vector_scores, vector_indices = self.index.search(query_embedding, candidate_count)
        vector_map = {
            int(index): float(score)
            for score, index in zip(vector_scores[0], vector_indices[0])
            if index >= 0
        }
        bm25_values = self._bm25.get_scores(tokenize(query))
        bm25_top = np.argsort(bm25_values)[::-1][:candidate_count]
        bm25_map = {int(index): float(bm25_values[index]) for index in bm25_top}

        vector_norm = self._normalize(vector_map)
        bm25_norm = self._normalize(bm25_map)
        candidates = set(vector_map) | set(bm25_map)
        query_tokens = set(tokenize(query))
        query_anchors = _query_anchors(query)
        hits: list[RetrievalHit] = []
        for index in candidates:
            chunk = self.chunks[index]
            searchable = self._search_text(chunk).lower()
            if query_anchors and not any(anchor.lower() in searchable for anchor in query_anchors):
                continue
            chunk_tokens = set(self._tokenized[index])
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            tag_overlap = len(query_tokens & set(tokenize(" ".join(chunk.knowledge_tags)))) / max(1, len(query_tokens))
            rerank = (
                0.52 * vector_norm.get(index, 0.0)
                + 0.30 * bm25_norm.get(index, 0.0)
                + 0.10 * overlap
                + 0.08 * tag_overlap
            )
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=rerank,
                    vector_score=vector_map.get(index, 0.0),
                    bm25_score=bm25_map.get(index, 0.0),
                    rerank_score=rerank,
                )
            )
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        if prefer_questions:
            question_hits = self._question_hits(query, query_embedding, min(3, k))
            textbook_hits = self._diversify_sources(hits, max(0, k - len(question_hits)))
            return question_hits + textbook_hits
        return self._diversify_sources(hits, k)
