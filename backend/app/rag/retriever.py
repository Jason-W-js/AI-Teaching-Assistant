from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.app.rag.models import RetrievalHit, TextChunk


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
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()

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

    @staticmethod
    def _normalize(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        minimum, maximum = min(values.values()), max(values.values())
        if math.isclose(minimum, maximum):
            return {key: 1.0 if maximum > 0 else 0.0 for key in values}
        return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}

    def search(self, query: str, k: int = 6, prefer_questions: bool = False) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        candidate_count = min(len(self.chunks), max(k * 4, 16))
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
        if prefer_questions:
            candidates.update(
                index for index, chunk in enumerate(self.chunks) if chunk.doc_type == "question"
            )
        query_tokens = set(tokenize(query))
        hits: list[RetrievalHit] = []
        for index in candidates:
            chunk = self.chunks[index]
            chunk_tokens = set(self._tokenized[index])
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            tag_overlap = len(query_tokens & set(tokenize(" ".join(chunk.knowledge_tags)))) / max(1, len(query_tokens))
            type_bonus = 0.22 if prefer_questions and chunk.doc_type == "question" else 0.0
            rerank = (
                0.52 * vector_norm.get(index, 0.0)
                + 0.30 * bm25_norm.get(index, 0.0)
                + 0.10 * overlap
                + 0.08 * tag_overlap
                + type_bonus
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
            question_hits = [hit for hit in hits if hit.chunk.doc_type == "question"][: min(3, k)]
            textbook_hits = [hit for hit in hits if hit.chunk.doc_type != "question"][: max(0, k - len(question_hits))]
            return question_hits + textbook_hits
        return hits[:k]
