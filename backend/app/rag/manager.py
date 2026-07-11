from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.rag.pipeline import build_knowledge_base
from backend.app.rag.retriever import HybridRetriever


logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    def __init__(self) -> None:
        self._retrievers: dict[str, HybridRetriever] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def validate_id(knowledge_base: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", knowledge_base):
            raise ValueError("知识库名称仅允许字母、数字、连字符和下划线")
        return knowledge_base

    def resource_dir(self, knowledge_base: str) -> Path:
        knowledge_base = self.validate_id(knowledge_base)
        if knowledge_base == "default":
            return settings.resources_dir
        return settings.resources_dir / "knowledge_bases" / knowledge_base

    def index_dir(self, knowledge_base: str) -> Path:
        return settings.vector_stores_dir / self.validate_id(knowledge_base)

    def load_existing(self) -> None:
        settings.vector_stores_dir.mkdir(parents=True, exist_ok=True)
        for index_dir in settings.vector_stores_dir.iterdir():
            if not index_dir.is_dir():
                continue
            knowledge_base = index_dir.name
            try:
                self._retrievers[knowledge_base] = HybridRetriever(
                    index_dir, settings.embedding_model_path
                )
                meta = self._retrievers[knowledge_base].meta
                self._states[knowledge_base] = {
                    "id": knowledge_base,
                    "state": "ready",
                    "documents": meta.get("documents", 0),
                    "chunks": meta.get("chunks", 0),
                    "message": "索引已加载",
                }
            except Exception as exc:
                logger.exception("Failed to load knowledge base %s", knowledge_base)
                self._states[knowledge_base] = {
                    "id": knowledge_base,
                    "state": "error",
                    "documents": 0,
                    "chunks": 0,
                    "message": str(exc),
                }
        self._states.setdefault(
            "default",
            {"id": "default", "state": "missing", "documents": 0, "chunks": 0, "message": "请先构建默认知识库"},
        )

    def get(self, knowledge_base: str) -> HybridRetriever:
        knowledge_base = self.validate_id(knowledge_base)
        if knowledge_base not in self._retrievers:
            raise RuntimeError(f"知识库 {knowledge_base} 尚未构建完成")
        return self._retrievers[knowledge_base]

    def statuses(self) -> list[dict[str, Any]]:
        return sorted(self._states.values(), key=lambda item: (item["id"] != "default", item["id"]))

    def start_build(self, knowledge_base: str, *, chapter_limit: int | None = None) -> None:
        knowledge_base = self.validate_id(knowledge_base)
        running = self._tasks.get(knowledge_base)
        if running and not running.done():
            raise RuntimeError(f"知识库 {knowledge_base} 正在构建")
        self._tasks[knowledge_base] = asyncio.create_task(
            self._build(knowledge_base, chapter_limit=chapter_limit)
        )

    async def _build(self, knowledge_base: str, *, chapter_limit: int | None) -> None:
        self._states[knowledge_base] = {
            "id": knowledge_base,
            "state": "building",
            "documents": 0,
            "chunks": 0,
            "message": "正在清洗、切分和向量化",
        }
        try:
            resource_dir = self.resource_dir(knowledge_base)
            resource_dir.mkdir(parents=True, exist_ok=True)
            meta = await asyncio.to_thread(
                build_knowledge_base,
                resource_dir,
                self.index_dir(knowledge_base),
                settings.embedding_model_path,
                chapter_limit=chapter_limit,
            )
            retriever = await asyncio.to_thread(
                HybridRetriever, self.index_dir(knowledge_base), settings.embedding_model_path
            )
            self._retrievers[knowledge_base] = retriever
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "ready",
                "documents": meta.get("documents", 0),
                "chunks": meta.get("chunks", 0),
                "message": "知识库已更新",
            }
        except Exception as exc:
            logger.exception("Knowledge base build failed: %s", knowledge_base)
            self._states[knowledge_base] = {
                "id": knowledge_base,
                "state": "error",
                "documents": 0,
                "chunks": 0,
                "message": str(exc),
            }


def read_index_meta(index_dir: Path) -> dict[str, Any]:
    path = index_dir / "index_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

