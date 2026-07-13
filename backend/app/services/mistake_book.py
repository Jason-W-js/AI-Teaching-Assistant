from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


def _normalized_content(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.strip().splitlines()).strip()


def related_mistake_attachments(
    history: list[dict[str, Any]], content: str, agent: str
) -> list[dict[str, Any]]:
    """Recover the attachment on the archived turn, including assistant answers."""
    target = _normalized_content(content)
    for index, message in enumerate(history):
        if _normalized_content(str(message.get("content", ""))) != target:
            continue
        direct = message.get("attachments")
        if isinstance(direct, list) and direct:
            return [item for item in direct if isinstance(item, dict)]
        if message.get("role") != "assistant" and agent == "学生原题":
            return []
        for previous in reversed(history[:index]):
            if previous.get("role") != "user":
                continue
            attachments = previous.get("attachments")
            if isinstance(attachments, list) and attachments:
                return [item for item in attachments if isinstance(item, dict)]
            break
    return []


class MistakeBook:
    """Small durable mistake store with atomic writes and per-student isolation."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "mistake_book.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, items: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    async def list(self, student_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            items = [item for item in self._read() if item.get("student_id") == student_id]
        return sorted(items, key=lambda item: str(item.get("created_at", "")), reverse=True)

    async def add(
        self,
        *,
        student_id: str,
        session_id: str,
        content: str,
        agent: str,
        knowledge_points: list[str],
        summary: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = _normalized_content(content)
        stored_attachments = [
            dict(attachment)
            for attachment in (attachments or [])
            if isinstance(attachment, dict) and attachment.get("id")
        ][:5]
        async with self._lock:
            items = self._read()
            duplicate = next(
                (
                    item
                    for item in items
                    if item.get("student_id") == student_id
                    and item.get("content") == normalized
                ),
                None,
            )
            if duplicate:
                if stored_attachments and not duplicate.get("attachments"):
                    duplicate["attachments"] = stored_attachments
                    duplicate["session_id"] = session_id
                    self._write(items)
                return duplicate
            item = {
                "id": uuid4().hex,
                "student_id": student_id,
                "session_id": session_id,
                "content": normalized,
                "summary": summary.strip() or normalized[:80],
                "agent": agent.strip() or "学习 Agent",
                "knowledge_points": list(dict.fromkeys(point.strip() for point in knowledge_points if point.strip()))[:12],
                "attachments": stored_attachments,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            items.append(item)
            self._write(items)
            return item

    async def delete(self, student_id: str, mistake_id: str) -> bool:
        async with self._lock:
            items = self._read()
            kept = [
                item
                for item in items
                if not (item.get("student_id") == student_id and item.get("id") == mistake_id)
            ]
            if len(kept) == len(items):
                return False
            self._write(kept)
            return True
