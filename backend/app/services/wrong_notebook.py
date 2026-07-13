from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


class WrongNotebookStore:
    """Small durable store for wrong-question records and user categories."""

    DEFAULT_CATEGORY_ID = "uncategorized"

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir or settings.root_dir / "data" / "wrong_notebook"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._storage_dir / "notebook.json"
        self._lock = asyncio.Lock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty(self) -> dict[str, Any]:
        now = self._now()
        return {
            "schema_version": "1.0",
            "categories": [
                {
                    "id": self.DEFAULT_CATEGORY_ID,
                    "name": "未分类",
                    "created_at": now,
                    "updated_at": now,
                }
            ],
            "items": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._empty()
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty()
        if not isinstance(value, dict):
            return self._empty()
        value.setdefault("categories", [])
        value.setdefault("items", [])
        if not any(
            item.get("id") == self.DEFAULT_CATEGORY_ID for item in value["categories"]
        ):
            value["categories"].insert(0, self._empty()["categories"][0])
        return value

    def _write(self, value: dict[str, Any]) -> None:
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._path)

    @staticmethod
    def _default_title(messages: list[dict[str, Any]]) -> str:
        content = next(
            (
                str(item.get("content", "")).strip()
                for item in messages
                if item.get("role") == "user" and str(item.get("content", "")).strip()
            ),
            "未命名错题",
        )
        content = content.split("\n[附件：", 1)[0].replace("\n", " ").strip()
        return content if len(content) <= 56 else f"{content[:56].rstrip()}…"

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            value = self._read()
            categories = sorted(
                value["categories"],
                key=lambda item: (item.get("id") != self.DEFAULT_CATEGORY_ID, item.get("name", "")),
            )
            items = sorted(
                value["items"], key=lambda item: item.get("updated_at", ""), reverse=True
            )
            return {"categories": categories, "items": items}

    async def create(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        knowledge_base: str,
        knowledge_points: list[str],
        title: str = "",
        category_id: str = DEFAULT_CATEGORY_ID,
    ) -> dict[str, Any]:
        async with self._lock:
            value = self._read()
            category_ids = {item["id"] for item in value["categories"]}
            if category_id not in category_ids:
                category_id = self.DEFAULT_CATEGORY_ID
            now = self._now()
            item = {
                "id": uuid4().hex,
                "title": title.strip() or self._default_title(messages),
                "category_id": category_id,
                "session_id": session_id,
                "knowledge_base": knowledge_base,
                "knowledge_points": list(dict.fromkeys(point.strip() for point in knowledge_points if point.strip())),
                "messages": messages,
                "created_at": now,
                "updated_at": now,
            }
            value["items"].append(item)
            self._write(value)
            return item

    async def update(
        self, item_id: str, *, title: str | None = None, category_id: str | None = None
    ) -> dict[str, Any] | None:
        async with self._lock:
            value = self._read()
            item = next((entry for entry in value["items"] if entry.get("id") == item_id), None)
            if item is None:
                return None
            if title is not None:
                item["title"] = title.strip()
            if category_id is not None:
                category_ids = {entry["id"] for entry in value["categories"]}
                if category_id not in category_ids:
                    raise ValueError("错题分类不存在")
                item["category_id"] = category_id
            item["updated_at"] = self._now()
            self._write(value)
            return item

    async def delete(self, item_id: str) -> bool:
        async with self._lock:
            value = self._read()
            original = len(value["items"])
            value["items"] = [item for item in value["items"] if item.get("id") != item_id]
            if len(value["items"]) == original:
                return False
            self._write(value)
            return True

    async def create_category(self, name: str) -> dict[str, Any]:
        async with self._lock:
            value = self._read()
            normalized = name.strip()
            if any(item.get("name", "").casefold() == normalized.casefold() for item in value["categories"]):
                raise ValueError("已存在同名分类")
            now = self._now()
            category = {"id": uuid4().hex, "name": normalized, "created_at": now, "updated_at": now}
            value["categories"].append(category)
            self._write(value)
            return category

    async def rename_category(self, category_id: str, name: str) -> dict[str, Any] | None:
        if category_id == self.DEFAULT_CATEGORY_ID:
            raise ValueError("默认分类不能重命名")
        async with self._lock:
            value = self._read()
            category = next(
                (item for item in value["categories"] if item.get("id") == category_id), None
            )
            if category is None:
                return None
            normalized = name.strip()
            if any(
                item.get("id") != category_id
                and item.get("name", "").casefold() == normalized.casefold()
                for item in value["categories"]
            ):
                raise ValueError("已存在同名分类")
            category["name"] = normalized
            category["updated_at"] = self._now()
            self._write(value)
            return category

    async def delete_category(self, category_id: str) -> bool:
        if category_id == self.DEFAULT_CATEGORY_ID:
            raise ValueError("默认分类不能删除")
        async with self._lock:
            value = self._read()
            original = len(value["categories"])
            value["categories"] = [
                item for item in value["categories"] if item.get("id") != category_id
            ]
            if len(value["categories"]) == original:
                return False
            for item in value["items"]:
                if item.get("category_id") == category_id:
                    item["category_id"] = self.DEFAULT_CATEGORY_ID
                    item["updated_at"] = self._now()
            self._write(value)
            return True
