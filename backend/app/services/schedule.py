from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import settings


class StudentSchedule:
    """Durable, student-scoped schedule storage with atomic file updates."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.root_dir / "data" / "student_schedule.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
        with self._lock:
            items = [item for item in self._read() if item.get("student_id") == student_id]
        return sorted(
            items,
            key=lambda item: (
                str(item.get("date", "")),
                str(item.get("time", "")) or "99:99",
                str(item.get("created_at", "")),
            ),
        )

    async def add(
        self,
        *,
        student_id: str,
        title: str,
        date: str,
        time: str,
        category: str,
        note: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "id": uuid4().hex,
            "student_id": student_id,
            "title": title.strip(),
            "date": date,
            "time": time,
            "category": category,
            "note": note.strip(),
            "completed": False,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            items = self._read()
            items.append(item)
            self._write(items)
        return item

    async def set_completed(
        self, student_id: str, item_id: str, completed: bool
    ) -> dict[str, Any] | None:
        with self._lock:
            items = self._read()
            target = next(
                (
                    item
                    for item in items
                    if item.get("student_id") == student_id and item.get("id") == item_id
                ),
                None,
            )
            if target is None:
                return None
            target["completed"] = completed
            target["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(items)
            return dict(target)

    async def delete(self, student_id: str, item_id: str) -> bool:
        with self._lock:
            items = self._read()
            kept = [
                item
                for item in items
                if not (item.get("student_id") == student_id and item.get("id") == item_id)
            ]
            if len(kept) == len(items):
                return False
            self._write(kept)
            return True
