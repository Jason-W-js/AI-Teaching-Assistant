from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from backend.app.config import settings


class ProblemSessionStore:
    """Small, portable persistence layer for the student problem-solving state."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.root_dir / "data" / "problem_sessions"
        self._lock = asyncio.Lock()

    def _path(self, session_id: str) -> Path:
        if not session_id or not all(char.isalnum() or char in "-_" for char in session_id):
            raise ValueError("解题会话标识不合法")
        return self.root / f"{session_id}.json"

    async def load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            return {}
        async with self._lock:
            try:
                return json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

    async def save(self, session_id: str, state: dict[str, Any]) -> None:
        path = self._path(session_id)
        async with self._lock:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            payload = json.dumps(state, ensure_ascii=False, indent=2)
            temporary = path.with_suffix(".tmp")
            await asyncio.to_thread(temporary.write_text, payload, encoding="utf-8")
            await asyncio.to_thread(temporary.replace, path)

    async def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        async with self._lock:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        return True
