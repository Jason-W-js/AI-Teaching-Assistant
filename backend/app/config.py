from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    resources_dir: Path = ROOT_DIR / "RAG_Resources"
    vector_stores_dir: Path = ROOT_DIR / "data" / "vector_stores"
    embedding_model_path: Path = ROOT_DIR / os.getenv(
        "EMBEDDING_MODEL_PATH", "models/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    memory_turns: int = int(os.getenv("MEMORY_TURNS", "8"))
    session_history_messages: int = int(os.getenv("SESSION_HISTORY_MESSAGES", "100"))
    max_ollama_concurrency: int = int(os.getenv("MAX_OLLAMA_CONCURRENCY", "2"))
    remote_max_tokens: int = int(os.getenv("REMOTE_MAX_TOKENS", "8192"))
    remote_max_continuations: int = int(os.getenv("REMOTE_MAX_CONTINUATIONS", "2"))
    initial_chapter_limit: int = int(os.getenv("INITIAL_CHAPTER_LIMIT", "1"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "80"))
    max_attachment_mb: int = int(os.getenv("MAX_ATTACHMENT_MB", "20"))
    max_chat_attachments: int = int(os.getenv("MAX_CHAT_ATTACHMENTS", "5"))
    frontend_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv(
            "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if value.strip()
    )


settings = Settings()
