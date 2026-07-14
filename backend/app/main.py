from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.agents.workflow import CircuitTutorEngine
from backend.app.config import settings
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.pipeline import INGESTION_MANIFEST
from backend.app.schemas import (
    ChatRequest,
    KnowledgeBaseRebuildRequest,
    WrongQuestionCategoryCreate,
    WrongQuestionCreate,
    WrongQuestionUpdate,
)
from backend.app.services.memory import ConversationMemory
from backend.app.services.ollama_client import OllamaClient
from backend.app.services.openai_compatible_client import OpenAICompatibleClient
from backend.app.services.attachments import ALLOWED_ATTACHMENT_SUFFIXES, AttachmentStore
from backend.app.services.problem_sessions import ProblemSessionStore
from backend.app.services.knowledge_graph import KnowledgeGraphService
from backend.app.services.wrong_notebook import WrongNotebookStore
from backend.app.services.model_catalog import canonical_model_id


def configure_logging() -> None:
    log_dir = settings.root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = RotatingFileHandler(
            log_dir / "backend.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(console)
        root.addHandler(file_handler)


configure_logging()
logger = logging.getLogger(__name__)


class FirstTokenTimeoutError(TimeoutError):
    """Raised only when a workflow has not produced visible content in time."""

ollama = OllamaClient()
lmstudio = OpenAICompatibleClient(
    provider="lmstudio",
    model=settings.lmstudio_model,
    base_url=settings.lmstudio_base_url,
    allow_images=True,
    trust_env=False,
)
specialist_clients = {
    role: OpenAICompatibleClient(
        provider="lmstudio",
        model=model,
        base_url=settings.lmstudio_base_url,
        allow_images=role == "understanding",
        trust_env=False,
    )
    for role, model in {
        "understanding": settings.understanding_model,
        "solver": settings.solver_model,
        "diagnosis": settings.diagnosis_model,
        "tutor": settings.tutor_model,
    }.items()
    if model
}
memory = ConversationMemory()
knowledge_bases = KnowledgeBaseManager()
problem_sessions = ProblemSessionStore()
engine = CircuitTutorEngine(ollama, knowledge_bases, problem_sessions)
attachments = AttachmentStore()
wrong_notebook = WrongNotebookStore()
knowledge_graph = KnowledgeGraphService(knowledge_bases)


@asynccontextmanager
async def lifespan(_: FastAPI):
    knowledge_bases.load_existing()
    await memory.connect()
    yield
    await ollama.close()
    await lmstudio.close()
    await asyncio.gather(*(client.close() for client in specialist_clients.values()))
    await memory.close()
    knowledge_bases.close_all()


app = FastAPI(
    title="CircuitMind 多智能体电路助教",
    version="0.1.0",
    description="本地 Qwen + LangGraph + Hybrid RAG 教学服务",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - start) * 1000
    logger.info("%s %s -> %s %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    safe_details = [
        {key: value for key, value in item.items() if key not in {"input", "ctx"}}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数不合法", "details": safe_details},
    )


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"error": "服务内部错误", "detail": str(exc)})


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def response_chunks(content: str) -> list[str]:
    paragraphs = re.split(r"(?<=\n\n)", content)
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 420:
            if paragraph:
                chunks.append(paragraph)
            continue
        chunks.extend(paragraph[index : index + 420] for index in range(0, len(paragraph), 420))
    return chunks


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ollama_health, lmstudio_health = await asyncio.gather(ollama.health(), lmstudio.health())
    default_health = lmstudio_health if settings.default_model_provider == "lmstudio" else ollama_health
    return {
        "status": "ok" if default_health.get("ok") else "degraded",
        "ollama": ollama_health,
        "lmstudio": lmstudio_health,
        "memory": memory.backend,
        "knowledge_bases": knowledge_bases.statuses(),
        "thinking_enabled": True,
    }


@app.get("/api/kb/status")
async def knowledge_base_status() -> dict[str, Any]:
    return {"knowledge_bases": knowledge_bases.statuses()}


@app.get("/api/kb/{knowledge_base}/graph")
async def persisted_knowledge_graph(knowledge_base: str) -> dict[str, Any]:
    try:
        return knowledge_bases.graph(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base(payload: KnowledgeBaseRebuildRequest) -> dict[str, Any]:
    api_key = payload.api_key
    base_url = payload.base_url
    if payload.model_provider == "deepseek":
        api_key = api_key or settings.deepseek_api_key
        base_url = base_url or settings.deepseek_base_url
    elif payload.model_provider == "qwen":
        api_key = api_key or settings.qwen_api_key
        base_url = base_url or settings.qwen_base_url
    elif payload.model_provider == "lmstudio":
        base_url = base_url or settings.lmstudio_base_url
    config = BuildModelConfig(
        provider=payload.model_provider,
        model=canonical_model_id(payload.model_provider, payload.model),
        api_key=api_key,
        base_url=base_url,
    )
    try:
        build_state = knowledge_bases.start_build(
            payload.knowledge_base,
            chapter_limit=payload.chapter_limit,
            model_config=config if config.enabled else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": payload.knowledge_base,
        "state": "building",
        "build": build_state,
        "message": "多模态知识库已开始后台重建",
    }


@app.delete("/api/kb/{knowledge_base}/build")
async def cancel_knowledge_base_build(knowledge_base: str) -> dict[str, Any]:
    try:
        state = knowledge_bases.cancel_build(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": knowledge_base,
        "state": state,
        "message": "取消请求已提交，正在清理未完成缓存",
    }


@app.delete("/api/kb/{knowledge_base}")
async def delete_knowledge_base(knowledge_base: str) -> dict[str, Any]:
    try:
        await knowledge_bases.delete(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "knowledge_base": knowledge_base,
        "message": f"知识库 {knowledge_base} 已删除",
    }


@app.get("/api/sessions")
async def conversation_sessions() -> dict[str, Any]:
    return {"sessions": await memory.list_sessions()}


@app.get("/api/wrong-questions")
async def wrong_questions() -> dict[str, Any]:
    return await wrong_notebook.snapshot()


@app.post("/api/wrong-questions/categories")
async def create_wrong_question_category(
    payload: WrongQuestionCategoryCreate,
) -> dict[str, Any]:
    try:
        return {"category": await wrong_notebook.create_category(payload.name)}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.patch("/api/wrong-questions/categories/{category_id}")
async def rename_wrong_question_category(
    category_id: str, payload: WrongQuestionCategoryCreate
) -> dict[str, Any]:
    try:
        category = await wrong_notebook.rename_category(category_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if category is None:
        raise HTTPException(status_code=404, detail="错题分类不存在")
    return {"category": category}


@app.delete("/api/wrong-questions/categories/{category_id}")
async def delete_wrong_question_category(category_id: str) -> dict[str, Any]:
    try:
        deleted = await wrong_notebook.delete_category(category_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="错题分类不存在")
    return {"ok": True}


@app.post("/api/wrong-questions")
async def create_wrong_question(payload: WrongQuestionCreate) -> dict[str, Any]:
    try:
        knowledge_bases.validate_id(payload.knowledge_base)
        message_values = [item.model_dump() for item in payload.messages]
        combined = "\n".join(item["content"] for item in message_values)
        knowledge_points = knowledge_graph.infer_knowledge_points(
            combined, payload.knowledge_base
        )
        item = await wrong_notebook.create(
            session_id=payload.session_id,
            title=payload.title,
            category_id=payload.category_id,
            knowledge_base=payload.knowledge_base,
            messages=message_values,
            knowledge_points=knowledge_points,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": item}


@app.patch("/api/wrong-questions/{item_id}")
async def update_wrong_question(
    item_id: str, payload: WrongQuestionUpdate
) -> dict[str, Any]:
    try:
        item = await wrong_notebook.update(
            item_id, title=payload.title, category_id=payload.category_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="错题记录不存在")
    return {"item": item}


@app.delete("/api/wrong-questions/{item_id}")
async def delete_wrong_question(item_id: str) -> dict[str, Any]:
    if not await wrong_notebook.delete(item_id):
        raise HTTPException(status_code=404, detail="错题记录不存在")
    return {"ok": True}


@app.get("/api/knowledge-graph")
async def course_knowledge_graph(knowledge_base: str = "default") -> dict[str, Any]:
    try:
        knowledge_bases.validate_id(knowledge_base)
        notebook = await wrong_notebook.snapshot()
        relevant_wrong_questions = [
            item
            for item in notebook["items"]
            if item.get("knowledge_base", "default") == knowledge_base
        ]
        return knowledge_graph.build(knowledge_base, relevant_wrong_questions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
async def conversation_session(session_id: str) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": session_id, "messages": await memory.history(session_id)}


@app.delete("/api/sessions/{session_id}")
async def delete_conversation_session(session_id: str) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    deleted_history = await memory.delete(session_id)
    deleted_attachments = await attachments.delete_session(session_id)
    deleted_problem = await problem_sessions.delete(session_id)
    if not deleted_history and not deleted_attachments and not deleted_problem:
        raise HTTPException(status_code=404, detail="历史会话不存在或已被删除")
    return {"ok": True, "session_id": session_id}


@app.get("/api/models")
async def available_models() -> dict[str, Any]:
    model_health, lmstudio_health = await asyncio.gather(ollama.health(), lmstudio.health())
    local_models = model_health.get("models", []) if model_health.get("ok") else model_health.get("models", [])
    if settings.ollama_model not in local_models:
        local_models = [settings.ollama_model, *local_models]
    return {
        "default": {
            "provider": settings.default_model_provider,
            "model": settings.lmstudio_model if settings.default_model_provider == "lmstudio" else settings.ollama_model,
        },
        "providers": [
            {
                "id": "lmstudio",
                "label": "本地 LM Studio",
                "description": "通过本机 OpenAI 兼容接口运行模型，数据不离开本机",
                "models": list(dict.fromkeys([settings.lmstudio_model, *lmstudio_health.get("models", [])])),
                "default_model": settings.lmstudio_model,
                "base_url": settings.lmstudio_base_url,
                "requires_api_key": False,
                "configured": bool(lmstudio_health.get("ok")),
            },
            {
                "id": "ollama",
                "label": "本地 Ollama",
                "description": "使用本机已安装模型，数据不离开本机",
                "models": list(dict.fromkeys(local_models)),
                "default_model": settings.ollama_model,
                "base_url": settings.ollama_base_url,
                "requires_api_key": False,
                "configured": bool(model_health.get("ok")),
            },
            {
                "id": "deepseek",
                "label": "DeepSeek API",
                "description": "DeepSeek 官方 OpenAI 兼容接口",
                "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
                "default_model": "deepseek-v4-flash",
                "base_url": settings.deepseek_base_url,
                "requires_api_key": True,
                "configured": bool(settings.deepseek_api_key),
            },
            {
                "id": "qwen",
                "label": "通义千问 API",
                "description": "阿里云百炼 OpenAI 兼容接口",
                "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
                "default_model": "qwen-plus",
                "base_url": settings.qwen_base_url,
                "requires_api_key": True,
                "configured": bool(settings.qwen_api_key),
            },
            {
                "id": "custom",
                "label": "自定义 API",
                "description": "连接其他 OpenAI Chat Completions 兼容服务",
                "models": [],
                "default_model": "",
                "base_url": "",
                "requires_api_key": True,
                "configured": False,
            },
        ],
    }


def select_model_client(payload: ChatRequest) -> tuple[Any, bool]:
    if payload.model_provider == "ollama":
        if payload.model == settings.ollama_model:
            return ollama, False
        return OllamaClient(model=payload.model), True

    if payload.model_provider == "lmstudio":
        base_url = payload.base_url or settings.lmstudio_base_url
        if payload.model == settings.lmstudio_model and base_url.rstrip("/") == settings.lmstudio_base_url.rstrip("/"):
            return lmstudio, False
        return (
            OpenAICompatibleClient(
                provider="lmstudio",
                model=payload.model,
                base_url=base_url,
                allow_images=True,
                trust_env=False,
            ),
            True,
        )

    if payload.model_provider == "deepseek":
        api_key = payload.api_key or settings.deepseek_api_key
        base_url = payload.base_url or settings.deepseek_base_url
    elif payload.model_provider == "qwen":
        api_key = payload.api_key or settings.qwen_api_key
        base_url = payload.base_url or settings.qwen_base_url
    else:
        api_key = payload.api_key
        base_url = payload.base_url

    if not api_key:
        raise ValueError("所选云端模型尚未配置 API Key")
    if not base_url:
        raise ValueError("所选模型尚未配置 API Base URL")
    return (
        OpenAICompatibleClient(
            provider=payload.model_provider,
            model=payload.model,
            api_key=api_key,
            base_url=base_url,
        ),
        True,
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        selected_client: Any | None = None
        close_selected_client = False
        task: asyncio.Task[Any] | None = None
        user_persisted = False
        assistant_persisted = False
        effective_message = payload.message

        async def persist_interruption(detail: str) -> None:
            nonlocal assistant_persisted
            if not user_persisted or assistant_persisted:
                return
            safe_detail = re.sub(r"\s+", " ", detail).strip()[:360]
            content = (
                f"> ⚠️ 本次回答未完成：{safe_detail or '生成连接意外中断'}\n\n"
                "原问题已经保留，可以点击“重新生成本题”继续。"
            )
            try:
                await memory.append(
                    payload.session_id,
                    "assistant",
                    content,
                    {
                        "agent": "系统恢复",
                        "provider": payload.model_provider,
                        "model": payload.model,
                        "failed": True,
                        "retry_message": effective_message,
                        "retry_attachment_ids": payload.attachment_ids,
                    },
                )
                assistant_persisted = True
            except Exception:
                logger.exception("Unable to persist interrupted chat state")

        try:
            selected_client, close_selected_client = select_model_client(payload)
            yield sse(
                "connected",
                {
                    "session_id": payload.session_id,
                    "provider": payload.model_provider,
                    "model": payload.model,
                    "knowledge_base": payload.knowledge_base,
                },
            )
            history = await memory.recent(payload.session_id)
            resolved = await attachments.resolve(payload.session_id, payload.attachment_ids)
            effective_message = payload.message or (
                "请根据附件中的原题生成一道同类型新题。"
                if payload.mode == "quiz"
                else "请识别并解答附件中的电路题。"
            )
            attachment_names = [item["name"] for item in resolved.items]
            memory_message = effective_message
            if attachment_names:
                memory_message += f"\n[附件：{'、'.join(attachment_names)}]"
            await memory.append(
                payload.session_id,
                "user",
                memory_message,
                {"attachment_ids": payload.attachment_ids},
            )
            user_persisted = True
            event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            streamed_answer = False
            last_event_at = perf_counter()
            first_token_deadline = (
                last_event_at + settings.chat_first_token_timeout_seconds
            )

            async def on_status(status: dict[str, Any]) -> None:
                await event_queue.put(("status", status))

            async def on_delta(content: str) -> None:
                await event_queue.put(("delta", {"content": content}))

            task = asyncio.create_task(
                engine.run(
                    session_id=payload.session_id,
                    message=effective_message,
                    mode=payload.mode,
                    tutor_action=payload.tutor_action,
                    hint_level=payload.hint_level,
                    tutoring_mode=payload.tutoring_mode,
                    knowledge_base=payload.knowledge_base,
                    history=history,
                    attachment_text=resolved.text,
                    attachment_images=resolved.images,
                    attachment_names=attachment_names,
                    llm=selected_client,
                    agent_clients=specialist_clients,
                    on_status=on_status,
                    on_delta=on_delta,
                )
            )
            while not task.done() or not event_queue.empty():
                if not streamed_answer and not task.done():
                    remaining = first_token_deadline - perf_counter()
                    if remaining <= 0:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        raise FirstTokenTimeoutError
                    queue_wait_timeout = min(0.5, remaining)
                else:
                    # Once visible output starts there is intentionally no total
                    # generation deadline.  The student can still stop it manually.
                    queue_wait_timeout = 0.5
                try:
                    event_name, event_data = await asyncio.wait_for(
                        event_queue.get(), timeout=queue_wait_timeout
                    )
                    if event_name == "delta" and event_data.get("content"):
                        streamed_answer = True
                    yield sse(event_name, event_data)
                    last_event_at = perf_counter()
                except asyncio.TimeoutError:
                    if (
                        not streamed_answer
                        and not task.done()
                        and perf_counter() >= first_token_deadline
                    ):
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        raise FirstTokenTimeoutError
                    if perf_counter() - last_event_at >= 8:
                        # Keep the fetch/SSE connection alive while a local model
                        # is doing a long non-streaming reasoning pass.
                        yield ": keep-alive\n\n"
                        last_event_at = perf_counter()
                    continue
            result = await task
            yield sse(
                "meta",
                {
                    "intent": result.intent,
                    "agent": result.agent,
                    "provider": payload.model_provider,
                    "model": payload.model,
                    "sources": result.sources,
                    "verification": result.verification,
                    "tutor_action": result.tutor_action,
                    "hint_level": result.hint_level,
                    "problem": result.problem,
                    "diagnosis": result.diagnosis,
                },
            )
            if not streamed_answer:
                for chunk in response_chunks(result.content):
                    yield sse("delta", {"content": chunk})
                    await asyncio.sleep(0)
            await memory.append(
                payload.session_id,
                "assistant",
                result.content,
                {
                    "agent": result.agent,
                    "provider": payload.model_provider,
                    "model": payload.model,
                    "intent": result.intent,
                    "tutor_action": result.tutor_action,
                },
            )
            assistant_persisted = True
            yield sse("done", {"ok": True})
        except asyncio.CancelledError:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            logger.info("Chat workflow cancelled for session %s", payload.session_id)
            await persist_interruption("页面切换、停止生成或连接断开")
            raise
        except FirstTokenTimeoutError:
            detail = (
                f"本地模型在 {settings.chat_first_token_timeout_seconds} 秒内未输出首个内容，"
                "请确认 LM Studio 正常运行后重试"
            )
            logger.warning(
                "Chat workflow produced no first token for session %s",
                payload.session_id,
            )
            await persist_interruption(detail)
            yield sse("error", {"message": detail})
        except Exception as exc:
            logger.exception("Chat workflow failed")
            await persist_interruption(str(exc))
            yield sse("error", {"message": str(exc)})
        finally:
            # StreamingResponse may close an async generator without surfacing a
            # CancelledError.  Always terminate unfinished work and leave a
            # durable retry record so the conversation can never end with only
            # the user's question.
            if user_persisted and not assistant_persisted:
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await persist_interruption("页面切换、停止生成或连接断开")
            if close_selected_client and selected_client is not None:
                await selected_client.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


ALLOWED_UPLOADS = {".pdf", ".md", ".txt", ".docx", ".xlsx", ".json", ".png", ".jpg", ".jpeg", ".webp"}
DOCUMENT_TYPES = {"auto", "textbook", "exam", "question_bank", "notes"}


@app.post("/api/attachments")
async def upload_chat_attachment(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> dict[str, Any]:
    try:
        attachments.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    original_name = Path(file.filename or "attachment.bin").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_ATTACHMENT_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"不支持的聊天附件类型：{suffix or '未知'}")
    content_type = file.content_type
    max_bytes = settings.max_attachment_mb * 1024 * 1024
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > max_bytes:
            await file.close()
            raise HTTPException(
                status_code=413,
                detail=f"聊天附件不能超过 {settings.max_attachment_mb} MB",
            )
    await file.close()
    try:
        item = await attachments.save(
            session_id=session_id,
            filename=original_name,
            content_type=content_type,
            data=bytes(content),
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    item["url"] = f"/api/attachments/{item['id']}?session_id={session_id}"
    return {"ok": True, "attachment": item}


@app.get("/api/attachments/{attachment_id}")
async def get_chat_attachment(attachment_id: str, session_id: str) -> FileResponse:
    try:
        meta, path = attachments.file_for_response(session_id, attachment_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=meta["content_type"], filename=meta["name"])


async def _save_knowledge_file(file: UploadFile, target_dir: Path) -> dict[str, Any]:
    original_name = Path(file.filename or "upload.bin").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_UPLOADS:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型：{suffix or '未知'}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / original_name
    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with target.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_mb} MB")
            handle.write(chunk)
    await file.close()
    return {
        "filename": original_name,
        "size": size,
        "content_type": file.content_type or mimetypes.guess_type(original_name)[0],
        "indexable": suffix in {".pdf", ".md", ".txt", ".docx", ".xlsx", ".json"},
    }


def _record_document_types(target_dir: Path, files: list[dict[str, Any]], document_type: str) -> None:
    manifest_path = target_dir / INGESTION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except json.JSONDecodeError:
        manifest = {}
    entries = manifest.setdefault("files", {})
    for item in files:
        if item["indexable"]:
            entries[item["filename"]] = {"document_type": document_type}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    knowledge_base: str = Form("default"),
    rebuild: bool = Form(True),
    document_type: str = Form("auto"),
) -> dict[str, Any]:
    try:
        knowledge_base = knowledge_bases.validate_id(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="资料类型不合法")
    target_dir = knowledge_bases.resource_dir(knowledge_base)
    saved = await _save_knowledge_file(file, target_dir)
    _record_document_types(target_dir, [saved], document_type)
    indexable = saved["indexable"]
    if rebuild and indexable:
        try:
            knowledge_bases.start_build(knowledge_base, chapter_limit=None)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        **{key: saved[key] for key in ("filename", "size", "content_type")},
        "knowledge_base": knowledge_base,
        "indexing": bool(rebuild and indexable),
        "message": "文件已保存，知识库正在后台更新" if rebuild and indexable else "文件已保存",
    }


@app.post("/api/kb/ingest")
async def ingest_knowledge_files(
    files: list[UploadFile] = File(...),
    knowledge_base: str = Form("default"),
    document_type: str = Form("auto"),
) -> dict[str, Any]:
    """Batch ingestion prevents rebuilding the same index once per uploaded file."""
    try:
        knowledge_base = knowledge_bases.validate_id(knowledge_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="资料类型不合法")
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="每批请上传 1 到 20 个文件")
    target_dir = knowledge_bases.resource_dir(knowledge_base)
    saved = [await _save_knowledge_file(file, target_dir) for file in files]
    _record_document_types(target_dir, saved, document_type)
    if any(item["indexable"] for item in saved):
        try:
            knowledge_bases.start_build(knowledge_base, chapter_limit=None)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "files": saved,
        "knowledge_base": knowledge_base,
        "document_type": document_type,
        "indexing": True,
        "message": f"已接收 {len(saved)} 份资料，正在自动分类、抽题并关联教材知识",
    }


@app.get("/api/teacher/status")
async def teacher_status() -> dict[str, Any]:
    return {"available": False, "message": "教师工作台接口已预留，业务功能将在后续版本开放。"}


frontend_dist = settings.root_dir / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def frontend_app(full_path: str):
        requested = (frontend_dist / full_path).resolve()
        if requested.is_file() and frontend_dist.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(frontend_dist / "index.html")
