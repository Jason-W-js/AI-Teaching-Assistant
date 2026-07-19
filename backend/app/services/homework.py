from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
import re
import shutil
import threading
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import fitz
import numpy as np
from PIL import Image, ImageDraw

from backend.app.config import settings
from backend.app.rag.pdf_extract_kit import PDFExtractKitAdapter
from backend.app.services.qwen_multimodal_client import QwenVisionClient


logger = logging.getLogger(__name__)

HOMEWORK_SOURCE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ANSWER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HOMEWORK_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
ASSET_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,160}")
STUDENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,96}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any, limit: int = 24000) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()[:limit]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "是"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _bbox_list(value: Any) -> list[list[float]]:
    if isinstance(value, (tuple, list)) and len(value) == 4 and all(
        isinstance(item, (int, float)) for item in value
    ):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[list[float]] = []
    for bbox in value:
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            continue
        try:
            left, top, right, bottom = (float(item) for item in bbox)
        except (TypeError, ValueError):
            continue
        left, right = sorted((max(0.0, min(1000.0, left)), max(0.0, min(1000.0, right))))
        top, bottom = sorted((max(0.0, min(1000.0, top)), max(0.0, min(1000.0, bottom))))
        if right - left >= 3 and bottom - top >= 3:
            result.append([round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)])
    return result


def _field_bboxes(item: dict[str, Any], plural: str, singular: str) -> list[list[float]]:
    return _bbox_list(item.get(plural, item.get(singular, [])))


def _normalize_options(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for index, option in enumerate(value):
        if isinstance(option, dict):
            label = _clean_text(option.get("label", ""), 12)
            text = _clean_text(option.get("text", option.get("content", "")), 3000)
        else:
            label = chr(65 + index) if index < 26 else str(index + 1)
            text = _clean_text(option, 3000)
        if text:
            result.append({"label": label or chr(65 + index), "text": text})
    return result


def _part_label(value: Any) -> str:
    label = _clean_text(value, 24)
    label = re.sub(r"^[（(\[]\s*|\s*[）)\]]$", "", label)
    return re.sub(r"[.、：:]$", "", label).strip()


def _normalize_labeled_parts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for index, part in enumerate(value):
        if isinstance(part, dict):
            label = _part_label(part.get("label", part.get("number", "")))
            text = _clean_text(part.get("text", part.get("content", "")))
        else:
            label = str(index + 1)
            text = _clean_text(part)
        if text:
            result.append({"label": label or str(index + 1), "text": text})
    return result


_NUMBERED_PART_PATTERN = re.compile(r"(?<![\w$])(?:\(|（)\s*(\d{1,2})\s*(?:\)|）)")


def _split_labeled_text(value: Any) -> tuple[str, list[dict[str, str]]]:
    """Split legacy inline (1)/(2)/(3) text without treating later references as new parts."""
    text = _clean_text(value)
    if not text:
        return "", []
    accepted: list[re.Match[str]] = []
    expected = 1
    for match in _NUMBERED_PART_PATTERN.finditer(text):
        number = int(match.group(1))
        if number == expected:
            accepted.append(match)
            expected += 1
    if not accepted or int(accepted[0].group(1)) != 1:
        return text, []
    stem = text[:accepted[0].start()].strip()
    parts: list[dict[str, str]] = []
    for index, match in enumerate(accepted):
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        part_text = text[match.end():end].strip()
        if part_text:
            parts.append({"label": match.group(1), "text": part_text})
    return stem, parts


def _merge_labeled_parts(parts: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    for part in parts:
        label = _part_label(part.get("label"))
        text = _clean_text(part.get("text"))
        if label and text:
            grouped.setdefault(label, []).append(text)
    result: list[dict[str, str]] = []
    for label, texts in grouped.items():
        merged_text = _merge_prompt_parts(texts)
        if merged_text:
            result.append({"label": label, "text": merged_text})
    return result


def _compose_labeled_text(stem: Any, parts: Any) -> str:
    lines = [_clean_text(stem)] if _clean_text(stem) else []
    for part in _normalize_labeled_parts(parts):
        lines.append(f"({part['label']}) {part['text']}")
    return "\n".join(lines).strip()


def _option_columns(value: Any) -> int:
    columns = int(_as_float(value, 1))
    return columns if columns in {1, 2, 4} else 1


def _figure_position(value: Any) -> str:
    position = _clean_text(value or "after_question", 40)
    return position if position in {"before_question", "after_question", "after_options"} else "after_question"


def _question_type(value: Any) -> str:
    raw = _clean_text(value or "other", 40).lower()
    aliases = {
        "multiple_choice": "choice",
        "single_choice": "choice",
        "选择题": "choice",
        "calculation": "calculation",
        "计算题": "calculation",
        "short_answer": "short_answer",
        "简答题": "short_answer",
        "design": "design",
        "设计题": "design",
    }
    allowed = {"choice", "calculation", "short_answer", "design", "other"}
    return aliases.get(raw, raw if raw in allowed else "other")


def _comparison_text(value: str) -> str:
    return re.sub(r"[\s`$\\，。；：、（）()【】\[\]{}]", "", value).lower()


def _merge_prompt_parts(parts: Iterable[str]) -> str:
    """Merge true continuations while dropping repeated or hallucinated restatements."""
    merged: list[str] = []
    comparisons: list[str] = []
    for raw in parts:
        text = _clean_text(raw)
        compact = _comparison_text(text)
        if not compact:
            continue
        duplicate = False
        for existing in comparisons:
            shorter = min(len(existing), len(compact))
            if compact in existing or (existing in compact and shorter >= 80):
                duplicate = True
                break
            prefix = 0
            for left, right in zip(existing, compact):
                if left != right:
                    break
                prefix += 1
            similarity = SequenceMatcher(None, existing, compact, autojunk=False).ratio()
            if similarity >= 0.72 or (shorter >= 100 and prefix >= max(60, round(shorter * 0.35))):
                duplicate = True
                break
        if not duplicate:
            merged.append(text)
            comparisons.append(compact)
    return "\n".join(merged).strip()


class HomeworkStore:
    """Durable single-course homework store with answer-safe public views."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.root_dir / "data" / "homework").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "homework.json"
        self._lock = threading.RLock()
        self._recover_interrupted_jobs()

    @staticmethod
    def validate_homework_id(homework_id: str) -> str:
        if not HOMEWORK_ID_PATTERN.fullmatch(homework_id):
            raise ValueError("作业标识不合法")
        return homework_id

    @staticmethod
    def validate_submission_id(submission_id: str) -> str:
        if not HOMEWORK_ID_PATTERN.fullmatch(submission_id):
            raise ValueError("提交标识不合法")
        return submission_id

    @staticmethod
    def validate_student_id(student_id: str) -> str:
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            raise ValueError("学生标识不合法")
        return student_id

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError
            return {
                "homeworks": value.get("homeworks", []) if isinstance(value.get("homeworks"), list) else [],
                "submissions": value.get("submissions", []) if isinstance(value.get("submissions"), list) else [],
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {"homeworks": [], "submissions": []}

    def _write(self, value: dict[str, list[dict[str, Any]]]) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.index_path)

    def _recover_interrupted_jobs(self) -> None:
        """Make jobs interrupted by a server restart retryable instead of stuck."""
        with self._lock:
            state = self._read()
            changed = False
            for homework in state["homeworks"]:
                if homework.get("status") == "processing":
                    homework.update({
                        "status": "error",
                        "processing_error": "服务重启导致识别任务中断，请重新识别",
                        "processing_progress": 0,
                        "processing_message": "识别任务已中断",
                        "updated_at": _now(),
                    })
                    changed = True
            for submission in state["submissions"]:
                if submission.get("status") == "grading":
                    submission.update({
                        "status": "error",
                        "processing_error": "服务重启导致批改任务中断，请重新提交答案",
                        "updated_at": _now(),
                    })
                    changed = True
            if changed:
                self._write(state)

    def _homework_dir(self, homework_id: str) -> Path:
        return self.root / self.validate_homework_id(homework_id)

    @staticmethod
    def _asset_url(homework_id: str, asset: dict[str, Any]) -> dict[str, Any]:
        value = dict(asset)
        value["url"] = f"/api/homeworks/{homework_id}/assets/{asset['file']}"
        return value

    def _public_question(
        self, homework_id: str, question: dict[str, Any], *, include_answers: bool
    ) -> dict[str, Any]:
        result = {
            key: question.get(key)
            for key in (
                "id", "section_key", "section_title", "number", "question_type",
                "prompt", "subquestions", "options", "option_columns", "figure_position", "points",
                "page_start", "page_end", "sequence",
            )
        }
        result["section_key"] = result.get("section_key") or "questions"
        result["section_title"] = result.get("section_title") or "题目"
        result["options"] = _normalize_options(result.get("options"))
        result["subquestions"] = _normalize_labeled_parts(result.get("subquestions"))
        result["option_columns"] = _option_columns(result.get("option_columns"))
        result["figure_position"] = _figure_position(result.get("figure_position"))
        result["layout_images"] = [
            self._asset_url(homework_id, item)
            for item in question.get("layout_images", [])
            if isinstance(item, dict) and item.get("file")
        ]
        result["figures"] = [
            self._asset_url(homework_id, item)
            for item in question.get("figures", [])
            if isinstance(item, dict) and item.get("file")
        ]
        if include_answers:
            result["answer"] = str(question.get("answer", ""))
            result["answer_subquestions"] = _normalize_labeled_parts(
                question.get("answer_subquestions")
            )
            result["rubric"] = str(question.get("rubric", ""))
        return result

    def _public_submission(self, submission: dict[str, Any]) -> dict[str, Any]:
        submission_id = str(submission["id"])
        result = dict(submission)
        result["answer_images"] = [
            {
                **item,
                "url": f"/api/homework-submissions/{submission_id}/files/{item['file']}",
            }
            for item in submission.get("answer_images", [])
            if isinstance(item, dict) and item.get("file")
        ]
        return result

    def _public_homework(
        self,
        homework: dict[str, Any],
        submissions: list[dict[str, Any]],
        *,
        role: str,
        student_id: str,
    ) -> dict[str, Any]:
        homework_id = str(homework["id"])
        include_answers = role == "teacher"
        result = {
            key: homework.get(key)
            for key in (
                "id", "title", "instructions", "due_at", "status", "source_name",
                "created_at", "updated_at", "published_at", "extraction_model",
                "grading_model", "review_model", "processing_error", "processing_warnings",
                "processing_progress", "processing_message", "page_count", "max_score",
            )
        }
        result["question_count"] = len(homework.get("questions", []))
        result["questions"] = [
            self._public_question(homework_id, question, include_answers=include_answers)
            for question in homework.get("questions", [])
            if isinstance(question, dict)
        ]
        if include_answers:
            result["source_url"] = f"/api/homeworks/{homework_id}/source"
            result["submissions"] = [self._public_submission(item) for item in submissions]
            result["submission_count"] = len(submissions)
        else:
            own = [item for item in submissions if item.get("student_id") == student_id]
            latest = max(own, key=lambda item: str(item.get("created_at", "")), default=None)
            result["submission"] = self._public_submission(latest) if latest else None
        return result

    def create_homework(
        self,
        *,
        title: str,
        instructions: str,
        due_at: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> dict[str, Any]:
        safe_name = Path(filename).name or "homework.pdf"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in HOMEWORK_SOURCE_SUFFIXES:
            raise ValueError(f"不支持的作业附件类型：{suffix or '未知'}")
        homework_id = uuid4().hex
        homework_dir = self._homework_dir(homework_id)
        homework_dir.mkdir(parents=True, exist_ok=False)
        source_name = f"source{suffix}"
        (homework_dir / source_name).write_bytes(data)
        timestamp = _now()
        item = {
            "id": homework_id,
            "title": _clean_text(title, 120) or Path(safe_name).stem,
            "instructions": _clean_text(instructions, 2000),
            "due_at": _clean_text(due_at, 80),
            "status": "processing",
            "source_name": safe_name,
            "source_file": source_name,
            "source_content_type": content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
            "created_at": timestamp,
            "updated_at": timestamp,
            "published_at": "",
            "extraction_model": settings.qwen_homework_extraction_model,
            "grading_model": settings.qwen_homework_grading_model,
            "review_model": settings.qwen_homework_review_model,
            "processing_error": "",
            "processing_warnings": [],
            "processing_progress": 0,
            "processing_message": "等待开始识别",
            "page_count": 0,
            "max_score": 0,
            "questions": [],
        }
        with self._lock:
            state = self._read()
            state["homeworks"].append(item)
            self._write(state)
        return self.get_homework(homework_id, role="teacher")

    def list_homeworks(self, *, role: str, student_id: str = "") -> list[dict[str, Any]]:
        if role not in {"teacher", "student"}:
            raise ValueError("作业视图角色不合法")
        if role == "student":
            self.validate_student_id(student_id)
        with self._lock:
            state = self._read()
        items = state["homeworks"]
        if role == "student":
            items = [item for item in items if item.get("status") == "published"]
        result = [
            self._public_homework(
                item,
                [submission for submission in state["submissions"] if submission.get("homework_id") == item.get("id")],
                role=role,
                student_id=student_id,
            )
            for item in items
        ]
        return sorted(result, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get_homework(
        self, homework_id: str, *, role: str, student_id: str = ""
    ) -> dict[str, Any]:
        self.validate_homework_id(homework_id)
        items = self.list_homeworks(role=role, student_id=student_id)
        item = next((value for value in items if value.get("id") == homework_id), None)
        if item is None:
            raise FileNotFoundError("作业不存在或尚未发布")
        return item

    def get_raw_homework(self, homework_id: str) -> dict[str, Any]:
        self.validate_homework_id(homework_id)
        with self._lock:
            item = next(
                (value for value in self._read()["homeworks"] if value.get("id") == homework_id),
                None,
            )
        if item is None:
            raise FileNotFoundError("作业不存在")
        return json.loads(json.dumps(item, ensure_ascii=False))

    def update_homework(self, homework_id: str, **updates: Any) -> None:
        self.validate_homework_id(homework_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["homeworks"] if value.get("id") == homework_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("作业不存在")
            item.update(updates)
            item["updated_at"] = _now()
            self._write(state)

    def publish(self, homework_id: str) -> dict[str, Any]:
        raw = self.get_raw_homework(homework_id)
        if raw.get("status") not in {"draft", "published"} or not raw.get("questions"):
            raise RuntimeError("题目尚未识别完成，暂时不能发布")
        incomplete_choices = [
            item
            for item in raw.get("questions", [])
            if _question_type(item.get("question_type")) == "choice"
            and len(_normalize_options(item.get("options"))) < 2
        ]
        if incomplete_choices:
            numbers = "、".join(str(item.get("number", "?")) for item in incomplete_choices[:8])
            raise RuntimeError(f"选择题 {numbers} 缺少完整选项，请重新识别后再发布")
        timestamp = _now()
        self.update_homework(homework_id, status="published", published_at=timestamp)
        return self.get_homework(homework_id, role="teacher")

    def delete(self, homework_id: str) -> bool:
        homework_id = self.validate_homework_id(homework_id)
        with self._lock:
            state = self._read()
            before = len(state["homeworks"])
            state["homeworks"] = [item for item in state["homeworks"] if item.get("id") != homework_id]
            removed_submissions = [
                item for item in state["submissions"] if item.get("homework_id") == homework_id
            ]
            state["submissions"] = [
                item for item in state["submissions"] if item.get("homework_id") != homework_id
            ]
            if len(state["homeworks"]) == before:
                return False
            self._write(state)
        target = self._homework_dir(homework_id).resolve()
        if target.parent == self.root and target.exists():
            shutil.rmtree(target)
        for submission in removed_submissions:
            path = (self.root / "submissions" / str(submission.get("id"))).resolve()
            if path.parent == (self.root / "submissions").resolve() and path.exists():
                shutil.rmtree(path)
        return True

    def source_file(self, homework_id: str) -> tuple[dict[str, Any], Path]:
        raw = self.get_raw_homework(homework_id)
        path = self._homework_dir(homework_id) / str(raw["source_file"])
        if not path.is_file():
            raise FileNotFoundError("作业原始附件不存在")
        return raw, path

    def asset_file(self, homework_id: str, asset_name: str) -> Path:
        homework_dir = self._homework_dir(homework_id).resolve()
        asset_root = (homework_dir / "assets").resolve()
        if not ASSET_NAME_PATTERN.fullmatch(asset_name):
            raise ValueError("作业素材名称不合法")
        path = (asset_root / asset_name).resolve()
        if path.parent != asset_root or not path.is_file():
            raise FileNotFoundError("作业素材不存在")
        return path

    def create_submission(
        self,
        *,
        homework_id: str,
        student_id: str,
        files: list[tuple[str, str | None, bytes]],
    ) -> dict[str, Any]:
        raw = self.get_raw_homework(homework_id)
        if raw.get("status") != "published":
            raise RuntimeError("作业尚未发布")
        self.validate_student_id(student_id)
        if not files:
            raise ValueError("请至少上传一张作答图片")
        normalized_files: list[tuple[str, str, str | None, bytes]] = []
        for index, (filename, content_type, data) in enumerate(files, 1):
            safe_name = Path(filename).name or f"answer-{index}.jpg"
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ANSWER_IMAGE_SUFFIXES:
                raise ValueError(f"学生答案只支持图片：{suffix or '未知'}")
            normalized_files.append((safe_name, suffix, content_type, data))
        submission_id = uuid4().hex
        submission_dir = self.root / "submissions" / submission_id
        submission_dir.mkdir(parents=True, exist_ok=False)
        images: list[dict[str, Any]] = []
        for index, (safe_name, suffix, content_type, data) in enumerate(normalized_files, 1):
            stored_name = f"answer-{index:02d}{suffix}"
            (submission_dir / stored_name).write_bytes(data)
            images.append({
                "file": stored_name,
                "name": safe_name,
                "content_type": content_type or mimetypes.guess_type(safe_name)[0] or "image/jpeg",
                "size": len(data),
            })
        timestamp = _now()
        submission = {
            "id": submission_id,
            "homework_id": homework_id,
            "student_id": student_id,
            "student_name": "学生 1",
            "status": "grading",
            "answer_images": images,
            "extracted_answer": "",
            "grading": None,
            "review": None,
            "processing_error": "",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            state = self._read()
            state["submissions"].append(submission)
            self._write(state)
        return self._public_submission(submission)

    def get_raw_submission(self, submission_id: str) -> dict[str, Any]:
        self.validate_submission_id(submission_id)
        with self._lock:
            item = next(
                (value for value in self._read()["submissions"] if value.get("id") == submission_id),
                None,
            )
        if item is None:
            raise FileNotFoundError("学生提交不存在")
        return json.loads(json.dumps(item, ensure_ascii=False))

    def update_submission(self, submission_id: str, **updates: Any) -> None:
        self.validate_submission_id(submission_id)
        with self._lock:
            state = self._read()
            item = next(
                (value for value in state["submissions"] if value.get("id") == submission_id),
                None,
            )
            if item is None:
                raise FileNotFoundError("学生提交不存在")
            item.update(updates)
            item["updated_at"] = _now()
            self._write(state)

    def submission_file(self, submission_id: str, filename: str) -> Path:
        self.validate_submission_id(submission_id)
        if not ASSET_NAME_PATTERN.fullmatch(filename):
            raise ValueError("提交文件名不合法")
        root = (self.root / "submissions" / submission_id).resolve()
        path = (root / filename).resolve()
        if path.parent != root or not path.is_file():
            raise FileNotFoundError("提交图片不存在")
        return path


def _render_source(source_path: Path, assets_dir: Path) -> list[dict[str, Any]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".pdf":
        document = fitz.open(source_path)
        try:
            pages: list[dict[str, Any]] = []
            for page_index in range(document.page_count):
                page = document[page_index]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                image_path = assets_dir / f"page-{page_index + 1:03d}.png"
                pixmap.save(image_path)
                pages.append({
                    "page": page_index + 1,
                    "path": image_path,
                    "text": page.get_text("text")[:12000],
                    "width": pixmap.width,
                    "height": pixmap.height,
                    "native_answer_bboxes": _native_inline_answer_bboxes(page),
                })
            return pages
        finally:
            document.close()
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        image_path = assets_dir / "page-001.png"
        image.save(image_path, format="PNG")
        return [{
            "page": 1,
            "path": image_path,
            "text": "",
            "width": image.width,
            "height": image.height,
            "native_answer_bboxes": [],
        }]


def _native_inline_answer_bboxes(page: fitz.Page) -> list[list[float]]:
    """Locate filled content embedded between underline runs using PDF glyph boxes."""
    result: list[list[float]] = []
    raw = page.get_text("rawdict")
    page_width = max(float(page.rect.width), 1.0)
    page_height = max(float(page.rect.height), 1.0)
    for block in raw.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for line in block.get("lines", []):
            chars = [
                char
                for span in line.get("spans", [])
                for char in span.get("chars", [])
                if isinstance(char, dict) and isinstance(char.get("c"), str)
            ]
            text = "".join(char["c"] for char in chars)
            for match in re.finditer(r"[_＿]{2,}([^_＿\r\n]{1,32}?)[_＿]{2,}", text):
                content_start, content_end = match.span(1)
                content = text[content_start:content_end].strip()
                if not content or not re.search(r"[0-9A-Za-z\u3400-\u9fff]", content):
                    continue
                trim_left = len(text[content_start:content_end]) - len(text[content_start:content_end].lstrip())
                trim_right = len(text[content_start:content_end].rstrip())
                selected = chars[content_start + trim_left:content_start + trim_right]
                boxes = [char.get("bbox") for char in selected if len(char.get("bbox", [])) == 4]
                if not boxes:
                    continue
                left = min(float(box[0]) for box in boxes)
                top = min(float(box[1]) for box in boxes)
                right = max(float(box[2]) for box in boxes)
                bottom = max(float(box[3]) for box in boxes)
                pad_x, pad_y = 1.5, 1.0
                result.append([
                    round(max(0.0, left - pad_x) / page_width * 1000, 2),
                    round(max(0.0, top - pad_y) / page_height * 1000, 2),
                    round(min(page_width, right + pad_x) / page_width * 1000, 2),
                    round(min(page_height, bottom + pad_y) / page_height * 1000, 2),
                ])
    return result


def _normalized_regions(
    adapter: PDFExtractKitAdapter | Any, image: Image.Image
) -> list[dict[str, Any]]:
    try:
        rgb = np.asarray(image.convert("RGB"))
        regions = adapter.detect(rgb[:, :, ::-1].copy())
    except Exception as exc:
        logger.warning("PDF-Extract-Kit homework layout detection failed: %s", exc)
        return []
    width, height = image.size
    result: list[dict[str, Any]] = []
    for region in regions:
        bbox = getattr(region, "bbox_pixels", [])
        if len(bbox) != 4:
            continue
        result.append({
            "category": str(getattr(region, "category", "unknown")),
            "bbox": [
                round(float(bbox[0]) / width * 1000, 2),
                round(float(bbox[1]) / height * 1000, 2),
                round(float(bbox[2]) / width * 1000, 2),
                round(float(bbox[3]) / height * 1000, 2),
            ],
            "confidence": float(getattr(region, "confidence", 0)),
        })
    return result[:120]


def _page_prompt(
    page: dict[str, Any], regions: list[dict[str, Any]], previous_items: list[dict[str, Any]]
) -> str:
    return f"""你是高校电路课程作业内容提取器。当前是附件第 {page['page']} 页。附件可能是试卷、课后习题、习题册、学习指导书或扫描图片。
目标：只提取可直接布置给学生的独立题目，以及与每道题对应的参考答案。最终内容需要重新排版，不得把整页或题干截图当作学生题面。

坐标要求：所有 bbox 使用当前整页图片的归一化坐标 [left,top,right,bottom]，范围 0-1000。
内容筛选规则：
1. 只返回有明确题号或例题号、并包含提问/计算/证明/设计/选择等作答要求的独立题目。习题、练习题、思考题、自测题以及带完整题号和作答要求的例题都可以提取。
2. 必须忽略目录、前言、版权出版信息、教学要求、基本知识点、概念讲解、定理公式说明、例题之间的分析性过渡文字、章节总结、页眉页脚、广告、二维码和下载说明。不得把“解题方法介绍”或普通知识段落伪造成题目。若本页没有题目或题目答案，返回空 items。
3. “习题解答/参考答案”页面常把题干和“解：”放在一起：question_text 与 subquestions 只放题目，answer_text 与 answer_subquestions 只放解答。只有答案续页时沿用原 question_key，question_text 为空。

题目拆分规则：
4. question_key 必须在整份附件内唯一且稳定，例如“一-18”“二-2”“1.4-1.2.3”“例题-1.3.1”；number 必须忠实保留页面印刷的完整题号。跨页续题沿用原 question_key，新题即使版式相似也绝不能复用上一题 key。
5. question_type 是附件的客观事实，不得改题型。大题标明“选择题”时，其下每题必须是 choice；横线中已印有 A/B/C/D 是答案标记，不代表填空题。
6. choice 题必须完整返回页面上的 A/B/C/D 选项，放入 options，question_text 不包含选项。选项在下一页顶部续排时，即使本页没有重复题干，也要用原 question_key 返回一个 question_text 为空、但 options 完整的续接片段。
7. 多小问题必须结构化：question_text 只放所有小问共享的题干；每个“(1)/(2)/(3)”分别放入 subquestions，label 只写数字，text 不重复括号和共同题干。不要把多个小问挤在 question_text 的同一段。答案也用 answer_text + answer_subquestions 对齐拆分。subquestions 只能来自“解：/答案”之前实际印刷的提问；“解：”之后的假设、推导、分步计算即使也标有 (1)/(2)/(3)，只能进入 answer_subquestions，绝不能进入 subquestions 或泄露给学生。
8. question_text 只能转录当前页面肉眼可见的题干，不得从“最近已出现的题目”复制、改写或补全题干。若当前页只有上一题的题图、答案或评分过程，question_text 必须为空。
9. 使用 Markdown + LaTeX。所有电路变量、下标、希腊字母、单位和算式都必须放在 $...$ 中，例如 $\\beta=150$、$V_{{T}}=26\\,\\mathrm{{mV}}$、$V_{{BE(on)}}=0.7\\,\\mathrm{{V}}$、$r'_{{bb}}=100\\,\\Omega$、$R_{{B1}}=60\\,\\mathrm{{k}}\\Omega$、$A_{{v1}}=v_o/v_i$。禁止输出裸露的 V_T、R_B1、r_bb'、26mV 或 4kΩ。
10. 已填写答案的横线改回纯空白“______”，不得把答案字符写进题干。section_key 是大题、章节或习题组编号，section_title 是对应标题；没有明确分值时 points 返回 0。option_columns 按原页选项排布返回 1、2 或 4；figure_position 返回 before_question、after_question 或 after_options。
11. question_bboxes 只框题干与小问；figure_bboxes 必须单独精确框出题目引用的电路图、波形图或表格。答案中新增的推导图、页眉装饰图不要放入题目 figure_bboxes。
12. answer_bboxes 必须框出本页所有会泄露答案的区域，包括填在横线中的字母/数值、答案汇总表、“解：”之后的过程和评分说明；rubric 只保留明确的评分点。
13. 图必须归到使用它的题目，不能成为独立题目；若同一行有相邻题目的图，只框本题引用的图。

最近已出现的题目（用于判断跨页续接，不得覆盖页面上的新题号）：{json.dumps(previous_items[-12:], ensure_ascii=False)}
PDF 原生文本（可能为空或错序）：
{page['text'][:10000]}

PDF-Extract-Kit 检测区域：
{json.dumps(regions, ensure_ascii=False)}

仅返回 JSON：
{{"items":[{{"question_key":"1.4-1.2.1","section_key":"1.4","section_title":"1.4 习题解答","number":"1.2.1","question_type":"choice|calculation|short_answer|design|other","question_text":"所有小问共享的题干","subquestions":[{{"label":"1","text":"第一个小问"}},{{"label":"2","text":"第二个小问"}}],"options":[{{"label":"A","text":"选项内容"}}],"option_columns":2,"figure_position":"after_question","points":0,"question_bboxes":[[0,0,1000,1000]],"figure_bboxes":[[0,0,1000,1000]],"answer_bboxes":[[0,0,1000,1000]],"answer_text":"所有小问共享的答案说明","answer_subquestions":[{{"label":"1","text":"第一问答案"}},{{"label":"2","text":"第二问答案"}}],"rubric":"明确评分点"}}],"warnings":[]}}。"""


def _normalized_page_items(value: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    raw_items = value.get("items", value.get("questions", []))
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        key = _clean_text(raw.get("question_key", raw.get("id", "")), 80)
        number = _clean_text(raw.get("number", key), 80)
        if not key:
            continue
        question_text = _clean_text(raw.get("question_text", raw.get("prompt", "")))
        subquestions = _normalize_labeled_parts(raw.get("subquestions", []))
        parsed_question_text, parsed_subquestions = _split_labeled_text(question_text)
        if subquestions and parsed_subquestions:
            question_text = parsed_question_text
        elif not subquestions:
            question_text, subquestions = parsed_question_text, parsed_subquestions
        answer_text = _clean_text(raw.get("answer_text", raw.get("answer", "")))
        answer_subquestions = _normalize_labeled_parts(raw.get("answer_subquestions", []))
        parsed_answer_text, parsed_answer_subquestions = _split_labeled_text(answer_text)
        if answer_subquestions and parsed_answer_subquestions:
            answer_text = parsed_answer_text
        elif not answer_subquestions:
            answer_text, answer_subquestions = parsed_answer_text, parsed_answer_subquestions
        result.append({
            "question_key": key,
            "section_key": _clean_text(raw.get("section_key", ""), 40),
            "section_title": _clean_text(raw.get("section_title", ""), 240),
            "number": number or key,
            "question_type": _question_type(raw.get("question_type")),
            "question_text": question_text,
            "subquestions": subquestions,
            "options": _normalize_options(raw.get("options", [])),
            "option_columns": _option_columns(raw.get("option_columns")),
            "figure_position": _figure_position(raw.get("figure_position")),
            "points": max(0.0, _as_float(raw.get("points"))),
            "question_bboxes": _field_bboxes(raw, "question_bboxes", "question_bbox"),
            "figure_bboxes": _field_bboxes(raw, "figure_bboxes", "figure_bbox"),
            "answer_bboxes": _field_bboxes(raw, "answer_bboxes", "answer_bbox"),
            "answer_text": answer_text,
            "answer_subquestions": answer_subquestions,
            "rubric": _clean_text(raw.get("rubric", raw.get("scoring", ""))),
            "page": page_number,
        })
    return result


def _choice_recovery_prompt(page: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    compact = [
        {
            "question_key": item["question_key"],
            "number": item["number"],
            "page": item["page"],
            "text_start": item["question_text"][:300],
        }
        for item in candidates
    ]
    return f"""你是试卷选择题选项校对员。当前是第 {page['page']} 页，首次识别已确认下列题为选择题，但没有取得完整选项。
任务：
1. 只转录当前页面肉眼可见的 A/B/C/D 选项，不要改写、猜测或从答案反推选项。
2. 页首如果是上一页末尾题目的选项，必须归入候选列表中的原 question_key。
3. 横线里已印的 A/B/C/D 是标准答案，不要把它混入任何选项文本。
4. 选项中的变量、公式和单位使用 Markdown + LaTeX；option_columns 依原页布局只能是 1、2 或 4。
候选题：{json.dumps(compact, ensure_ascii=False)}
PDF 原生文本（只作辅助，以图像为准）：{page['text'][:8000]}
仅返回 JSON：
{{"recoveries":[{{"question_key":"一-3","number":"3","options":[{{"label":"A","text":"$1\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"B","text":"$2\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"C","text":"$4\\\\,\\\\mathrm{{k}}\\\\Omega$"}},{{"label":"D","text":"$5\\\\,\\\\mathrm{{k}}\\\\Omega$"}}],"option_columns":4}}]}}。"""


def _normalized_choice_recoveries(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = value.get("recoveries", value.get("items", []))
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        options = _normalize_options(raw.get("options"))
        if len(options) < 2:
            continue
        result.append({
            "question_key": _clean_text(raw.get("question_key"), 80),
            "number": _clean_text(raw.get("number"), 80),
            "options": options,
            "option_columns": _option_columns(raw.get("option_columns")),
        })
    return result


def _apply_choice_recoveries(
    recoveries: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> None:
    for recovery in recoveries:
        candidates = [
            item
            for item in targets
            if item["question_key"] == recovery["question_key"]
        ]
        if not candidates and recovery["number"]:
            candidates = [
                item
                for item in targets
                if item["number"] == recovery["number"]
                and _question_type(item["question_type"]) == "choice"
            ]
        if not candidates:
            continue
        target = candidates[-1]
        target["question_type"] = "choice"
        target["options"] = recovery["options"]
        target["option_columns"] = recovery["option_columns"]


def _repair_numbered_key(item: dict[str, Any]) -> str:
    key = str(item["question_key"])
    number = str(item.get("number", "")).strip()
    match = re.fullmatch(r"(.+?)[-—_](\d+)", key)
    if match and number.isdigit() and match.group(2) != number:
        return f"{match.group(1)}-{number}"
    return key


def _consolidate_question_keys(
    client: QwenVisionClient | Any,
    items: list[dict[str, Any]],
    *,
    page_count: int,
    page_contexts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run a whole-document pass to prevent cross-page key reuse across new questions."""
    if page_count <= 1 or len(items) <= 1:
        for item in items:
            item["question_key"] = _repair_numbered_key(item)
        return items, []
    compact = [
        {
            "segment_index": index,
            "page": item["page"],
            "raw_key": item["question_key"],
            "printed_number": item["number"],
            "current_points": item.get("points", 0),
            "question_type": item["question_type"],
            "text_start": _compose_labeled_text(
                item["question_text"], item.get("subquestions")
            )[:900],
            "has_question_bbox": bool(item["question_bboxes"]),
            "has_answer_bbox": bool(item["answer_bboxes"]),
        }
        for index, item in enumerate(items)
    ]
    has_point_evidence = any(_as_float(item.get("points")) > 0 for item in items)
    prompt = """你是整份作业附件的题号归并审查员。附件可能是试卷、习题册或学习指导书。下面是逐页提取的题目片段，逐页模型可能错误复用旧 question_key。
请为每个 segment_index 指定 canonical_key，保证：
1. 同一大题、章节或习题组内，页面印刷的新完整题号必须形成新 key，key 后缀必须等于 printed_number；点分题号（如 1.2.3）和例题号必须完整保留。
2. 只有明显属于上一页同一题的答案、解题过程或续接小问才沿用上一题 key；续接片段的 printed_number 可能被 OCR 误读，此时依据页面顺序和 text_start 判断。
3. 单纯换页不能切换章节前缀；连续题号序列（例如 1、2、3、4……20）必须使用同一前缀，即使 raw_key 的前缀被逐页模型误写。只有题号重新从 1 开始或出现明确的新大题标题时才切换中文章节前缀，例如“一-20”之后的计算题为“二-1”，下一部分重新从 1 开始时为“三-1”。
4. 根据页面计分说明校正 points：例如同一连续选择题部分写明“每空2分，共40分”，则该部分第1至20题都应为2分。只有 current_points 中已存在正分值或页面文字有明确分值证据时才能修改；普通习题册没有分值时 points 必须为0，禁止臆造每题2分。跨页续接片段沿用该题分值。
5. 对每个片段返回 keep。只有明确的独立题目、与题目对应的答案或跨页续接内容 keep=true；目录、教学要求、基本知识点、普通讲解、过渡文字等非题目内容 keep=false。
6. 不改题目内容；必须覆盖每个 segment_index。
页面开头原生文本（用于识别大题标题与计分说明）：
""" + json.dumps(page_contexts or [], ensure_ascii=False) + """
仅返回 JSON：{"assignments":[{"segment_index":0,"canonical_key":"一-1","points":2,"keep":true,"reason":"页面明确出现新题1"}]}。
题目片段：
""" + json.dumps(compact, ensure_ascii=False)
    try:
        result = client.complete_json(prompt)
    except Exception as exc:
        for item in items:
            item["question_key"] = _repair_numbered_key(item)
        return items, [f"全卷题号归并失败，已使用规则校正：{_clean_text(exc, 240)}"]
    raw_assignments = result.get("assignments", [])
    assignments: dict[int, tuple[str, float | None, bool]] = {}
    if isinstance(raw_assignments, list):
        for assignment in raw_assignments:
            if not isinstance(assignment, dict):
                continue
            try:
                index = int(assignment.get("segment_index"))
            except (TypeError, ValueError):
                continue
            key = _clean_text(assignment.get("canonical_key"), 80)
            if 0 <= index < len(items) and key:
                points_value = assignment.get("points")
                try:
                    points = float(points_value) if points_value is not None else None
                except (TypeError, ValueError):
                    points = None
                keep = _as_bool(assignment.get("keep", True))
                assignments[index] = (
                    key,
                    points
                    if has_point_evidence and points is not None and points > 0
                    else None,
                    keep,
                )
    warnings: list[str] = []
    if len(assignments) < len(items):
        warnings.append(
            f"全卷题号归并仅覆盖 {len(assignments)}/{len(items)} 个片段，未覆盖片段使用规则校正"
        )
    kept_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if index in assignments:
            key, points, keep = assignments[index]
            if not keep:
                continue
            item["question_key"] = key
            if points is not None:
                item["points"] = round(points, 2)
        else:
            item["question_key"] = _repair_numbered_key(item)
        kept_items.append(item)
    return kept_items, warnings


def _pixel_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, round(bbox[0] / 1000 * width))),
        max(0, min(height, round(bbox[1] / 1000 * height))),
        max(0, min(width, round(bbox[2] / 1000 * width))),
        max(0, min(height, round(bbox[3] / 1000 * height))),
    )


def _bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def _save_question_assets(
    *,
    assets_dir: Path,
    question_id: str,
    sequence: int,
    segments: list[dict[str, Any]],
    pages: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    figures: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, 1):
        figure_boxes = segment["figure_bboxes"]
        if not figure_boxes:
            continue
        page = pages.get(int(segment["page"]))
        if not page:
            continue
        with Image.open(page["path"]) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            sanitized = image.copy()
            draw = ImageDraw.Draw(sanitized)
            native_redactions = [
                bbox
                for bbox in page.get("native_answer_bboxes", [])
                if any(_bbox_intersects(bbox, figure_bbox) for figure_bbox in figure_boxes)
            ]
            redaction_boxes = segment["answer_bboxes"] + native_redactions
            for answer_bbox in redaction_boxes:
                answer_pixels = _pixel_bbox(answer_bbox, width, height)
                draw.rectangle(answer_pixels, fill="white", outline="#e8eceb", width=2)
            for figure_index, figure_bbox in enumerate(figure_boxes, 1):
                figure_pixels = _pixel_bbox(figure_bbox, width, height)
                left = max(0, figure_pixels[0] - 8)
                top = max(0, figure_pixels[1] - 8)
                right = min(width, figure_pixels[2] + 8)
                bottom = min(height, figure_pixels[3] + 8)
                figure_crop = sanitized.crop((left, top, right, bottom))
                if figure_crop.width < 8 or figure_crop.height < 8:
                    continue
                figure_name = (
                    f"question-{sequence:03d}-{question_id[:8]}-figure-"
                    f"{segment_index:02d}-{figure_index:02d}.png"
                )
                figure_crop.save(assets_dir / figure_name, format="PNG", optimize=True)
                figures.append({
                    "file": figure_name,
                    "page": segment["page"],
                    "width": figure_crop.width,
                    "height": figure_crop.height,
                    "source_top": figure_bbox[1],
                    "source_left": figure_bbox[0],
                    "position": segment.get("figure_position", "after_question"),
                })
    return [], figures


def process_homework(
    store: HomeworkStore,
    homework_id: str,
    *,
    client: QwenVisionClient | Any | None = None,
    layout_adapter: PDFExtractKitAdapter | Any | None = None,
) -> None:
    owned_client = False
    processing_dir: Path | None = None
    try:
        raw, source_path = store.source_file(homework_id)
        if client is None:
            if not settings.qwen_api_key:
                raise RuntimeError("未配置 QWEN_API_KEY，无法使用 qwen3-vl-plus 拆分作业")
            client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_homework_extraction_model,
                base_url=settings.qwen_base_url,
            )
            owned_client = True
        adapter = layout_adapter or PDFExtractKitAdapter()
        homework_dir = store._homework_dir(homework_id)
        assets_dir = homework_dir / "assets"
        if assets_dir.exists():
            resolved_assets = assets_dir.resolve()
            if resolved_assets.parent != homework_dir.resolve() or resolved_assets.name != "assets":
                raise RuntimeError("作业素材目录不安全")
            shutil.rmtree(resolved_assets)
        assets_dir.mkdir(parents=True, exist_ok=True)
        processing_dir = homework_dir / "processing"
        if processing_dir.exists():
            shutil.rmtree(processing_dir)
        pages = _render_source(source_path, processing_dir)
        store.update_homework(
            homework_id,
            page_count=len(pages),
            processing_progress=8,
            processing_message=f"已渲染 {len(pages)} 页，正在分析版面",
        )
        page_map = {int(page["page"]): page for page in pages}
        all_items: list[dict[str, Any]] = []
        warnings: list[str] = []
        previous_items: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages, 1):
            with Image.open(page["path"]) as image:
                regions = _normalized_regions(adapter, image)
                try:
                    result = client.complete_json(
                        _page_prompt(page, regions, previous_items),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                except Exception as exc:
                    warnings.append(f"第 {page['page']} 页视觉识别失败：{exc}")
                    logger.warning("Homework page %s extraction failed: %s", page["page"], exc)
                    continue
            page_items = _normalized_page_items(result, int(page["page"]))
            recovery_candidates: dict[str, dict[str, Any]] = {}
            for item in all_items + page_items:
                if (
                    _question_type(item["question_type"]) == "choice"
                    and len(item["options"]) < 2
                    and int(item["page"]) >= int(page["page"]) - 1
                ):
                    recovery_candidates.setdefault(item["question_key"], item)
            if recovery_candidates:
                try:
                    recovery_result = client.complete_json(
                        _choice_recovery_prompt(page, list(recovery_candidates.values())),
                        image_bytes=Path(page["path"]).read_bytes(),
                        image_mime="image/png",
                    )
                    _apply_choice_recoveries(
                        _normalized_choice_recoveries(recovery_result),
                        all_items + page_items,
                    )
                except Exception as exc:
                    warnings.append(
                        f"第 {page['page']} 页选择题选项补录失败：{_clean_text(exc, 240)}"
                    )
            all_items.extend(page_items)
            previous_items.extend({
                "page": item["page"],
                "key": item["question_key"],
                "section": item["section_key"],
                "number": item["number"],
                "text_start": _compose_labeled_text(
                    item["question_text"], item.get("subquestions")
                )[:180],
            } for item in page_items)
            raw_warnings = result.get("warnings", [])
            if isinstance(raw_warnings, list):
                warnings.extend(_clean_text(item, 240) for item in raw_warnings if _clean_text(item, 240))
            store.update_homework(
                homework_id,
                processing_progress=min(82, 8 + round(page_index / len(pages) * 74)),
                processing_message=f"正在识别第 {page_index}/{len(pages)} 页",
            )
        if not all_items:
            raise RuntimeError("视觉模型没有识别出题目，请检查附件清晰度或模型配置")

        all_items, consolidation_warnings = _consolidate_question_keys(
            client,
            all_items,
            page_count=len(pages),
            page_contexts=[
                {"page": page["page"], "text_start": page["text"][:1600]}
                for page in pages
            ],
        )
        warnings.extend(consolidation_warnings)
        if not all_items:
            raise RuntimeError("附件中没有识别到可直接布置的独立题目")
        choice_items: dict[str, list[dict[str, Any]]] = {}
        for item in all_items:
            if _question_type(item["question_type"]) == "choice":
                choice_items.setdefault(item["question_key"], []).append(item)
        incomplete_choices = [
            parts[0]
            for parts in choice_items.values()
            if not any(len(item["options"]) >= 2 for item in parts)
        ]
        if incomplete_choices:
            labels = [f"第{item['page']}页第{item['number']}题" for item in incomplete_choices]
            warnings.append(
                f"仍有 {len(incomplete_choices)} 道选择题缺少完整选项："
                + "、".join(labels[:12])
            )

        grouped: dict[str, dict[str, Any]] = {}
        for item_index, item in enumerate(all_items):
            key = item["question_key"]
            question = grouped.setdefault(key, {
                "id": hashlib.sha256(f"{homework_id}|{key}".encode("utf-8")).hexdigest()[:32],
                "section_key": key.rsplit("-", 1)[0] if "-" in key else item["section_key"],
                "section_title": item["section_title"],
                "number": item["number"],
                "question_type": item["question_type"],
                "points": item["points"],
                "prompt_parts": [],
                "subquestion_parts": [],
                "options": [],
                "option_columns": item["option_columns"],
                "figure_position": item["figure_position"],
                "answer_parts": [],
                "answer_subquestion_parts": [],
                "rubric_parts": [],
                "segments": [],
                "first_seen": item_index,
            })
            if item["question_text"] and item["question_text"] not in question["prompt_parts"]:
                question["prompt_parts"].append(item["question_text"])
            question["subquestion_parts"].extend(item.get("subquestions", []))
            if item["section_title"] and not question["section_title"]:
                question["section_title"] = item["section_title"]
            known_option_labels = {
                option["label"]: index for index, option in enumerate(question["options"])
            }
            for option in item["options"]:
                known_index = known_option_labels.get(option["label"])
                if known_index is None:
                    question["options"].append(option)
                    known_option_labels[option["label"]] = len(question["options"]) - 1
                elif len(option["text"]) > len(question["options"][known_index]["text"]):
                    question["options"][known_index] = option
            if item["options"]:
                question["question_type"] = "choice"
            if item["option_columns"] > question["option_columns"]:
                question["option_columns"] = item["option_columns"]
            if item["figure_bboxes"]:
                question["figure_position"] = item["figure_position"]
            if item["answer_text"] and item["answer_text"] not in question["answer_parts"]:
                question["answer_parts"].append(item["answer_text"])
            question["answer_subquestion_parts"].extend(item.get("answer_subquestions", []))
            if item["rubric"] and item["rubric"] not in question["rubric_parts"]:
                question["rubric_parts"].append(item["rubric"])
            if item["points"] > question["points"]:
                question["points"] = item["points"]
            question["segments"].append(item)

        questions: list[dict[str, Any]] = []
        for sequence, question in enumerate(
            sorted(grouped.values(), key=lambda item: int(item["first_seen"])), 1
        ):
            segments = question["segments"]
            layouts, figures = _save_question_assets(
                assets_dir=assets_dir,
                question_id=question["id"],
                sequence=sequence,
                segments=segments,
                pages=page_map,
            )
            pages_used = sorted({int(item["page"]) for item in segments})
            questions.append({
                "id": question["id"],
                "sequence": sequence,
                "section_key": question["section_key"],
                "section_title": question["section_title"] or (
                    f"{question['section_key']}、选择题"
                    if question["question_type"] == "choice"
                    else f"{question['section_key']}、题目"
                ),
                "number": question["number"],
                "question_type": question["question_type"],
                "prompt": _merge_prompt_parts(question["prompt_parts"]),
                "subquestions": _merge_labeled_parts(question["subquestion_parts"]),
                "options": question["options"],
                "option_columns": question["option_columns"],
                "figure_position": question["figure_position"],
                "points": question["points"],
                "answer": "\n".join(question["answer_parts"]).strip(),
                "answer_subquestions": _merge_labeled_parts(
                    question["answer_subquestion_parts"]
                ),
                "rubric": "\n".join(question["rubric_parts"]).strip(),
                "page_start": pages_used[0] if pages_used else None,
                "page_end": pages_used[-1] if pages_used else None,
                "layout_images": layouts,
                "figures": figures,
                "source_segments": segments,
            })
        max_score = round(sum(float(item.get("points", 0)) for item in questions), 2)
        store.update_homework(
            homework_id,
            status="draft",
            questions=questions,
            page_count=len(pages),
            max_score=max_score,
            processing_error="",
            processing_warnings=list(dict.fromkeys(warnings))[:30],
            processing_progress=100,
            processing_message="作业内容与参考答案的结构化数据已生成",
            extraction_schema_version=3,
        )
    except Exception as exc:
        logger.exception("Homework extraction failed for %s", homework_id)
        try:
            store.update_homework(
                homework_id,
                status="error",
                processing_error=_clean_text(exc, 1000),
                processing_progress=0,
                processing_message="识别失败",
            )
        except Exception:
            logger.exception("Unable to persist homework extraction failure")
    finally:
        if processing_dir is not None:
            resolved_processing = processing_dir.resolve()
            homework_dir = store._homework_dir(homework_id).resolve()
            if (
                resolved_processing.parent == homework_dir
                and resolved_processing.name == "processing"
                and resolved_processing.exists()
            ):
                shutil.rmtree(resolved_processing)
        if owned_client and client is not None:
            client.close()


def _answer_contact_sheet(paths: Iterable[Path], output_path: Path) -> Path:
    images: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((1600, 2200))
                images.append(image.copy())
        if not images:
            raise ValueError("没有可批改的答案图片")
        width = max(image.width for image in images) + 40
        total_height = sum(image.height + 54 for image in images) + 20
        scale = min(1.0, 7600 / max(total_height, 1))
        if scale < 1:
            images = [
                image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
                for image in images
            ]
            width = max(image.width for image in images) + 40
            total_height = sum(image.height + 54 for image in images) + 20
        sheet = Image.new("RGB", (width, total_height), "white")
        draw = ImageDraw.Draw(sheet)
        y = 18
        for index, image in enumerate(images, 1):
            draw.text((20, y), f"Submission image {index}", fill="#234744")
            y += 32
            sheet.paste(image, ((width - image.width) // 2, y))
            y += image.height + 22
        sheet.save(output_path, format="JPEG", quality=90, optimize=True)
        return output_path
    finally:
        for image in images:
            image.close()


def _grading_reference(homework: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": item.get("id"),
            "number": item.get("number"),
            "question": _compose_labeled_text(item.get("prompt"), item.get("subquestions")),
            "points": item.get("points"),
            "standard_answer": _compose_labeled_text(
                item.get("answer"), item.get("answer_subquestions")
            ),
            "rubric": item.get("rubric"),
        }
        for item in homework.get("questions", [])
    ]


def _normalize_grading(value: dict[str, Any], homework: dict[str, Any]) -> dict[str, Any]:
    references = {str(item.get("id")): item for item in homework.get("questions", [])}
    raw_items = value.get("items", [])
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            question_id = str(raw.get("question_id", ""))
            reference = references.get(question_id, {})
            max_score = _as_float(reference.get("points"), _as_float(raw.get("max_score")))
            score = max(0.0, min(max_score, _as_float(raw.get("score"))))
            items.append({
                "question_id": question_id,
                "number": reference.get("number", raw.get("number", "")),
                "student_answer": _clean_text(raw.get("student_answer", ""), 8000),
                "score": score,
                "max_score": max_score,
                "is_correct": _as_bool(raw.get("is_correct", score >= max_score and max_score > 0)),
                "feedback": _clean_text(raw.get("feedback", ""), 2000),
                "evidence": _clean_text(raw.get("evidence", ""), 2000),
            })
    total = round(sum(float(item["score"]) for item in items), 2)
    maximum = round(sum(_as_float(item.get("points")) for item in homework.get("questions", [])), 2)
    return {
        "items": items,
        "total_score": total,
        "max_score": maximum,
        "summary": _clean_text(value.get("summary", ""), 2000),
    }


def _normalize_review(value: dict[str, Any]) -> dict[str, Any]:
    issues = value.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    return {
        "passed": _as_bool(value.get("passed", False)),
        "confidence": max(0.0, min(1.0, _as_float(value.get("confidence")))),
        "issues": [_clean_text(item, 1000) for item in issues if _clean_text(item, 1000)][:30],
        "recommendation": _clean_text(value.get("recommendation", ""), 2000),
        "review_model": settings.qwen_homework_review_model,
    }


def grade_submission(
    store: HomeworkStore,
    submission_id: str,
    *,
    grading_client: QwenVisionClient | Any | None = None,
    review_client: QwenVisionClient | Any | None = None,
) -> None:
    owned_grader = False
    owned_reviewer = False
    try:
        submission = store.get_raw_submission(submission_id)
        homework = store.get_raw_homework(str(submission["homework_id"]))
        if grading_client is None or review_client is None:
            if not settings.qwen_api_key:
                raise RuntimeError("未配置 QWEN_API_KEY，无法自动批改作业")
        if grading_client is None:
            grading_client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_homework_grading_model,
                base_url=settings.qwen_base_url,
            )
            owned_grader = True
        if review_client is None:
            review_client = QwenVisionClient(
                api_key=settings.qwen_api_key,
                model=settings.qwen_homework_review_model,
                base_url=settings.qwen_base_url,
            )
            owned_reviewer = True
        submission_dir = store.root / "submissions" / submission_id
        answer_paths = [submission_dir / item["file"] for item in submission["answer_images"]]
        contact_sheet = _answer_contact_sheet(
            answer_paths, submission_dir / "answer-contact-sheet.jpg"
        )
        reference = _grading_reference(homework)
        grading_result = grading_client.complete_json(
            """你是高校电路课程阅卷教师。识别学生手写答案，并严格依据标准答案和评分点逐题评分。
不得因字迹风格扣分；计算题应按步骤给分；没有作答的题得 0 分；不要臆测图片中看不清的内容。
只返回 JSON：{"extracted_answer":"完整转写","items":[{"question_id":"...","number":"...","student_answer":"...","score":0,"max_score":0,"is_correct":false,"feedback":"...","evidence":"判分依据"}],"summary":"总评"}。
题目、标准答案与评分标准：\n"""
            + json.dumps(reference, ensure_ascii=False),
            image_bytes=contact_sheet.read_bytes(),
            image_mime="image/jpeg",
        )
        grading = _normalize_grading(grading_result, homework)
        extracted_answer = _clean_text(grading_result.get("extracted_answer", ""), 32000)
        review_result = review_client.complete_json(
            """你是独立的作业批改审查员。检查前一模型对学生答案的识别、逐题得分、步骤分和总分是否与标准答案及评分标准一致。
发现任何漏题、错读、加总错误或不合理扣分时 passed=false，并逐条说明；不要重新发明评分标准。
只返回 JSON：{"passed":true,"confidence":0.0,"issues":[],"recommendation":""}。
标准答案与评分标准：\n"""
            + json.dumps(reference, ensure_ascii=False)
            + "\n前一模型批改结果：\n"
            + json.dumps(grading, ensure_ascii=False),
            image_bytes=contact_sheet.read_bytes(),
            image_mime="image/jpeg",
        )
        review = _normalize_review(review_result)
        status = "graded" if review["passed"] else "review_required"
        store.update_submission(
            submission_id,
            status=status,
            extracted_answer=extracted_answer,
            grading=grading,
            review=review,
            processing_error="",
        )
    except Exception as exc:
        logger.exception("Homework submission grading failed for %s", submission_id)
        try:
            store.update_submission(
                submission_id,
                status="error",
                processing_error=_clean_text(exc, 1000),
            )
        except Exception:
            logger.exception("Unable to persist homework grading failure")
    finally:
        if owned_grader and grading_client is not None:
            grading_client.close()
        if owned_reviewer and review_client is not None:
            review_client.close()
