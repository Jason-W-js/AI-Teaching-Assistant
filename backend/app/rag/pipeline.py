from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import faiss
import fitz
import numpy as np
from docx import Document
from openpyxl import load_workbook
from sentence_transformers import SentenceTransformer

from backend.app.rag.models import PageDocument, TextChunk


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".xlsx", ".json"}
AD_NOISE = (
    "扫码关注",
    "微信公众号",
    "关注公众号",
    "购买正版",
    "资源下载",
    "广告",
)
TAG_KEYWORDS = (
    "本征半导体",
    "N型半导体",
    "P型半导体",
    "PN结",
    "二极管",
    "稳压二极管",
    "晶体管",
    "场效应管",
    "静态工作点",
    "伏安特性",
    "单向导电性",
    "反向击穿",
    "欧姆定律",
    "基尔霍夫定律",
    "戴维南定理",
)


def _normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace("\u200b", "")
    line = re.sub(r"https?\s*:\s*[/\\]+\s*\S+", "", line, flags=re.I)
    line = re.sub(r"www\s*\.\s*\S+", "", line, flags=re.I)
    line = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z0-9])", "", line)
    line = re.sub(r"(?<=[0-9])\s+(?=[A-Za-z])", "", line)
    line = re.sub(r"(?<=[A-Za-z])\s+(?=[0-9])", "", line)
    line = re.sub(r"P\s*N\s*结", "PN结", line, flags=re.I)
    line = re.sub(r"([NP])\s*型", r"\1型", line, flags=re.I)
    line = re.sub(r"M\s*O\s*S", "MOS", line, flags=re.I)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _edge_noise(raw_pages: list[str]) -> set[str]:
    candidates: Counter[str] = Counter()
    for text in raw_pages:
        lines = [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]
        for line in lines[:2] + lines[-2:]:
            if len(line) <= 80:
                candidates[line] += 1
    threshold = max(3, int(len(raw_pages) * 0.08))
    return {line for line, count in candidates.items() if count >= threshold}


def clean_page_text(text: str, repeated_noise: set[str] | None = None) -> str:
    repeated_noise = repeated_noise or set()
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line or line in repeated_noise:
            continue
        if re.fullmatch(r"[-—·•\s]*\d{1,4}[-—·•\s]*", line):
            continue
        if any(noise in line for noise in AD_NOISE):
            continue
        if sum(char == "�" for char in line) > 1:
            continue
        cleaned_lines.append(line)

    paragraphs: list[str] = []
    buffer = ""
    heading_pattern = re.compile(
        r"^(?:第[一二三四五六七八九十百0-9]+章|\d+(?:\.\d+){0,3}\s+|本章小结|自测题|习题)"
    )
    for line in cleaned_lines:
        is_heading = bool(heading_pattern.match(line)) and len(line) < 70
        if is_heading:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(line)
            continue
        buffer += line
        if line.endswith(("。", "！", "？", ":", "；")) or len(buffer) >= 260:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return "\n\n".join(paragraphs)


def _is_chapter_title(title: str) -> bool:
    return bool(re.match(r"^第[一二三四五六七八九十百0-9]+章", title.replace(" ", "")))


def extract_pdf(path: Path, chapter_limit: int | None = None) -> list[PageDocument]:
    document = fitz.open(path)
    toc = [item for item in document.get_toc(simple=True) if len(item) >= 3]
    chapters = [(str(title).strip(), int(page)) for level, title, page in toc if level == 1 and _is_chapter_title(str(title).strip())]
    if chapters:
        start_page = chapters[0][1]
        if chapter_limit and len(chapters) > chapter_limit:
            end_page = chapters[chapter_limit][1] - 1
        else:
            end_page = document.page_count
    else:
        start_page, end_page = 1, document.page_count

    raw_pages = [document[index - 1].get_text("text") for index in range(start_page, end_page + 1)]
    repeated_noise = _edge_noise(raw_pages)
    page_docs: list[PageDocument] = []
    for page_number, raw_text in zip(range(start_page, end_page + 1), raw_pages):
        current_chapter = ""
        current_section = ""
        for level, title, toc_page in toc:
            if int(toc_page) > page_number:
                break
            title = _normalize_line(re.sub(r"\s+", " ", str(title)).strip())
            if level == 1 and _is_chapter_title(title):
                current_chapter = title
                current_section = ""
            elif level >= 2:
                current_section = title
        text = clean_page_text(raw_text, repeated_noise)
        if len(text) < 30:
            continue
        page_docs.append(
            PageDocument(
                text=text,
                source=path.name,
                page=page_number,
                chapter=current_chapter or path.stem,
                section=current_section or current_chapter or path.stem,
            )
        )
    document.close()
    return page_docs


def extract_markdown_or_text(path: Path) -> list[PageDocument]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chapter = path.stem
    section = path.stem
    documents: list[PageDocument] = []
    page = 1
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        cleaned = clean_page_text("\n".join(buffer))
        if cleaned:
            documents.append(PageDocument(cleaned, path.name, page, chapter, section))
        buffer = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if heading:
            flush()
            title = heading.group(2).strip()
            if len(heading.group(1)) == 1:
                chapter = title
            section = title
        elif line.strip() == "\f":
            flush()
            page += 1
        else:
            buffer.append(line)
    flush()
    return documents


def extract_docx(path: Path) -> list[PageDocument]:
    document = Document(path)
    chapter = path.stem
    section = path.stem
    blocks: list[PageDocument] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = clean_page_text("\n".join(buffer))
        if text:
            blocks.append(PageDocument(text, path.name, 1, chapter, section))
        buffer = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if "heading" in style or "标题" in style:
            flush()
            section = text
            if style.endswith("1"):
                chapter = text
        else:
            buffer.append(text)
    flush()
    return blocks


def _split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[,，、;；|]", str(value or "")) if item.strip()]


QUESTION_HEADER_ALIASES = {
    "question_id": ("题号", "question_id", "id"),
    "question_text": ("题目文本", "题目", "question_text"),
    "knowledge_tags": ("知识点标签", "知识点", "knowledge_tags"),
    "standard_answer": ("标准答案", "答案", "standard_answer"),
    "common_mistakes": ("易错点", "common_mistakes"),
    "difficulty": ("难度", "difficulty"),
    "question_type": ("题型", "question_type"),
    "solution_steps": ("解题步骤", "解析", "solution_steps"),
}


def extract_question_xlsx(path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []
    header_index = next(
        (
            index
            for index, row in enumerate(rows[:12])
            if "题号" in [str(value or "").strip() for value in row]
            and "题目文本" in [str(value or "").strip() for value in row]
        ),
        0,
    )
    headers = [str(value or "").strip() for value in rows[header_index]]
    mapping: dict[str, int] = {}
    for field, aliases in QUESTION_HEADER_ALIASES.items():
        for alias in aliases:
            if alias in headers:
                mapping[field] = headers.index(alias)
                break
    required = {"question_id", "question_text", "knowledge_tags", "standard_answer", "common_mistakes"}
    if not required.issubset(mapping):
        workbook.close()
        return []
    questions: list[dict[str, Any]] = []
    for row in rows[header_index + 1 :]:
        if not row or not row[mapping["question_text"]]:
            continue
        item = {
            field: (row[index] if index < len(row) else "")
            for field, index in mapping.items()
        }
        item["question_id"] = str(item["question_id"])
        item["question_text"] = str(item["question_text"]).strip()
        item["knowledge_tags"] = _split_tags(item["knowledge_tags"])
        item["standard_answer"] = str(item["standard_answer"] or "").strip()
        item["common_mistakes"] = str(item["common_mistakes"] or "").strip()
        item["difficulty"] = str(item.get("difficulty") or "基础").strip()
        item["question_type"] = str(item.get("question_type") or "综合题").strip()
        item["solution_steps"] = str(item.get("solution_steps") or "").strip()
        item["source"] = path.name
        questions.append(item)
    workbook.close()
    return questions


def extract_question_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        return []
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        return []
    questions = []
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not item.get("question_text"):
            continue
        normalized = dict(item)
        normalized.setdefault("question_id", f"JSON-{index:03d}")
        normalized["knowledge_tags"] = _split_tags(normalized.get("knowledge_tags"))
        normalized.setdefault("standard_answer", "")
        normalized.setdefault("common_mistakes", "")
        normalized.setdefault("difficulty", "基础")
        normalized.setdefault("question_type", "综合题")
        normalized.setdefault("solution_steps", "")
        normalized["source"] = path.name
        questions.append(normalized)
    return questions


def _knowledge_tags(text: str, section: str) -> list[str]:
    tags = [keyword for keyword in TAG_KEYWORDS if keyword.lower() in text.lower()]
    normalized_section = re.sub(r"^\d+(?:\.\d+)*\s*", "", section).strip()
    if normalized_section and normalized_section not in tags:
        tags.insert(0, normalized_section)
    return tags[:8]


def _sentence_pieces(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        sentences = [piece for piece in re.split(r"(?<=[。！？；])", paragraph) if piece]
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > max_chars:
                pieces.append(current)
                current = ""
            if len(sentence) > max_chars:
                pieces.extend(
                    sentence[index : index + max_chars]
                    for index in range(0, len(sentence), max_chars)
                )
                continue
            current += sentence
        if current:
            pieces.append(current)
    return pieces


def chunk_documents(documents: Iterable[PageDocument], max_chars: int = 900) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        pieces = _sentence_pieces(document.text, max_chars=max_chars)
        current: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current, current_length
            if not current:
                return
            text = "\n\n".join(current).strip()
            raw_id = f"{document.source}|{document.page}|{document.section}|{text[:120]}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=text,
                    source=document.source,
                    chapter=document.chapter,
                    section=document.section,
                    page_start=document.page,
                    page_end=document.page,
                    doc_type=document.doc_type,
                    knowledge_tags=_knowledge_tags(text, document.section),
                )
            )
            overlap = text[-140:] if len(text) > 140 else ""
            current = [overlap] if overlap else []
            current_length = len(overlap)

        for piece in pieces:
            if current and current_length + len(piece) > max_chars:
                flush()
            if current and current_length + len(piece) > max_chars:
                current = []
                current_length = 0
            current.append(piece)
            current_length += len(piece)
        flush()
    return chunks


def question_chunks(questions: Iterable[dict[str, Any]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for item in questions:
        tags = _split_tags(item.get("knowledge_tags"))
        text = (
            f"题目：{item.get('question_text', '')}\n"
            f"标准答案：{item.get('standard_answer', '')}\n"
            f"解题步骤：{item.get('solution_steps', '')}\n"
            f"易错点：{item.get('common_mistakes', '')}"
        ).strip()
        question_id = str(item.get("question_id", "Q"))
        chunks.append(
            TextChunk(
                id=f"question-{question_id}",
                text=text,
                source=str(item.get("source", "question_bank.json")),
                chapter="示例题库",
                section="、".join(tags) or "综合",
                page_start=None,
                page_end=None,
                doc_type="question",
                knowledge_tags=tags,
            )
        )
    return chunks


def _write_clean_markdown(path: Path, documents: list[PageDocument], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {path.stem}", "", f"> 清洗来源：{path.name}", ""]
    last_chapter = last_section = ""
    for document in documents:
        if document.chapter and document.chapter != last_chapter:
            lines.extend([f"## {document.chapter}", ""])
            last_chapter = document.chapter
        if document.section and document.section != last_section and document.section != document.chapter:
            lines.extend([f"### {document.section}", ""])
            last_section = document.section
        lines.extend([f"<!-- source={document.source}; page={document.page} -->", document.text, ""])
    (output_dir / f"{path.stem}.clean.md").write_text("\n".join(lines), encoding="utf-8")


def build_knowledge_base(
    resources_dir: Path,
    output_dir: Path,
    embedding_model_path: Path,
    *,
    chapter_limit: int | None = None,
) -> dict[str, Any]:
    resources_dir = resources_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = output_dir / "cleaned_documents"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    documents: list[PageDocument] = []
    questions: list[dict[str, Any]] = []
    source_files = [
        path for path in sorted(resources_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    for path in source_files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            extracted = extract_pdf(path, chapter_limit)
            documents.extend(extracted)
            _write_clean_markdown(path, extracted, cleaned_dir)
        elif suffix in {".md", ".txt"}:
            extracted = extract_markdown_or_text(path)
            documents.extend(extracted)
            _write_clean_markdown(path, extracted, cleaned_dir)
        elif suffix == ".docx":
            extracted = extract_docx(path)
            documents.extend(extracted)
            _write_clean_markdown(path, extracted, cleaned_dir)
        elif suffix == ".xlsx":
            questions.extend(extract_question_xlsx(path))
        elif suffix == ".json":
            questions.extend(extract_question_json(path))

    structured_questions = {"schema_version": "1.0", "questions": questions}
    (output_dir / "question_bank.json").write_text(
        json.dumps(structured_questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chunks = chunk_documents(documents) + question_chunks(questions)
    if not chunks:
        raise RuntimeError(f"在 {resources_dir} 中没有提取到可索引内容")

    chunk_path = output_dir / "chunks.jsonl"
    chunk_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks),
        encoding="utf-8",
    )

    model = SentenceTransformer(str(embedding_model_path), device="cpu")
    embedding_texts = [
        "\n".join(
            filter(
                None,
                [chunk.doc_type, chunk.chapter, chunk.section, " ".join(chunk.knowledge_tags), chunk.text],
            )
        )
        for chunk in chunks
    ]
    embeddings = model.encode(
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # FAISS' Windows file writer cannot open paths containing Chinese characters.
    # Serialize in memory and let Python handle the Unicode path instead.
    serialized_index = faiss.serialize_index(index)
    (output_dir / "vectors.faiss").write_bytes(serialized_index.tobytes())

    metadata = {
        "state": "populated",
        "resource_dir": str(resources_dir),
        "embedding_model": str(embedding_model_path),
        "dimension": int(embeddings.shape[1]),
        "documents": len(source_files),
        "text_pages": len(documents),
        "questions": len(questions),
        "chunks": len(chunks),
        "chapter_limit": chapter_limit,
        "sources": [path.name for path in source_files],
    }
    (output_dir / "index_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Knowledge base populated: %s", metadata)
    return metadata
