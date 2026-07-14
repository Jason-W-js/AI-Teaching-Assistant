from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.services.validated_knowledge import VALIDATED_KNOWLEDGE_CARDS


CATEGORY_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("基础电路分析", "基础电路分析", ("基尔霍夫", "kcl", "kvl", "节点", "回路", "戴维南", "诺顿", "叠加", "网络定理", "欧姆定律", "电阻")),
    ("半导体与器件", "半导体与器件", ("半导体", "pn结", "二极管", "稳压管", "晶体管", "场效应管", "载流子", "空穴", "热激发", "本征", "n型", "p型")),
    ("模拟电子电路", "模拟电子电路", ("放大", "运算放大器", "运放", "负反馈", "差分", "互补", "静态工作点", "频率特性", "比较器", "振荡电路", "波形", "运算电路", "乘法器", "转换电路", "锁相环", "有源滤波", "自动增益")),
    ("动态与频域电路", "动态与频域电路", ("rc", "rl", "rlc", "一阶电路", "二阶电路", "三要素", "时间常数", "暂态", "相量", "相位差", "正弦稳态", "频率响应", "传递函数", "谐振", "功率因数")),
    ("数字电路", "数字电路", ("逻辑", "触发器", "计数器", "寄存器", "编码器", "译码器", "状态机", "verilog")),
    ("工具与仿真", "工具与仿真", ("ewb", "eda", "电路仿真", "仿真分析", "仪器", "元器件模型库")),
    ("电源与能量变换", "电源与能量变换", ("整流", "滤波", "稳压", "直流电源", "功率电子", "逆变")),
)

CATEGORY_ORDER = {
    category_id: index for index, (category_id, _, _) in enumerate(CATEGORY_RULES)
}
OTHER_CATEGORY_ID = "其他知识"

_FIGURE_OR_LAYOUT_NOISE = re.compile(
    r"(?:如图|见图|图中|下图|上图|图\s*[A-Za-z]?\d|"
    r"如表|见表|表\s*[A-Za-z]?\d|本章讨论的问题|读图举例|"
    r"思考题|习题|扫描|版权所有|第\s*\d+\s*页)",
    re.I,
)
_DEFINITION_MARKERS = (
    "是指",
    "指的是",
    "称为",
    "定义为",
    "叫作",
    "叫做",
    "是一种",
    "是由",
    "组成",
    "形成",
)
_KEY_POINT_MARKERS = (
    "具有",
    "特点",
    "作用",
    "条件",
    "关系",
    "表现为",
    "可用",
    "因此",
    "当",
    "适用于",
)
_SECONDARY_CONTEXT_MARKERS = (
    "衬底",
    "场效应管",
    "晶体管",
    "集成电路",
    "晶闸管",
    "单结晶体管",
)

_OCR_GARBAGE = re.compile(
    r"(?:\ufffd|[0OoIl]{3,}|[A-Za-z0-9%+\-=]{9,}|"
    r"(?:[A-Za-z]\s*){8,}|[=~<>|^_%]{4,})"
)

_EMBEDDED_OCR_TOKEN = re.compile(
    r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[a-z])"
    r"[A-Za-z0-9]{2,}(?=[\u4e00-\u9fff])"
)

_OCR_CAPTION_FRAGMENT = re.compile(r"(?:\([a-z]\)|（[a-z]）|\d+(?:\.\d+){1,})", re.I)
_OCR_MIXED_IDENTIFIER = re.compile(r"(?:[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]{2,})")

_PARSER_PRIORITY = {
    "pymupdf": 30,
    "external-layout-ocr": 24,
    "docling": 24,
    "mineru": 24,
    "macos-vision-ocr": 8,
}

ALIASES = {
    "kcl": "基尔霍夫电流定律",
    "kvl": "基尔霍夫电压定律",
    "pn结": "PN结",
    "n型半导体": "N型半导体",
    "p型半导体": "P型半导体",
    "特二极管": "特殊二极管",
}


class KnowledgeGraphService:
    """Build a portable course graph from imported chunks, questions and mistakes."""

    def __init__(self, manager: KnowledgeBaseManager) -> None:
        self._manager = manager

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                items.append(value)
        return items

    @staticmethod
    def _clean_point(value: str) -> str:
        raw = value.strip()
        if re.match(r"^(?:[.．]\s*)?\d+\s*(?:[.．]\s*\d+)+", raw):
            return ""
        if re.match(r"^[A-Z]\s*(?:[.．]\s*)?\d+", raw, re.I) or re.match(
            r"^[A-Z]\s+[在中与]", raw, re.I
        ):
            return ""
        point = re.sub(r"^[\s.、第\d．]+", "", raw).strip(" 。：:;；")
        point = re.sub(r"^[一二三四五六七八九十]+[、.．]\s*", "", point).strip()
        if point in {"概述", "绪论", "基本电路", "基本分析方法"}:
            return ""
        if (
            not point
            or len(point) > 24
            or re.fullmatch(r"[\d.\s]+", point)
            or re.search(r"[,，。！？?=]", point)
            or "..." in point
            or "…" in point
            or "⋯" in point
            or re.match(r"^(?:第?[一二三四五六七八九十百0-9]+章|本章|习题|思考题)", point)
            or any(marker in point for marker in ("如图", "见图", "题解", "例题", "已知", "试求", "请判断", "通过调节", "改变"))
        ):
            return ""
        alias = ALIASES.get(point.casefold())
        return alias or point

    @staticmethod
    def _point_id(label: str) -> str:
        import hashlib

        return f"kp-{hashlib.sha1(label.casefold().encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def _clean_section(value: str) -> str:
        section = re.sub(r"\s+", " ", value).strip()
        if (
            not section
            or len(section) > 64
            or _FIGURE_OR_LAYOUT_NOISE.search(section)
            or re.search(r"[?？=]", section)
            or re.search(r"(?:^|[，,。；;])\s*(?:设|已知|试求|请|求)", section)
        ):
            return ""
        return section

    @staticmethod
    def _category(label: str, context: str = "") -> tuple[str, str]:
        label_text = label.casefold()
        for category_id, category_label, markers in CATEGORY_RULES:
            if any(marker.casefold() in label_text for marker in markers):
                return category_id, category_label
        context_text = context.casefold()
        for category_id, category_label, markers in CATEGORY_RULES:
            if any(marker.casefold() in context_text for marker in markers):
                return category_id, category_label
        return "其他知识", "其他知识"

    @staticmethod
    def _category_sort_key(category_id: str, label: str = "") -> tuple[int, str]:
        if category_id == OTHER_CATEGORY_ID:
            return len(CATEGORY_ORDER) + 1, label
        return CATEGORY_ORDER.get(category_id, len(CATEGORY_ORDER)), label

    @staticmethod
    def _normalize_ocr_sentence(sentence: str) -> str:
        sentence = _EMBEDDED_OCR_TOKEN.sub("", sentence)
        sentence = sentence.replace("输人", "输入").replace("品体管", "晶体管")
        sentence = sentence.replace("组M0合", "组合")
        sentence = re.sub(r"\s+([，。；：、])", r"\1", sentence)
        sentence = re.sub(r"([，。；：、])(?=[^\s])", r"\1", sentence)
        return re.sub(r"\s+", " ", sentence).strip()

    @staticmethod
    def _is_readable_sentence(sentence: str) -> bool:
        """Reject OCR fragments that should never become user-facing prose."""
        compact = re.sub(r"\s+", "", sentence)
        chinese = len(re.findall(r"[\u4e00-\u9fff]", compact))
        latin_digits = len(re.findall(r"[A-Za-z0-9]", compact))
        formula_symbols = len(re.findall(r"[=~<>|\\^_%{}\[\]]", compact))
        if chinese < 6 or _OCR_GARBAGE.search(compact):
            return False
        if latin_digits / max(len(compact), 1) > 0.28 and chinese < 28:
            return False
        if formula_symbols / max(len(compact), 1) > 0.08 or compact.count("=") > 2:
            return False
        if re.match(r"^(?:的值|值与|由式|式中[，,]|[，,。；;])", compact):
            return False
        if _OCR_CAPTION_FRAGMENT.search(compact) or _OCR_MIXED_IDENTIFIER.search(compact):
            return False
        # A digit glued into ordinary Chinese prose is usually a misrecognized
        # subscript or figure label. Keep only common measure-word contexts.
        if re.search(r"\d+(?=[\u4e00-\u9fff])", compact):
            for match in re.finditer(r"\d+(?=[\u4e00-\u9fff])", compact):
                next_char = compact[match.end() : match.end() + 1]
                if next_char not in "个级种次阶路管端倍位项章节点":
                    return False
        return True

    @classmethod
    def _source_sentences(cls, snippets: list[dict[str, str]]) -> list[tuple[int, str]]:
        """Return readable source sentences without page-layout or missing-figure noise."""
        sentences: list[tuple[int, str]] = []
        seen: set[str] = set()
        for snippet in snippets:
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", snippet.get("text", ""))
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            parser_score = _PARSER_PRIORITY.get(snippet.get("parser", ""), 12)
            for raw in re.split(r"(?<=[。！？；])\s*|\n+", text):
                sentence = re.sub(r"^[\s●•◆◇■□▪·.．\-—]+", "", raw).strip()
                sentence = re.sub(
                    r"^(?:(?:\d+\s*[.．]\s*)+\d*|[一二三四五六七八九十]+[、.．])\s*",
                    "",
                    sentence,
                ).strip()
                sentence = cls._normalize_ocr_sentence(sentence)
                compact = re.sub(r"\s+", "", sentence).casefold()
                if (
                    len(sentence) < 8
                    or len(sentence) > 220
                    or _FIGURE_OR_LAYOUT_NOISE.search(sentence)
                    or sentence.count("?") + sentence.count("？") > 0
                    or not cls._is_readable_sentence(sentence)
                    or compact in seen
                ):
                    continue
                seen.add(compact)
                sentences.append((parser_score, sentence))
        return sentences

    @classmethod
    def _summarize_point(
        cls, label: str, snippets: list[dict[str, str]]
    ) -> tuple[str, str, list[str]]:
        """Build an extractive, evidence-preserving node card.

        Curated definitions win for formula-heavy foundational concepts. Other
        definitions stay extractive, but only after passing the OCR quality gate.
        """
        validated = VALIDATED_KNOWLEDGE_CARDS.get(label)
        if validated:
            definition = validated["definition"]
            return definition, definition, list(validated["key_points"])

        sentences = cls._source_sentences(snippets)
        compact_label = re.sub(r"\s+", "", label).casefold()

        definition = ""
        definition_score = -1.0
        for parser_score, sentence in sentences:
            compact = re.sub(r"\s+", "", sentence).casefold()
            if compact_label not in compact:
                continue
            score = 3
            marker_pattern = "|".join(map(re.escape, _DEFINITION_MARKERS))
            if re.search(
                rf"(?:{re.escape(compact_label)}.{{0,12}}(?:{marker_pattern})|"
                rf"(?:{marker_pattern}).{{0,12}}{re.escape(compact_label)})",
                compact,
                re.I,
            ):
                score += 5
            if re.search(
                rf"形成.{{0,6}}{re.escape(compact_label)}(?:[。；,，]|$)",
                compact,
                re.I,
            ):
                score += 7
            if re.match(
                rf"{re.escape(compact_label)}(?:是指|指的是|是由|由|是一种|称为)",
                compact,
                re.I,
            ):
                score += 7
            if re.search(
                rf"(?:称为|叫作|叫做|定义为).{{0,10}}{re.escape(compact_label)}",
                compact,
                re.I,
            ):
                score += 4
            if re.search(rf"(?:对称|不对称|扩散|势垒){re.escape(compact_label)}", compact):
                score -= 7
            if 20 <= len(sentence) <= 130:
                score += 2
            # Parser trust resolves close candidates; it must not turn a mere
            # mention of the label into a definition by itself.
            score += parser_score / 100
            if score > definition_score:
                definition = sentence
                definition_score = score
        if definition_score < 8:
            definition = ""

        ranked: list[tuple[int, int, int, str]] = []
        for index, (parser_score, sentence) in enumerate(sentences):
            if sentence == definition:
                continue
            compact = re.sub(r"\s+", "", sentence).casefold()
            relevance = 0
            if compact_label in compact:
                relevance += 5
            relevance += sum(1 for marker in _KEY_POINT_MARKERS if marker in sentence)
            if relevance == 0:
                continue
            score = relevance
            score -= sum(
                3
                for marker in _SECONDARY_CONTEXT_MARKERS
                if marker not in label and marker in sentence
            )
            if 8 <= len(sentence) <= 40:
                score += 6
            elif len(sentence) <= 100:
                score += 4
            elif len(sentence) <= 120:
                score += 2
            else:
                continue
            if score >= 3:
                ranked.append((-score, -parser_score, index, sentence))
        ranked.sort()
        key_points = [item[3] for item in ranked[:2]]

        if definition:
            summary = definition
        elif key_points:
            summary = key_points[0]
        else:
            summary = f"该节点汇总了教材中与“{label}”相关的定义、性质和应用条件。"
        return summary, definition, key_points

    def _raw_material(self, knowledge_base: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        index_dir = self._manager.index_dir(knowledge_base)
        chunks = self._load_jsonl(index_dir / "chunks.jsonl")
        question_data = self._load_json(index_dir / "question_bank.json", {})
        questions = question_data.get("questions", []) if isinstance(question_data, dict) else []
        return chunks, [item for item in questions if isinstance(item, dict)]

    def infer_knowledge_points(self, text: str, knowledge_base: str = "default") -> list[str]:
        chunks, questions = self._raw_material(knowledge_base)
        candidates: set[str] = set()
        for item in [*chunks, *questions]:
            for raw_tag in item.get("knowledge_tags", []) or []:
                point = self._clean_point(str(raw_tag))
                if point:
                    candidates.add(point)
        compact = re.sub(r"\s+", "", text).casefold()
        matches = [
            point
            for point in sorted(candidates, key=len, reverse=True)
            if re.sub(r"\s+", "", point).casefold() in compact
        ]
        return matches[:6]

    def build(
        self, knowledge_base: str = "default", wrong_questions: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        chunks, questions = self._raw_material(knowledge_base)
        manifest = self._load_json(
            self._manager.index_dir(knowledge_base) / "source_manifest.json", {}
        )
        manifest_sources = manifest.get("sources", []) if isinstance(manifest, dict) else []
        parser_by_source = {
            str(item.get("source", "")): str(item.get("parser", ""))
            for item in manifest_sources
            if isinstance(item, dict)
        }
        entries: dict[str, dict[str, Any]] = {}

        def ensure_point(label: str, context: str = "") -> dict[str, Any] | None:
            cleaned = self._clean_point(label)
            if not cleaned:
                return None
            point_id = self._point_id(cleaned)
            if point_id not in entries:
                category_id, category_label = self._category(cleaned, context)
                entries[point_id] = {
                    "id": point_id,
                    "type": "knowledge_point",
                    "label": cleaned,
                    "category_id": category_id,
                    "category_label": category_label,
                    "summary": "",
                    "definition": "",
                    "key_points": [],
                    "sources": {},
                    "sections": set(),
                    "questions": [],
                    "wrong_questions": [],
                    "chunk_count": 0,
                    "_snippets": [],
                }
            return entries[point_id]

        for chunk in chunks:
            if chunk.get("doc_type") == "question":
                continue
            context = f"{chunk.get('chapter', '')} {chunk.get('section', '')}"
            for raw_tag in chunk.get("knowledge_tags", []) or []:
                entry = ensure_point(str(raw_tag), context)
                if entry is None:
                    continue
                source = str(chunk.get("source", "")).strip()
                if source:
                    entry["sources"][source] = entry["sources"].get(source, 0) + 1
                section = self._clean_section(
                    str(chunk.get("section") or chunk.get("chapter") or "")
                )
                if section:
                    entry["sections"].add(section)
                snippet = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
                if snippet:
                    entry["_snippets"].append(
                        {
                            "text": snippet[:2400],
                            "source": source,
                            "parser": parser_by_source.get(source, ""),
                        }
                    )
                entry["chunk_count"] += 1

        for question in questions:
            for raw_tag in question.get("knowledge_tags", []) or []:
                entry = ensure_point(str(raw_tag), str(question.get("question_text", "")))
                if entry is not None and len(entry["questions"]) < 8:
                    entry["questions"].append(
                        {
                            "id": str(question.get("question_id", "")),
                            "title": str(question.get("question_text", "")),
                            "difficulty": str(question.get("difficulty", "")),
                        }
                    )

        for wrong in wrong_questions or []:
            points = wrong.get("knowledge_points", []) or []
            if not points:
                points = ["待归类"]
            for raw_point in points:
                entry = ensure_point(str(raw_point))
                if entry is not None:
                    entry["wrong_questions"].append(
                        {
                            "id": str(wrong.get("id", "")),
                            "title": str(wrong.get("title", "未命名错题")),
                            "updated_at": str(wrong.get("updated_at", "")),
                        }
                    )

        categories: dict[str, dict[str, Any]] = {}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        root_id = f"course-{knowledge_base}"
        for entry in sorted(
            entries.values(),
            key=lambda item: (
                self._category_sort_key(item["category_id"], item["category_label"]),
                item["label"],
            ),
        ):
            category_id = entry["category_id"]
            categories.setdefault(
                category_id,
                {"id": f"category-{category_id}", "label": entry["category_label"], "count": 0},
            )["count"] += 1
            summary, definition, key_points = self._summarize_point(
                entry["label"], entry.pop("_snippets")
            )
            entry["summary"] = summary
            entry["definition"] = definition
            entry["key_points"] = key_points
            entry["sources"] = [
                {"name": name, "chunks": count}
                for name, count in sorted(entry["sources"].items())
            ]
            entry["sections"] = sorted(entry["sections"])[:12]
            nodes.append(entry)
            edges.append({"source": f"category-{category_id}", "target": entry["id"], "type": "contains"})
            for wrong in entry["wrong_questions"]:
                edges.append({"source": entry["id"], "target": f"wrong-{wrong['id']}", "type": "has_wrong_question"})

        category_list = sorted(
            categories.values(),
            key=lambda item: self._category_sort_key(
                item["id"].removeprefix("category-"), item["label"]
            ),
        )
        for category in category_list:
            edges.append({"source": root_id, "target": category["id"], "type": "contains"})
        return {
            "knowledge_base": knowledge_base,
            "root": {"id": root_id, "label": "电路课程知识体系"},
            "categories": category_list,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "sources": len(
                    {
                        str(item.get("source"))
                        for item in [*chunks, *questions]
                        if item.get("source")
                    }
                ),
                "knowledge_points": len(nodes),
                "questions": len(questions),
                "wrong_questions": len(wrong_questions or []),
            },
        }
