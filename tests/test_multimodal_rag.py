from __future__ import annotations

import io

import fitz
from PIL import Image, ImageDraw

from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.pdf_extract_kit import PDFExtractKitAdapter
from backend.app.rag.multimodal import (
    LayoutElement,
    _analyze_image,
    _normalize_circuit_result,
    _safe_partial_noise_fragment,
    build_local_knowledge_graph,
    enhance_pdf,
)
from backend.app.services.qwen_multimodal_client import QwenMultimodalAPIError


def _diagram_png() -> bytes:
    image = Image.new("RGB", (420, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 110, 120, 110), fill="black", width=3)
    draw.rectangle((120, 80, 220, 140), outline="black", width=3)
    draw.line((220, 110, 390, 110), fill="black", width=3)
    draw.line((30, 110, 30, 190, 390, 190, 390, 110), fill="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pdf_fallback_preserves_layout_and_image_metadata(tmp_path, monkeypatch):
    # Unit tests exercise the auditable fallback without loading 400 MB GPU models.
    monkeypatch.setattr(PDFExtractKitAdapter, "detect", lambda _self, _image: [])
    pdf_path = tmp_path / "lesson.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=500, height=700)
    page.insert_text((40, 50), "R1 + R2 = 3 ohm")
    page.insert_image(fitz.Rect(40, 100, 460, 320), stream=_diagram_png())
    pdf.save(pdf_path)
    pdf.close()

    docs = [PageDocument("R1 + R2 = 3 ohm", pdf_path.name, 1, "chapter", "section")]
    kept, elements, audit = enhance_pdf(pdf_path, docs, tmp_path / "index")

    assert kept == docs
    assert audit[0]["keep"] is True
    assert all(len(element.bbox) == 4 for element in elements)
    image = next(element for element in elements if element.image_path)
    assert (tmp_path / "index" / image.image_path).exists()
    assert image.page == 1
    assert image.content_hash


def test_multimodal_chunk_and_graph_keep_circuit_relationships():
    chunk = TextChunk(
        id="circuit-1",
        text="R1 连接节点 n1 与 n2",
        source="lesson.pdf",
        chapter="第一章",
        section="串联电路",
        page_start=8,
        page_end=8,
        doc_type="multimodal",
        knowledge_tags=["电阻", "串联"],
        element_type="circuit",
        multimodal={
            "components": [{"id": "R1", "type": "resistor", "terminals": ["n1", "n2"]}],
            "nets": [{"id": "n1", "terminals": ["R1.1"]}],
        },
    )
    graph = build_local_knowledge_graph([chunk])

    assert any(node["type"] == "component" for node in graph["nodes"])
    assert any(edge["type"] == "MENTIONS" for edge in graph["edges"])
    assert any(edge["type"] == "CONTAINS" for edge in graph["edges"])
    assert any(edge["type"] == "CONNECTED_TO" for edge in graph["edges"])


def test_graph_separates_documents_pages_concepts_and_components():
    chunk = TextChunk(
        id="circuit-pages",
        text="第101页共射放大电路，Rb设置静态工作点。",
        source="analog_electronics_pages_101_103.pdf",
        chapter="analog_electronics_pages_101_103",
        section="analog_electronics_pages_101_103",
        page_start=101,
        page_end=101,
        doc_type="multimodal",
        knowledge_tags=["analog_electronics_pages_101_103", "formula", "晶体管", "静态工作点"],
        element_type="circuit",
        multimodal={
            "components": [{"id": "Rb", "type": "resistor", "terminals": ["n1", "n2"]}],
            "nets": [{"id": "n1", "terminals": ["Rb.1"]}],
        },
    )
    graph = build_local_knowledge_graph([chunk])
    names_by_type = {
        kind: {node["name"] for node in graph["nodes"] if node["type"] == kind}
        for kind in ("document", "page", "concept", "component")
    }
    assert "第 101–103 页教材节选" in names_by_type["document"]
    assert "第 101 页" in names_by_type["page"]
    assert {"晶体管", "静态工作点", "电阻"}.issubset(names_by_type["concept"])
    assert "formula" not in names_by_type["concept"]
    assert "analog_electronics_pages_101_103" not in names_by_type["concept"]
    assert "Rb" in names_by_type["component"]


def test_malformed_vision_json_is_safely_normalized():
    value = _normalize_circuit_result({
        "is_circuit": "false",
        "components": ["R1", {"id": "R2", "type": "resistor"}],
        "nets": [None, {"id": "n1"}],
    })
    assert value["is_circuit"] is False
    assert value["components"] == [{"id": "R2", "type": "resistor"}]
    assert value["nets"] == [{"id": "n1"}]


def test_missing_netlist_is_synthesized_without_inventing_values():
    value = _normalize_circuit_result({
        "is_circuit": True,
        "components": [
            {
                "id": "R1",
                "type": "resistor",
                "value": None,
                "terminals": ["n1", "n2"],
            }
        ],
        "nets": [{"id": "n1", "terminals": ["R1.1"]}],
        "netlist": "",
    })

    assert value["netlist"].startswith("* Generated from Qwen3-VL")
    assert "R1 n1 n2 UNKNOWN" in value["netlist"]


def test_partial_cleaning_only_accepts_explicit_publishing_noise():
    assert _safe_partial_noise_fragment("版权所有，扫码关注公众号") is True
    assert _safe_partial_noise_fragment("Q 是英文 Quiescent 的字头") is False
    assert _safe_partial_noise_fragment("2.2 基本共射放大电路的工作原理") is False


def test_waveform_figure_is_not_promoted_to_circuit_when_vision_fails(monkeypatch):
    class FailedVision:
        model = "qwen3-vl-flash"

        def complete_json(self, *_args, **_kwargs):
            raise QwenMultimodalAPIError("invalid json")

    monkeypatch.setattr(
        "backend.app.rag.multimodal._circuit_image_heuristic",
        lambda _image: (True, 0.85),
    )
    element = LayoutElement(
        id="wave",
        source="lesson.pdf",
        page=103,
        element_type="image",
        bbox=[0, 0, 100, 100],
        nearby_text="图2.2.3 基本共射放大电路的波形分析",
    )
    _analyze_image(element, _diagram_png(), FailedVision())
    assert element.element_type == "image"
    assert element.components == []


def test_index_activation_replaces_complete_directory(tmp_path):
    final = tmp_path / "default"
    staging = tmp_path / ".default.building-test"
    final.mkdir()
    staging.mkdir()
    (final / "version.txt").write_text("old", encoding="utf-8")
    (staging / "version.txt").write_text("new", encoding="utf-8")

    KnowledgeBaseManager._activate_index(final, staging)

    assert (final / "version.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
