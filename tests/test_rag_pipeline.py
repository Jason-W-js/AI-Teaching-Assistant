from backend.app.rag.models import PageDocument
from backend.app.rag.pipeline import (
    _infer_ocr_hierarchy,
    _load_cached_pdf,
    _write_cached_pdf,
    chunk_documents,
    classify_document,
    clean_page_text,
    extract_questions_from_documents,
    list_source_files,
)


def test_clean_page_text_removes_noise_and_page_number():
    text = "1.2 半导体二极管\n13\n访问 http://www.example.com 下载\nPN 结具有单向导电性。\n扫码关注公众号"
    cleaned = clean_page_text(text)
    assert "13" not in cleaned
    assert "http" not in cleaned
    assert "扫码" not in cleaned
    assert "PN结具有单向导电性" in cleaned


def test_chunks_keep_metadata():
    docs = [
        PageDocument(
            text="PN结正向偏置时势垒降低。" * 50,
            source="教材.pdf",
            page=29,
            chapter="第一章 常用半导体器件",
            section="1.1.3 PN结",
        )
    ]
    chunks = chunk_documents(docs, max_chars=260)
    assert len(chunks) >= 2
    assert all(chunk.page_start == 29 for chunk in chunks)
    assert all(chunk.chapter.startswith("第一章") for chunk in chunks)
    assert any("PN结" in chunk.knowledge_tags for chunk in chunks)


def test_long_ocr_question_is_not_promoted_to_knowledge_tag():
    docs = [
        PageDocument(
            text="二极管正向偏置时可以导通。",
            source="扫描习题.pdf",
            page=24,
            chapter="第一章",
            section="1.3.2 二极管电路如图所示,请判断二极管是导通还是截止",
        )
    ]
    chunks = chunk_documents(docs)
    assert "二极管" in chunks[0].knowledge_tags
    assert not any("如图" in tag for tag in chunks[0].knowledge_tags)


def test_exam_pdf_is_classified_and_questions_keep_source_anchor(tmp_path):
    path = tmp_path / "电路分析期末试卷.pdf"
    documents = [
        PageDocument(
            text="1. 已知 $R=10\\Omega$，求支路电流。\n2. 写出节点电压方程并求解。",
            source=path.name,
            page=2,
            chapter="试卷",
            section="计算题",
            doc_type="exam",
        )
    ]
    assert classify_document(path, documents) == "exam"
    questions = extract_questions_from_documents(documents, path.name)
    assert len(questions) == 2
    assert all(item["source_page"] == 2 for item in questions)
    assert all(item["question_id"].startswith("AUTO-") for item in questions)


def test_declared_document_type_overrides_heuristics(tmp_path):
    path = tmp_path / "资料.pdf"
    assert classify_document(path, [], "notes") == "notes"


def test_internal_ingestion_manifest_is_not_a_course_source(tmp_path):
    (tmp_path / ".ingestion_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".hidden.json").write_text("{}", encoding="utf-8")
    (tmp_path / "教材.txt").write_text("欧姆定律", encoding="utf-8")
    assert [path.name for path in list_source_files(tmp_path)] == ["教材.txt"]


def test_scanned_section_number_advances_stale_chapter_heading():
    chapter, section = _infer_ocr_hierarchy(
        "2.1 双极型晶体管\n双极型晶体管是常用半导体器件。",
        "电子电路基础",
        "第一章 半导体基础知识及二极管电路",
        "1.3 二极管电路",
    )
    assert chapter == "第二章"
    assert section == "2.1 双极型晶体管"


def test_scanned_chapter_reference_sentence_is_not_a_heading():
    chapter, _ = _infer_ocr_hierarchy(
        "第二章中我们已经知道,晶体管可以看成双端口网络。",
        "电子电路基础",
        "第三章 放大电路",
        "3.1 小信号模型",
    )
    assert chapter == "第三章 放大电路"


def test_pdf_extraction_cache_round_trip(tmp_path):
    cache_path = tmp_path / "extract.json"
    document = PageDocument(
        text="PN结具有单向导电性。",
        source="扫描教材.pdf",
        page=12,
        chapter="第一章",
        section="1.2 PN结",
    )
    _write_cached_pdf(cache_path, [document], "macos-vision-ocr", ["扫描版"])
    cached = _load_cached_pdf(cache_path)
    assert cached is not None
    documents, parser, warnings = cached
    assert documents == [document]
    assert parser == "macos-vision-ocr"
    assert warnings == ["扫描版"]
