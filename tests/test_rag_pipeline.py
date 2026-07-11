from backend.app.rag.models import PageDocument
from backend.app.rag.pipeline import chunk_documents, clean_page_text


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

