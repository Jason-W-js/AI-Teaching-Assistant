import asyncio
import base64

import fitz

from backend.app.services.attachments import AttachmentStore


def test_text_attachment_persists_and_resolves(tmp_path):
    store = AttachmentStore()
    store.root = tmp_path
    meta = store._save_sync(
        session_id="student-test",
        filename="question.md",
        content_type="text/markdown",
        data="# 题目\n求电阻两端电压。".encode("utf-8"),
    )
    resolved = store._resolve_sync("student-test", [meta["id"]])
    assert "求电阻两端电压" in resolved.text
    assert resolved.images == []
    assert resolved.items[0]["name"] == "question.md"
    assert resolved.items[0]["url"].startswith("/api/attachments/")


def test_pdf_attachment_pages_are_rendered_for_vision_recognition(tmp_path):
    document = fitz.open()
    page = document.new_page(width=320, height=240)
    page.insert_text((30, 50), "R1 = 10 ohm; calculate current")
    pdf_bytes = document.tobytes()
    document.close()

    store = AttachmentStore()
    store.root = tmp_path
    meta = store._save_sync(
        session_id="student-pdf",
        filename="circuit.pdf",
        content_type="application/pdf",
        data=pdf_bytes,
    )
    resolved = store._resolve_sync("student-pdf", [meta["id"]])

    assert "calculate current" in resolved.text
    assert len(resolved.images) == 1
    assert base64.b64decode(resolved.images[0]).startswith(b"\x89PNG\r\n\x1a\n")


def test_history_restores_new_and_legacy_attachment_metadata(tmp_path):
    store = AttachmentStore()
    store.root = tmp_path
    first = store._save_sync(
        session_id="student-history",
        filename="circuit.png",
        content_type="image/png",
        data=b"not-read-during-history-restore",
    )
    second = store._save_sync(
        session_id="student-history",
        filename="legacy.png",
        content_type="image/png",
        data=b"legacy-image-placeholder",
    )

    restored = store.enrich_history("student-history", [
        {"role": "user", "content": "分析新电路", "attachments": [{"id": first["id"]}]},
        {"role": "assistant", "content": "新电路回答"},
        {"role": "user", "content": "分析旧电路\n[附件：legacy.png]"},
    ])

    assert restored[0]["attachments"] == [first]
    assert restored[2]["attachments"] == [second]
    assert restored[2]["content"] == "分析旧电路"


def test_delete_session_removes_attachment_directory(tmp_path):
    store = AttachmentStore()
    store.root = tmp_path
    store._save_sync(
        session_id="student-delete",
        filename="question.md",
        content_type="text/markdown",
        data="待删除附件".encode("utf-8"),
    )
    session_dir = tmp_path / "student-delete"
    assert asyncio.run(store.delete_session("student-delete")) is True
    assert session_dir.exists() is False
    assert asyncio.run(store.delete_session("student-delete")) is False
