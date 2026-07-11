import asyncio

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
