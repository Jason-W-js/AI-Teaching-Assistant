import asyncio

from backend.app.services.mistake_book import MistakeBook, related_mistake_attachments


def test_mistake_book_is_durable_deduplicated_and_student_scoped(tmp_path):
    path = tmp_path / "mistakes.json"
    book = MistakeBook(path)
    first = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session-a",
        content="求二极管恒压降模型下的回路电流。",
        agent="答疑 Agent",
        knowledge_points=["二极管", "恒压降模型", "二极管"],
        summary="二极管恒压降计算",
    ))
    duplicate = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session-b",
        content="求二极管恒压降模型下的回路电流。",
        agent="出题 Agent",
        knowledge_points=["二极管"],
        summary="重复题",
    ))
    asyncio.run(book.add(
        student_id="learner-b",
        session_id="student-session-c",
        content="分析晶体管静态工作点。",
        agent="答疑 Agent",
        knowledge_points=["晶体管", "静态工作点"],
        summary="静态工作点",
    ))

    assert first["id"] == duplicate["id"]
    assert first["knowledge_points"] == ["二极管", "恒压降模型"]
    assert len(asyncio.run(MistakeBook(path).list("learner-a"))) == 1
    assert len(asyncio.run(MistakeBook(path).list("learner-b"))) == 1
    assert asyncio.run(book.delete("learner-a", first["id"])) is True
    assert asyncio.run(book.list("learner-a")) == []


def test_mistake_book_persists_attachments_and_upgrades_a_duplicate(tmp_path):
    path = tmp_path / "mistakes.json"
    book = MistakeBook(path)
    attachment = {
        "id": "a" * 32,
        "name": "circuit.png",
        "content_type": "image/png",
        "size": 1024,
        "kind": "image",
        "url": "/api/attachments/" + "a" * 32 + "?session_id=student-session",
    }
    first = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session",
        content="分析图中电路。",
        agent="学生原题",
        knowledge_points=["共射放大电路"],
        summary="电路分析",
    ))
    upgraded = asyncio.run(book.add(
        student_id="learner-a",
        session_id="student-session",
        content="分析图中电路。",
        agent="学生原题",
        knowledge_points=["共射放大电路"],
        summary="电路分析",
        attachments=[attachment],
    ))

    assert first["id"] == upgraded["id"]
    assert upgraded["attachments"] == [attachment]
    assert asyncio.run(MistakeBook(path).list("learner-a"))[0]["attachments"] == [attachment]


def test_legacy_mistake_recovers_direct_and_answer_attachments_from_history():
    attachment = {"id": "b" * 32, "name": "circuit.png", "kind": "image"}
    history = [
        {
            "role": "user",
            "content": "上述电路有什么问题",
            "attachments": [attachment],
        },
        {
            "role": "assistant",
            "content": "该电路缺少直流偏置。",
        },
    ]

    assert related_mistake_attachments(
        history, "上述电路有什么问题", "学生原题"
    ) == [attachment]
    assert related_mistake_attachments(
        history, "该电路缺少直流偏置。", "答疑 Agent"
    ) == [attachment]
