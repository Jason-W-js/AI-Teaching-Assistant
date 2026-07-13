import asyncio

from backend.app.services.memory import ConversationMemory


def test_local_memory_survives_process_object_restart(tmp_path):
    async def scenario():
        first = ConversationMemory(storage_dir=tmp_path)
        first.backend = "local-persistent"
        await first.append("student-session", "assistant", "上一次生成的题目")

        restarted = ConversationMemory(storage_dir=tmp_path)
        restarted.backend = "local-persistent"
        return await restarted.recent("student-session")

    history = asyncio.run(scenario())
    assert history[-1]["content"] == "上一次生成的题目"


def test_local_memory_lists_and_restores_conversations(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        await memory.append("student-first", "user", "请解释戴维南定理")
        await memory.append(
            "student-first",
            "assistant",
            "戴维南定理说明……",
            {
                "agent": "答疑 Agent",
                "provider": "qwen",
                "model": "qwen-plus",
                "sources": [{"id": "chunk-1", "source": "lesson.pdf"}],
            },
        )
        sessions = await memory.list_sessions()
        messages = await memory.history("student-first")
        return sessions, messages

    sessions, messages = asyncio.run(scenario())
    assert sessions[0]["session_id"] == "student-first"
    assert sessions[0]["title"] == "请解释戴维南定理"
    assert messages[-1]["model"] == "qwen-plus"
    assert messages[-1]["sources"][0]["source"] == "lesson.pdf"


def test_local_memory_deletes_history_file_index_and_cache(tmp_path):
    async def scenario():
        memory = ConversationMemory(storage_dir=tmp_path)
        memory.backend = "local-persistent"
        await memory.append("student-delete", "user", "准备删除的会话")
        history_path = memory._fallback_path("student-delete")
        deleted = await memory.delete("student-delete")
        return deleted, history_path.exists(), await memory.list_sessions(), await memory.history("student-delete")

    deleted, file_exists, sessions, history = asyncio.run(scenario())
    assert deleted is True
    assert file_exists is False
    assert sessions == []
    assert history == []
