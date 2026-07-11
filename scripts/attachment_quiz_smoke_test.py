from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

import httpx


def stream_quiz(client: httpx.Client, payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    parts: list[str] = []
    meta: dict[str, object] = {}
    current_event = "message"
    with client.stream(
        "POST",
        "http://127.0.0.1:8000/api/chat",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
                if current_event == "delta":
                    parts.append(data.get("content", ""))
                elif current_event == "meta":
                    meta = data
                elif current_event == "error":
                    raise RuntimeError(data.get("message", "SSE error"))
    return "".join(parts), meta


def question_from(answer: str) -> str:
    match = re.search(
        r"##\s*同类型新题[^\n]*\n+(.+?)(?=\n+###\s*解题思路|\Z)", answer, re.S
    )
    return match.group(1).strip() if match else ""


def preserves_parallel_rl_capacitor_blueprint(question: str) -> bool:
    required = (
        "并联",
        "支路",
        "功率因数",
        "总电流",
        "感抗",
        "容抗",
        "无功功率",
    )
    return all(marker in question for marker in required) and any(
        marker in question for marker in ("电容支路电流", "电容电流", "支路电流")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    session_id = f"attachment-smoke-{uuid.uuid4()}"

    with httpx.Client(timeout=420.0, trust_env=False) as client:
        with args.image.open("rb") as handle:
            upload = client.post(
                "http://127.0.0.1:8000/api/attachments",
                data={"session_id": session_id},
                files={"file": (args.image.name, handle, "image/png")},
            )
        upload.raise_for_status()
        attachment_id = upload.json()["attachment"]["id"]
        payload = {
            "session_id": session_id,
            "message": "请根据图片中的原题出一道同类型新题，不能重复上一题。",
            "mode": "quiz",
            "knowledge_base": "default",
            "attachment_ids": [attachment_id],
        }
        first_answer, first_meta = stream_quiz(client, payload)
        second_answer, second_meta = stream_quiz(client, payload)

    first_question = question_from(first_answer)
    second_question = question_from(second_answer)
    print(json.dumps({
        "first_question": first_question,
        "second_question": second_question,
        "first_verification": first_meta.get("verification"),
        "second_verification": second_meta.get("verification"),
        "first_sources": first_meta.get("sources"),
        "second_sources": second_meta.get("sources"),
    }, ensure_ascii=False, indent=2))
    assert first_question and second_question
    assert first_question != second_question
    assert "串联电路中 $R_1" not in first_question + second_question
    assert preserves_parallel_rl_capacitor_blueprint(first_question)
    assert preserves_parallel_rl_capacitor_blueprint(second_question)
    assert first_meta.get("verification", {}).get("passed") is True
    assert second_meta.get("verification", {}).get("passed") is True


if __name__ == "__main__":
    main()
