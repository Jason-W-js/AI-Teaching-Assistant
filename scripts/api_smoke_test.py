from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("answer", "quiz"), default="answer")
    args = parser.parse_args()
    payload = {
        "session_id": f"smoke-{args.mode}",
        "message": (
            "请出一道硅二极管恒压降模型的基础计算题"
            if args.mode == "quiz"
            else "PN结为什么具有单向导电性？请用公式分步解释。"
        ),
        "mode": args.mode,
        "knowledge_base": "default",
    }
    events: dict[str, int] = {}
    answer_parts: list[str] = []
    meta: dict[str, object] = {}
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "http://127.0.0.1:8000/api/chat",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            current_event = "message"
            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data = json.loads(line.removeprefix("data:").strip())
                    events[current_event] = events.get(current_event, 0) + 1
                    if current_event == "status":
                        print(f"status={data.get('agent')}::{data.get('message')}")
                    elif current_event == "delta":
                        answer_parts.append(data.get("content", ""))
                    elif current_event == "meta":
                        meta = data
                    elif current_event == "error":
                        raise RuntimeError(data.get("message", "SSE error"))
    answer = "".join(answer_parts)
    print(f"events={events}")
    print(f"answer_chars={len(answer)}")
    print(f"has_latex={'$' in answer}")
    print(f"has_sources={'检索依据' in answer}")
    print(f"verification={meta.get('verification')}")
    print(f"preview={answer[:240]}")


if __name__ == "__main__":
    main()
