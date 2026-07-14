from __future__ import annotations

import argparse
import json
from uuid import uuid4

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("auto", "answer", "quiz", "plan"), default="auto"
    )
    parser.add_argument("--message", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    default_messages = {
        "auto": "PN结为什么具有单向导电性？请简要解释。",
        "answer": "已知10欧姆电阻两端电压为20伏，求流过电阻的电流。",
        "quiz": "请出一道硅二极管恒压降模型的基础计算题。",
        "plan": "我对一阶RC电路不熟，请安排一个三步复习计划。",
    }
    payload = {
        "session_id": args.session_id or f"smoke-{args.mode}-{uuid4().hex[:8]}",
        "message": args.message.strip() or default_messages[args.mode],
        "mode": args.mode,
        "knowledge_base": "default",
    }
    events: dict[str, int] = {}
    answer_parts: list[str] = []
    meta: dict[str, object] = {}
    timeout = httpx.Timeout(connect=10.0, read=650.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        with client.stream(
            "POST",
            f"{args.base_url.rstrip('/')}/api/chat",
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
    print(f"intent={meta.get('intent')}")
    print(f"agent={meta.get('agent')}")
    sources = meta.get("sources") or []
    print(f"sources={len(sources)}")
    print(
        "source_names="
        + json.dumps(
            list(dict.fromkeys(str(item.get("source", "")) for item in sources)),
            ensure_ascii=False,
        )
    )
    print(f"verification={meta.get('verification')}")
    print(f"preview={answer[:240]}")


if __name__ == "__main__":
    main()
