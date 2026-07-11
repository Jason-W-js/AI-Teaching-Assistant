from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.ollama_client import OllamaClient


async def main() -> None:
    client = OllamaClient()
    try:
        health = await client.health()
        print(f"health={health}")
        answer = await client.chat(
            [
                {
                    "role": "user",
                    "content": '你是意图分类器。只输出 JSON：{"intent":"answer"}',
                }
            ],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=96,
        )
        print(f"final={answer}")
        print("thinking_enabled=True; private thinking was not printed")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
