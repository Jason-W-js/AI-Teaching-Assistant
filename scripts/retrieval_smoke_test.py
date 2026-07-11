from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.retriever import HybridRetriever


def main() -> None:
    retriever = HybridRetriever(
        settings.vector_stores_dir / "default", settings.embedding_model_path
    )
    for query in ("PN结为什么具有单向导电性", "请出一道二极管伏安特性同类题"):
        print(f"\nQUERY: {query}")
        for hit in retriever.search(query, k=5, prefer_questions="出" in query):
            print(
                f"{hit.score:.3f} | {hit.chunk.doc_type} | {hit.chunk.section} | "
                f"page={hit.chunk.page_start} | {hit.chunk.text[:90]}"
            )


if __name__ == "__main__":
    main()
