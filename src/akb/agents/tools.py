"""Agent tools.

Two retrieval surfaces:
  * ``search_knowledge_base`` — local hybrid retrieval pipeline (Phases 2-3).
  * ``search_web``            — DuckDuckGo (default) or Tavily (if configured).

Tools return plain ``str`` blocks separated by ``\\n---\\n`` so they can be
concatenated into an LLM context window without further parsing.
"""

from __future__ import annotations

from akb.config import load_settings
from akb.ingest.graph import VaultGraph
from akb.retrieve.pipeline import retrieve
from akb.schemas import Citation


def search_knowledge_base(
    query: str,
    *,
    top_k: int | None = None,
    n_results: int | None = None,
    graph: VaultGraph | None = None,
) -> tuple[str, list[Citation]]:
    """Run the full hybrid+rerank pipeline. Returns (joined_context, citations)."""
    res = retrieve(query, n_results=n_results, top_k=top_k, graph=graph)
    if not res.chunks:
        return "", []
    blocks: list[str] = []
    citations: list[Citation] = []
    for rc in res.chunks:
        title = (rc.chunk.metadata.get("title") or rc.chunk.source_id) if rc.chunk.metadata else rc.chunk.source_id
        header = " > ".join(rc.chunk.header_path) if rc.chunk.header_path else ""
        head = f"[{title}{(' :: ' + header) if header else ''}]"
        blocks.append(f"{head}\n{rc.chunk.text}")
        citations.append(
            Citation(
                source_id=rc.chunk.source_id,
                chunk_id=rc.chunk.chunk_id,
                snippet=rc.chunk.text[:240],
                score=rc.final_score,
            )
        )
    return "\n---\n".join(blocks), citations


def search_web(query: str) -> str:
    cfg = load_settings().agent
    if cfg.web_tool == "tavily":
        try:
            from tavily import TavilyClient  # type: ignore[import-not-found]
            import os

            key = os.getenv("TAVILY_API_KEY")
            if not key:
                return ""
            client = TavilyClient(api_key=key)
            res = client.search(query=query, max_results=5)
            results = res.get("results", [])
            return "\n---\n".join(r.get("content", "") for r in results)
        except Exception as e:
            return f"[tavily error: {e}]"
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = [r.get("body", "") for r in ddgs.text(query, max_results=5)]
        return "\n---\n".join(filter(None, results))
    except Exception as e:
        return f"[ddg error: {e}]"
