"""Query-side transforms.

Two cheap, well-documented techniques that consistently lift recall:
  * **Multi-query decomposition** — break the user's question into 1-3 simpler
    sub-queries and union the retrieved chunks. (Already proven on the legacy
    `run_reflective_agent_loop`; here it's formalized + tested.)
  * **HyDE** — ask the LLM to hallucinate a hypothetical *answer*, embed that,
    and use it for dense search. Surprisingly effective when the query is short
    and underspecified.

Both call out to Ollama using JSON mode; both fail open (return the original
query) so retrieval never *worsens* because the LLM was being weird.
"""

from __future__ import annotations

import json

from akb.config import LLMConfig, load_settings

_DECOMP_PROMPT = """You are a query analysis agent. Decompose the following user query into 1 to 3 simple, self-contained sub-queries that, together, cover the original intent.

If the query is already simple and atomic, return it as a single-item list.

Respond with ONLY a JSON object with one key "queries" whose value is a list of strings.

User Query: "{query}"
"""

_HYDE_PROMPT = """You are a helpful assistant. Write a short, factual, 2-4 sentence passage that would be a perfect answer to the user's question. Do NOT preface it with anything. Just the passage.

Question: {query}
Answer:"""


def _ollama_generate(prompt: str, model: str, llm_cfg: LLMConfig, fmt_json: bool = False) -> str:
    import ollama

    kwargs: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": llm_cfg.temperature},
    }
    if fmt_json:
        kwargs["format"] = "json"
    resp = ollama.generate(**kwargs)
    return str(resp.get("response", ""))


def decompose(query: str, llm_cfg: LLMConfig | None = None) -> list[str]:
    cfg = llm_cfg or load_settings().llm
    try:
        text = _ollama_generate(_DECOMP_PROMPT.format(query=query), cfg.local_model, cfg, fmt_json=True)
        data = json.loads(text)
        queries = data.get("queries", [])
        out = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return out or [query]
    except Exception:
        return [query]


def hyde(query: str, llm_cfg: LLMConfig | None = None) -> str:
    """Return a hypothetical answer to use as the dense-side query embedding source."""
    cfg = llm_cfg or load_settings().llm
    try:
        text = _ollama_generate(_HYDE_PROMPT.format(query=query), cfg.local_model, cfg)
        cleaned = text.strip()
        return cleaned or query
    except Exception:
        return query


def expand(query: str, *, use_hyde: bool = False, use_decomp: bool = True) -> list[str]:
    """Single entry point used by the hybrid retriever."""
    out: list[str] = []
    if use_decomp:
        out.extend(decompose(query))
    else:
        out.append(query)
    if use_hyde:
        out.append(hyde(query))
    # dedupe while preserving order
    seen: set[str] = set()
    return [q for q in out if not (q in seen or seen.add(q))]
