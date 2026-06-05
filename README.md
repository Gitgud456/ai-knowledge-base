# akb — Personal AI Knowledge Base

A local-first, Obsidian-aware RAG system with hybrid retrieval, Anthropic-style contextual chunking, a LangGraph agent, and incremental sync. Built for one user, your laptop, and your vault.

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](#license)
[![type-checked](https://img.shields.io/badge/typed-pydantic%20v2-purple)](https://docs.pydantic.dev/)

## What it does

- **Reads your Obsidian vault, PDFs, and EPUBs** — preserves headings, wikilinks, frontmatter, tags, and embeds (`![[Note]]` is recursively resolved with cycle protection).
- **Hybrid retrieval** — BGE-M3 dense + lexical-sparse vectors fused server-side by Qdrant via RRF, then cross-encoder reranked with `bge-reranker-v2-m3`.
- **Contextual Retrieval** (Anthropic, Sept 2024) — every chunk gets a 1-2 sentence situating prefix written by your local Ollama model, cached by content hash. ~35–50% recall lift documented in the original paper.
- **LangGraph agent** — router → retrieve → draft → CRAG-style critic → finalize. The critic can rewrite the query and re-retrieve.
- **Mentor mode** — generates a learning plan from your vault, teaches topic-by-topic, advances on LLM-classified intent.
- **Incremental sync** — SHA-256 keyed `ingest_state.db` + watchfiles observer. Editing one note re-embeds one note, not 50,000.
- **Wikilink graph expansion** — at query time, retrieved chunks pull in 1-hop wikilink neighbours before reranking. Uses the *real* graph you wrote, not embedding similarity.
- **CLI + Streamlit UI** — `akb chat`, `akb sync --watch`, `akb eval`, or `akb serve` for the 4-tab UI.
- **No cloud required** — every default model runs locally via Ollama + sentence-transformers. Gemini "deep mode" is optional and gated by config.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | **Qdrant embedded** (named dense + sparse vectors) | Real hybrid + payload filters in one engine, no daemon |
| Embeddings | **BGE-M3** (1024-d dense + lexical sparse) | One model, both vectors, multilingual, 8k context |
| Reranker | **bge-reranker-v2-m3** | Strict upgrade over `ms-marco-MiniLM` |
| LLM | **Ollama** (`llama3:8b-instruct-q4_K_M` by default) | Local; swap any Ollama tag in config |
| Chunking | **MarkdownHeaderTextSplitter → recursive fallback** | Header breadcrumb survives as filterable metadata |
| Agent | **LangGraph** | Explicit state machine for the router + CRAG loop |
| Sync | **watchfiles** (Rust-backed) + SHA-256 sidecar SQLite | Designed for large vaults |
| UI | **Streamlit** + **streamlit-agraph** for the wikilink graph | Same 4-tab shape as the original prototype |
| Eval | **RAGAS** + cheap citation/substring heuristics | LLM-as-judge metrics with a CI-friendly fallback |
| Logging | **structlog** (JSON or console) | One processor pipeline, optional Langfuse/OTel export |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  UI: Streamlit (4 tabs) + CLI (typer)                                │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Agent: LangGraph state machine                                      │
│    router → (retrieve_kb | retrieve_web | direct)                    │
│    draft → critic → [revise+retry | finalize]                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Retrieve: multi-query/HyDE → hybrid (RRF) → wikilink 1-hop → rerank │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────┬───────▼────────┬─────────────────────────────┐
│ Qdrant (embedded)     │ BGE-M3 embed   │ ingest_state.db (SQLite)    │
│ dense + sparse vecs   │ dense + sparse │ path → sha256 → chunk_ids   │
│ wikilinks in payload  │ contextualized │ watchfiles live observer    │
└───────────────────────┴───────┬────────┴─────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Ingest: Obsidian-aware loader                                       │
│    frontmatter → metadata · tags → metadata · wikilinks → graph      │
│    embeds → recursive inline · header path → breadcrumb              │
│    contextualizer (local Ollama, cached by content hash)             │
└──────────────────────────────────────────────────────────────────────┘
```

## Installation

Requires Python 3.11+ and a running [Ollama](https://ollama.com).

```bash
# 1. Pull a local model
ollama pull llama3:8b-instruct-q4_K_M

# 2. Get the code
git clone https://github.com/<you>/akb.git
cd akb

# 3. Install (uv is recommended; pip works too)
uv venv && uv pip install -e ".[dev]"
# or:
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 4. Point at your vault
cp .env.example .env
# edit OBSIDIAN_VAULT_PATH to your vault root
```

Optional extras:

```bash
uv pip install -e ".[tracing]"  # Langfuse + OpenTelemetry
uv pip install -e ".[late]"     # ColBERT rerank + late chunking (RAGatouille)
```

> **First run downloads** BGE-M3 (~2.3 GB) and `bge-reranker-v2-m3` (~600 MB) from Hugging Face. They're cached after that.

## Quickstart

```bash
akb info                 # show resolved config and paths
akb reindex --yes        # full one-time index of your vault
akb sync                 # incremental: only changed/added/deleted notes
akb sync --watch         # live: re-embed on file save
akb chat                 # REPL against the LangGraph agent
akb serve                # Streamlit UI on localhost:8501
akb eval                 # RAGAS over tests/golden/golden_set.yaml
```

### Querying with filters

The retrieval pipeline accepts simple metadata filters:

```python
from akb.retrieve.pipeline import retrieve

res = retrieve(
    "what did I write about ARP spoofing?",
    filter={"tags": ["security", "network"]},
    top_k=5,
)
for c in res.chunks:
    print(c.chunk.source_id, "→", c.chunk.text[:120])
```

## Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml). Override per-machine via `configs/local.yaml` (gitignored) or environment variables with the `AKB_` prefix:

```bash
AKB_RETRIEVE__TOP_K=12 akb chat
AKB_INGEST__CONTEXTUAL_RETRIEVAL=false akb reindex --yes
AKB_LLM__LOCAL_MODEL=mistral akb chat
```

Anything nested uses `__` as the path separator (`AKB_<section>__<key>`).

## Repository layout

```
src/akb/
  config.py           # pydantic-settings: default.yaml + local.yaml + env
  schemas.py          # Document, Chunk, RetrievedChunk, Citation, Answer
  cli.py              # `akb` typer commands
  ingest/
    obsidian_loader.py   pdf_loader.py   epub_loader.py   txt_loader.py
    chunkers.py          # MarkdownHeader → recursive
    contextualizer.py    # Anthropic-style chunk-context (Ollama, SQLite cache)
    dedupe.py            # SHA + shingle Jaccard
    graph.py             # VaultGraph (wikilinks)
    pipeline.py          # path → chunks
    upsert.py            # batched embed + qdrant.upsert
    sync.py              # plan + apply diff
    watcher.py           # watchfiles loop
    late_chunking.py     # opt-in (Phase 9)
  embed/
    providers.py         # BGE-M3
  store/
    qdrant_store.py      # embedded Qdrant, named vectors, payload indices
    sqlite_state.py      # ingest_state.db
  retrieve/
    query_transform.py   # multi-query + HyDE
    hybrid.py            # cross-query RRF over qdrant hybrid prefetch
    graph_expand.py      # 1-hop wikilink expansion
    rerank.py            # bge-reranker-v2-m3
    colbert_rerank.py    # opt-in (Phase 9)
    pipeline.py          # composes everything
  agents/
    tools.py             # search_kb, search_web
    memory.py            # short-term window + summary
    graph.py             # LangGraph router + CRAG
    mentor.py            # plan + lessons + Q/A
  sessions/db.py         # SQLite session/messages
  eval/
    golden.py            # YAML loader
    ragas_runner.py      # RAGAS + cheap heuristics
  prompts/
    chat.py  mentor.py   # versioned, A/B-friendly
  obs/
    logging.py           # structlog
    tracing.py           # optional Langfuse via OTel
  ui/app.py              # new Streamlit UI on top of the package
tests/
  unit/                  # 30 tests, no heavy deps required
  golden/golden_set.yaml # evaluation set (seed yours here)
configs/
  default.yaml
  local.yaml.example
legacy/                  # original prototype, kept for reference
```

## Testing

```bash
pytest                                            # full suite
pytest tests/unit/test_obsidian_loader.py -v      # one file
akb eval --no-ragas --json data/eval.json         # offline regression check
```

The unit suite runs in ~3 s and doesn't need Qdrant, Ollama, or any ML model loaded.

## How retrieval actually works

1. **Query transforms.** The query is decomposed into 1-3 atomic sub-queries via Ollama (JSON mode). Optionally augmented with a HyDE-style hypothetical answer.
2. **Hybrid prefetch.** For each sub-query, BGE-M3 produces a dense vector and a lexical-sparse vector. Both go to Qdrant as parallel `Prefetch` paths; Qdrant fuses them server-side with RRF.
3. **Cross-query RRF.** Per-sub-query hit lists are RRF-merged client-side.
4. **Graph expansion.** Top hits' notes pull in 1-hop wikilink neighbours' chunks.
5. **Rerank.** `bge-reranker-v2-m3` re-scores the union; the top `top_k` survive.
6. **Generate.** The agent's draft node sees those chunks plus the conversation window.
7. **Critique.** A CRAG-style critic decides whether the draft is grounded enough; if not, it rewrites the retrieval query and loops.
8. **Finalize.** The final node produces the user-facing answer with inline citations to the header-tagged source blocks.

## Why not just use [popular tool X]

| Tool | Why I didn't | What akb does differently |
|---|---|---|
| Smart Connections | JSON-on-disk vector store, dense-only, no rerank | Hybrid + reranker + contextual retrieval |
| Obsidian Copilot | Solid, but Plus tier and Orama-in-browser | Native filesystem, your data never leaves the box |
| Khoj | Excellent but heavy (Postgres + Django) | Single-binary feel, embedded everything |
| Quivr / AnythingLLM | General-purpose; wikilinks are dead text | First-class wikilink graph and embed expansion |
| GraphRAG | Quality leader, but $$$ at index time | Free local equivalent for personal-scale vaults |

## Roadmap

- [ ] Long-term Q/A memory collection (cross-session recall)
- [ ] Tokenizer-offset late chunking (current uses uniform proportional spans)
- [ ] Integration tests against a real Qdrant + Ollama in CI
- [ ] Markdown-table-aware chunker (treat tables as atomic chunks)
- [ ] Optional Tantivy full-text path for very long-tail token recall

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic, Sept 2024
- [BGE-M3](https://arxiv.org/abs/2402.03216) — Chen et al., BAAI
- [CRAG](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
- [Late Chunking](https://arxiv.org/abs/2409.04701) — Günther et al., 2024
- [LangGraph](https://github.com/langchain-ai/langgraph), [Qdrant](https://qdrant.tech/), [Ollama](https://ollama.com/)
