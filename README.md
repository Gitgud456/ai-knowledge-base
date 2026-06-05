# akb — Personal AI Knowledge Base

A local-first, Obsidian-aware RAG system. Hybrid retrieval, Anthropic-style contextual chunking, a LangGraph agent with CRAG-style reflection, cross-session memory, time-aware retrieval, RAPTOR hierarchical summaries, wikilink community summaries, cross-modal image search, speculative decoding, and incremental sync. Built for one user, your laptop, and your vault.

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](#license)
[![type-checked](https://img.shields.io/badge/typed-pydantic%20v2-purple)](https://docs.pydantic.dev/)
[![tests](https://img.shields.io/badge/tests-152%20passing-success)](#testing)

## What it does

### Core
- **Reads your Obsidian vault, PDFs, EPUBs, and the web.** Preserves headings, frontmatter, tags, aliases, wikilinks, and embeds (`![[Note]]` is recursively resolved with cycle protection). Unicode-safe casefold so `Straße` matches `strasse`.
- **Hybrid retrieval.** BGE-M3 dense + lexical-sparse vectors fused server-side by Qdrant via RRF, plus client-side cross-query RRF over multi-query / HyDE transforms.
- **Cross-encoder reranking** with `bge-reranker-v2-m3` (or optional ColBERT via RAGatouille).
- **Contextual Retrieval** (Anthropic, Sept 2024) — every chunk gets a 1-2 sentence situating prefix written by your local Ollama model, **batched** (up to 8 chunks per LLM call) and cached by content hash. ~35-50% recall lift documented in the original paper.
- **LangGraph agent** — router → retrieve → draft → CRAG-style critic → finalize. The critic can rewrite the query and re-retrieve.
- **Wikilink graph expansion.** At query time, top hits pull in 1-hop wikilink neighbours; the graph chunks get a baseline score so they survive the reranker pool cut.
- **Incremental sync** — SHA-256-keyed `ingest_state.db` + `watchfiles` observer. Editing one note re-embeds one note, not 50,000. The watcher restricts plan-sync to the touched paths.
- **Mentor mode** — generates a learning plan from your vault, teaches topic-by-topic, advances on LLM-classified intent. Conversational continuity preserved across turns.

### Chat UX
- **Token streaming** with a "Why this answer" reasoning panel (router decision, sub-queries, iterations, critic verdict + notes, improved-query rewrites).
- **Slash commands.** `/search` forces vault path, `/web` forces web, `/cite` returns chunks without synthesis, `/dry-run` shows what would have been sent.
- **Sticky `[[Note]]` mentions.** Pre-resolves to that note's chunks and pins them into the context as `[pinned: Note]` blocks.
- **Related notes panel** — 1-hop wikilink neighbours of the cited sources.
- **Save citations as notes** — writes `{vault}/akb_snippets/{slug}.md` with the snippet and a backlink.
- **Conversation export** to vault — markdown with Obsidian-friendly wikilinks (button + CLI).

### Retrieval intelligence
- **Cross-session memory.** Every finalized Q/A pair is embedded into a dedicated `chat_memory` Qdrant collection and surfaces as `[memory]` blocks on later turns ("what was I working on last week?").
- **Time-aware retrieval.** `dateparser` extracts temporal hints ("last March", "this week", "before 2024") and applies a `modified_at` range filter at retrieve time.
- **Recency-weighted rerank.** `rerank_score *= exp(-age_days / half_life)` so a 6-month-old note has to be meaningfully more relevant to outrank a recent one. Off by default; configurable half-life.
- **Secret scrubbing at ingest.** Regex patterns for AWS / OpenAI / Anthropic / GitHub / Google / Slack tokens + PEM private key blocks + JWTs. Policies: `off` / `warn` / `redact` (default) / `block`.

### Advanced features (opt-in)
- **RAPTOR** hierarchical summary index. UMAP → GMM clusters every chunk, an LLM summarises each cluster, summaries are themselves clustered for the next level. Answers "what do my notes *collectively* say about X" in one hop.
- **Wikilink community summaries** (light Graph-RAG). Louvain on the wikilink graph; an LLM summarises each large community. Cheaper than Microsoft GraphRAG because *you wrote the edges already* — no entity-extraction LLM cost.
- **SigLIP image search.** Discovers `![[image]]` / `![alt](path)` embeds across the vault and embeds them into a dedicated `vault_images` collection. Cross-modal text → image: search `"the diagram with dotted arrows"` against your screenshots.
- **Speculative RAG** (Wang et al., 2025). Round-robin partitions context into N subsets, drafts in parallel, verifier scores and picks the winner. ~2× latency cut + quality bump on multi-tool queries.

### Ops
- **`akb doctor`** — preflight: Ollama reachable, BGE-M3 + reranker weights cached, vault exists, data + qdrant dirs writable, ≥1 GB free.
- **`akb stats`** — Qdrant points, sources by type, ingest_state vs Qdrant drift, context-cache rows, session count.
- **`akb backup` / `akb restore`** — atomic `.tar.gz` of the entire `data/` directory; restore swaps the live dir aside instead of overwriting.
- **`akb schedule`** — persisted cron expressions + one-shot runner. Wire to OS cron / Task Scheduler.
- **Schema migrations** on every akb-owned SQLite DB; **index version stamp** in Qdrant. A mismatch between the live index and the active config raises `IndexCompatibilityError` pointing at `akb reindex` instead of silently embedding with one model and querying with another.
- **structlog JSON logs** wired throughout the hot paths (sync, retrieve, agent nodes, contextualizer, tools). Optional Langfuse / OpenTelemetry export.

### No cloud required
Every default runs locally via Ollama + sentence-transformers. Gemini "deep mode" is optional and gated by config. The leaked-key scrubber is on by default so secrets in pasted curl calls never reach the LLM context.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | **Qdrant embedded** (named dense + sparse vectors, optional binary quantization) | Real hybrid + payload filters in one engine, no daemon. BQ gives ~32× smaller / ~40× faster with oversampling + rescore. |
| Embeddings | **BGE-M3** (1024-d dense + lexical sparse) | One model, both vectors, multilingual, 8k context |
| Reranker | **bge-reranker-v2-m3** (optional ColBERT v2 via RAGatouille) | Strict upgrade over `ms-marco-MiniLM` |
| Image embed | **SigLIP** `google/siglip-base-patch16-224` | Joint text + image embedding for cross-modal vault search |
| LLM | **Ollama** (`llama3:8b-instruct-q4_K_M` by default) | Local; swap any Ollama tag in config |
| Chunking | **MarkdownHeaderTextSplitter → recursive fallback** | Header breadcrumb survives as filterable metadata |
| Clustering | **UMAP + GaussianMixture** for RAPTOR | Standard recipe from the RAPTOR paper |
| Communities | **NetworkX Louvain** on the wikilink graph | Cheaper than entity-extraction GraphRAG |
| Agent | **LangGraph** | Explicit state machine for the router + CRAG loop |
| Web ingest | **trafilatura** (HTML) + **yt-dlp** / **youtube-transcript-api** | Markdown-shaped output preserves headings |
| Sync | **watchfiles** (Rust-backed) + SHA-256 sidecar SQLite | Designed for large vaults |
| Time parsing | **dateparser** | Natural-language temporal hints → date ranges |
| UI | **Streamlit** + **streamlit-agraph** | 4-tab shape, token streaming, reasoning panel |
| Eval | **RAGAS** + cheap citation/substring heuristics | LLM-as-judge metrics with CI-friendly fallback |
| Logging | **structlog** (JSON or console) | Optional Langfuse / OpenTelemetry export |
| Packaging | **uv** + **hatchling** + **ruff** + **mypy --strict** | Modern Python toolchain |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  UI: Streamlit (Chat / Mentor / Graph / Settings)                    │
│  CLI: typer  (chat, sync, reindex, doctor, stats, summarize, …)      │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Agent: LangGraph state machine                                      │
│    /slash overrides → router → (retrieve_kb | retrieve_web | direct) │
│    draft → critic → [revise+retry | finalize]                        │
│    optional speculative finalize (N parallel drafts + verifier)      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Retrieve                                                            │
│    [[Note]] pinning + [memory] recall + time filter                  │
│    multi-query / HyDE → hybrid (dense + sparse, server-side RRF)     │
│    cross-query RRF → wikilink 1-hop expansion → cross-encoder rerank │
│    recency-weighted score multiplier                                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌──────────────────┬─────────────▼─────────────┬───────────────────────┐
│ Qdrant collections                                                   │
│ ──────────────────────────────────────────────────────────────────── │
│  knowledge_base  │  leaf chunks + RAPTOR summaries + community       │
│                  │  summaries (single collection, level/source_type) │
│  chat_memory     │  Q/A pairs for cross-session recall               │
│  vault_images    │  SigLIP image embeddings                          │
│  akb_meta        │  index version stamp                              │
└──────────────────┴────────────────────────────┴───────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  State: SQLite (per-DB schema_version + migrations)                  │
│    ingest_state.db  · context_cache.db  · session_history.db         │
│    schedules.db     · WAL mode, autocommit per row                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Ingest: Obsidian-aware loader                                       │
│    frontmatter → metadata · tags → metadata · wikilinks → graph      │
│    embeds → recursive inline · header path → breadcrumb              │
│    secret scrub → contextualize (batched Ollama) → dedupe → upsert   │
│  Also: PDF (pymupdf4llm) · EPUB · TXT · Web (trafilatura) · YouTube  │
│        Images (SigLIP, opt-in)                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Installation

Requires Python 3.11+ and a running [Ollama](https://ollama.com).

```bash
# 1. Pull a local model
ollama pull llama3:8b-instruct-q4_K_M

# 2. Get the code
git clone https://github.com/Gitgud456/ai-knowledge-base.git akb
cd akb

# 3. Install (uv is recommended; pip works too)
uv venv && uv pip install -e ".[dev]"
# or:
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 4. Point at your vault
cp .env.example .env
# edit OBSIDIAN_VAULT_PATH to your vault root

# 5. Verify everything is reachable
akb doctor
```

Optional extras:

```bash
uv pip install -e ".[tracing]"    # Langfuse + OpenTelemetry
uv pip install -e ".[late]"       # ColBERT rerank + late chunking (RAGatouille)
# Opt-in feature deps (not in core install):
uv pip install umap-learn scikit-learn   # RAPTOR clustering
uv pip install transformers pillow       # SigLIP image search
```

> **First run downloads** BGE-M3 (~2.3 GB) and `bge-reranker-v2-m3` (~600 MB) from Hugging Face. They're cached after that. SigLIP adds ~700 MB if enabled.

## Quickstart

```bash
akb doctor               # preflight (Ollama, model cache, vault, disk)
akb info                 # show resolved config and paths
akb stats                # current index size, sources by type, drift

akb reindex --yes        # full one-time index of your vault
akb sync                 # incremental: only changed/added/deleted notes
akb sync --watch         # live: re-embed on file save (targeted, not full sweep)

akb chat                 # REPL — supports /search /web /cite /dry-run /help
akb serve                # Streamlit UI on localhost:8501
akb eval                 # RAGAS over tests/golden/golden_set.yaml
```

### Other CLI verbs

```bash
# Generate
akb summarize source:obsidian:projects/X.md     # map-reduce a source
akb summarize tag:security                       # map-reduce a tag slice

# Ingest external sources
akb ingest-url https://example.com/article
akb ingest-youtube dQw4w9WgXcQ

# Sessions
akb export-chat 42 --out ./out      # session → markdown w/ wikilinks

# Ops
akb backup                          # tar.gz of data/
akb restore data/backups/akb-backup-…tar.gz

# Scheduled queries (wire to OS cron)
akb schedule add --name weekly-review --cron "0 9 * * MON" \
    --query "what new questions did I write this week?" \
    --out "akb_chats/weekly-review.md"
akb schedule list
akb schedule run                    # run anything that's due, then exit

# Advanced features (opt-in via config flags)
akb raptor build / akb raptor delete
akb communities build / akb communities delete
akb images ingest / akb images search "the network topology diagram"
```

### Querying with filters (Python)

```python
from akb.retrieve.pipeline import retrieve

res = retrieve(
    "what did I write about ARP spoofing?",
    filter={
        "tags": ["security", "network"],
        "modified_at": {"gte": "2025-01-01T00:00:00Z"},
    },
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
AKB_AGENT__ENABLE_MEMORY=false akb chat
```

Anything nested uses `__` as the path separator (`AKB_<section>__<key>`).

### Enabling advanced features

In `configs/local.yaml`:

```yaml
retrieve:
  recency_weight: 0.5            # bias toward recent notes
  recency_half_life_days: 90

embed:
  binary_quantization: true      # ~32x smaller index (requires reindex)
  binary_oversampling: 2.0

raptor:
  enabled: true                  # then: akb raptor build
  max_levels: 3
  min_cluster_size: 5

communities:
  enabled: true                  # then: akb communities build

images:
  enabled: true                  # then: akb images ingest

speculative:
  enabled: true                  # parallel drafters in _finalize
  n_drafts: 3
```

## Repository layout

```
src/akb/
  config.py                   # pydantic-settings: default.yaml + local.yaml + env
  schemas.py                  # Document, Chunk, RetrievedChunk, Citation, Answer
  cli.py                      # `akb` typer commands
  cli_ops.py                  # doctor / stats / export-chat helpers
  ingest/
    obsidian_loader.py        # frontmatter + tags + wikilinks + embeds + casefold
    pdf_loader.py             # pymupdf4llm
    epub_loader.py  txt_loader.py
    chunkers.py               # MarkdownHeader → recursive, header_path metadata
    contextualizer.py         # batched JSON-array Ollama prompts + WAL cache
    scrubber.py               # secret detection + redact / block / warn policies
    dedupe.py                 # SHA + shingle Jaccard
    graph.py                  # VaultGraph (wikilinks)
    pipeline.py               # path → chunks (scrub → contextualize → dedupe)
    upsert.py                 # batched embed + qdrant.upsert
    sync.py                   # plan_sync(restrict_paths) + apply_sync, orphan-safe
    watcher.py                # watchfiles, targeted plan calls
    web_loader.py             # trafilatura
    youtube_loader.py         # transcript-api + yt-dlp
    image_loader.py           # SigLIP, opt-in
    raptor.py                 # UMAP+GMM hierarchical summaries, opt-in
    communities.py            # Louvain community summaries, opt-in
    late_chunking.py          # opt-in
  embed/
    providers.py              # BGE-M3
    multimodal.py             # SigLIP, opt-in
  store/
    qdrant_store.py           # singleton lock, idempotent point_id, FusionParams
    sqlite_state.py           # ingest_state.db
    migrations.py             # SQLite schema_version + Qdrant index stamp
  retrieve/
    query_transform.py        # multi-query + HyDE
    hybrid.py                 # cross-query RRF, supports date range filters
    graph_expand.py           # 1-hop wikilinks, baseline-scored to survive rerank
    rerank.py                 # bge-reranker-v2-m3 + recency multiplier
    time_filter.py            # dateparser → modified_at range
    colbert_rerank.py         # opt-in
    pipeline.py               # composes everything
  agents/
    tools.py                  # search_kb + search_web (errors never leak to LLM)
    memory.py                 # short-term window + summary
    memory_store.py           # cross-session Q/A memory (chat_memory collection)
    pinning.py                # [[Note]] → pinned context
    slash.py                  # / command parser
    graph.py                  # LangGraph: router → retrieve → draft → critic → finalize
    speculative.py            # parallel drafters + verifier, opt-in
    mentor.py                 # plan + lessons + Q/A
    summarize.py              # map-reduce source / tag summarisation
  sessions/db.py              # SQLite session/messages
  eval/
    golden.py                 # YAML loader
    ragas_runner.py           # RAGAS + cheap heuristics
  prompts/
    chat.py  mentor.py        # versioned, A/B-friendly
  ops/
    backup.py                 # tar.gz of data/, atomic restore
    schedules.py              # persisted cron + one-shot runner
  obs/
    logging.py                # structlog
    tracing.py                # optional Langfuse via OTel
  ui/app.py                   # Streamlit: streaming, reasoning panel, slash, pinning
tests/
  unit/                       # 152 tests (+integration), pure-python where possible
  integration/                # In-memory Qdrant + fake embedder end-to-end
  golden/golden_set.yaml      # evaluation set (seed yours here)
configs/
  default.yaml                # source of truth
  local.yaml.example
legacy/                       # original prototype, kept for reference
```

## Testing

```bash
pytest                                            # full suite (~13s)
pytest tests/unit/test_obsidian_loader.py -v      # one file
akb eval --no-ragas --json data/eval.json         # offline regression check
```

```
152 tests passing (30 → 152 across Tier 0-3)
  – Unit (most): pure-python, no heavy deps required
  – Integration: in-memory Qdrant + fake embedder end-to-end
  – Regression: chunk_id determinism, RRF off-by-one, critic loop bounds,
                Unicode link resolution, recency math, contextualizer
                JSON fallback, slash parser, scrubber patterns, raptor
                clustering guards, speculative verifier robustness, …
```

The unit suite runs without Qdrant, Ollama, or any ML model loaded — heavy components are mocked.

## How retrieval actually works

1. **`[[Note]]` pinning.** Wikilink mentions in the query are pre-resolved against `VaultGraph`; that note's chunks become `[pinned: Title]` blocks at the head of context.
2. **Time-aware filter.** `dateparser` extracts temporal hints; a `modified_at` range filter is attached to the retrieval request.
3. **Cross-session memory recall.** Top-k embeddings from the `chat_memory` collection are surfaced as `[memory · 2025-04-12 · score 0.87]` blocks.
4. **Query transforms.** Decomposed into 1-3 atomic sub-queries via Ollama JSON mode; optionally augmented with a HyDE hypothetical answer.
5. **Hybrid prefetch.** Each sub-query produces a dense + sparse vector pair; Qdrant fuses them server-side with RRF.
6. **Cross-query RRF.** Per-sub-query hit lists are merged client-side.
7. **Graph expansion.** Top hits' notes pull in 1-hop wikilink neighbours' chunks; graph chunks carry a baseline RRF score so they survive the reranker pool cut.
8. **Rerank.** `bge-reranker-v2-m3` re-scores the union; recency multiplier applied if enabled; top `top_k` survive.
9. **Synthesize.** Draft → CRAG-style critic. Critic can re-issue with a sharper query. Optionally routes through Speculative RAG (parallel drafters + verifier).
10. **Finalize.** Streaming tokens. Reasoning + citations + related-notes panel rendered in the UI.

## Why not just use [popular tool X]

| Tool | Why I didn't | What akb does differently |
|---|---|---|
| Smart Connections | JSON-on-disk vector store, dense-only, no rerank | Hybrid + reranker + contextual retrieval + RAPTOR + communities |
| Obsidian Copilot | Solid, but Plus tier and Orama-in-browser | Native filesystem, everything embedded locally, no telemetry |
| Khoj | Excellent but heavy (Postgres + Django) | Single-binary feel, embedded Qdrant, embedded SQLite |
| Quivr / AnythingLLM | General-purpose; wikilinks are dead text | First-class wikilink graph (expansion + community summaries) |
| Microsoft GraphRAG | Quality leader, but $$$ at index time (LLM-driven entity extraction) | Free local equivalent using *your* wikilink edges, no entity-extraction cost |

## Roadmap

Done across Tier 0-3:

- [x] Cross-session Q/A memory collection
- [x] Time-aware retrieval (dateparser → Qdrant date filter)
- [x] Recency-weighted reranking
- [x] Token streaming + reasoning panel + slash commands + `[[Note]]` pinning
- [x] Save-as-note, conversation export
- [x] Secret scrubber at ingest
- [x] Binary quantization (Qdrant)
- [x] Batched async contextualizer
- [x] Schema versioning + index version stamp
- [x] Backup / restore + scheduled queries
- [x] Web + YouTube ingest
- [x] Long-doc summarization (`akb summarize`)
- [x] RAPTOR hierarchical summary index
- [x] Wikilink community summaries (light Graph-RAG)
- [x] SigLIP cross-modal image search
- [x] Speculative RAG (parallel drafters + verifier)

Still open:

- [ ] DSPy MIPROv2 prompt compile (gated on building a real golden set)
- [ ] Tokenizer-offset late chunking (current uses uniform proportional spans)
- [ ] Integration tests against a real Qdrant + Ollama in CI
- [ ] Markdown-table-aware chunker (treat tables as atomic)
- [ ] Voice input/output (Whisper + Piper)
- [ ] Optional Graphiti temporal knowledge graph
- [ ] Pre-commit hook running the scrubber against staged diffs

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Papers and projects that directly inform akb:

- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic, Sept 2024
- [BGE-M3](https://arxiv.org/abs/2402.03216) — Chen et al., BAAI
- [CRAG](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
- [RAPTOR](https://arxiv.org/abs/2401.18059) — Sarthi et al., 2024
- [Late Chunking](https://arxiv.org/abs/2409.04701) — Günther et al., 2024
- [HyDE](https://arxiv.org/abs/2212.10496) — Gao et al., 2022
- [Speculative RAG](https://arxiv.org/abs/2407.08223) — Wang et al., 2024
- [SigLIP](https://arxiv.org/abs/2303.15343) — Zhai et al., 2023
- [LangGraph](https://github.com/langchain-ai/langgraph), [Qdrant](https://qdrant.tech/), [Ollama](https://ollama.com/), [trafilatura](https://github.com/adbar/trafilatura)
