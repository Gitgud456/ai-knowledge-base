"""Streamlit shell on top of the new ``akb`` package.

Tabs:
  * **Chat** — token streaming, "why this answer" reasoning expander, slash
    commands, ``[[Note]]`` sticky context, citations with one-click
    "save as note" and "export chat", a backlinks panel under each answer.
  * **Mentor** — plan + lessons + Q/A (uses ``MentorState.commit_reply``).
  * **Knowledge Graph** — wikilink-based, seeded from any vault note.
  * **Settings** — resolved config (read-only).

Run: ``akb serve`` (or ``streamlit run src/akb/ui/app.py``).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from akb.agents.graph import ChatAgent, StreamEvent
from akb.agents.mentor import continue_session, parse_plan, start_session
from akb.agents.slash import HELP_TEXT, parse as parse_slash
from akb.cli_ops import export_session
from akb.config import load_settings
from akb.ingest.graph import VaultGraph, build_graph
from akb.ingest.obsidian_loader import iter_vault
from akb.ingest.sync import apply_sync, plan_sync
from akb.obs.logging import configure_logging
from akb.sessions.db import (
    add_message_to_session,
    create_new_session,
    delete_session,
    get_session_history,
    init_history_db,
    list_sessions,
)
from akb.store.qdrant_store import get_store

configure_logging()
st.set_page_config(page_title="akb · AI Knowledge Base", layout="wide")
settings = load_settings()


@st.cache_resource
def _store_handle():
    return get_store()


@st.cache_resource
def _vault_graph() -> VaultGraph:
    """Cache the wikilink graph for the session. Refresh via the sidebar button."""
    docs = list(iter_vault())
    return build_graph(docs)


@st.cache_resource
def _agent(_graph_hash: int) -> ChatAgent:
    # ``_graph_hash`` participates in the cache key so a "refresh graph" click
    # in the sidebar gives us a fresh agent bound to the new graph.
    return ChatAgent(graph=_vault_graph(), store=_store_handle())


init_history_db()

ss = st.session_state
ss.setdefault("chat_session_id", None)
ss.setdefault("mentor_session_id", None)
ss.setdefault("mentor_state", None)
ss.setdefault("last_reasoning", None)
ss.setdefault("last_citations", [])
ss.setdefault("graph_epoch", 0)

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("akb")
    st.caption(f"vault: `{settings.paths.vault}`")
    st.caption(f"embed: `{settings.embed.model}`  ·  llm: `{settings.llm.local_model}`")

    st.subheader("Sessions")
    name = st.text_input("New session name", key="new_session_name")
    if st.button("Start chat session"):
        if name:
            ss.chat_session_id = create_new_session(name)
            st.rerun()
    if st.button("Start mentor session"):
        if name:
            ss.mentor_session_id = create_new_session(f"Mentor: {name}")
            ss.mentor_state = None
            st.rerun()

    saved = list_sessions()
    if saved:
        opts = {f"{s['name']} (id={s['id']})": int(s["id"]) for s in saved}
        chosen = st.selectbox("Load / delete / export", [""] + list(opts.keys()))
        cols = st.columns(3)
        with cols[0]:
            if st.button("Load") and chosen:
                sid = opts[chosen]
                if "Mentor:" in chosen:
                    ss.mentor_session_id = sid
                    ss.chat_session_id = None
                    ss.mentor_state = None
                else:
                    ss.chat_session_id = sid
                    ss.mentor_session_id = None
                st.rerun()
        with cols[1]:
            if st.button("Export") and chosen:
                try:
                    out = export_session(opts[chosen])
                    st.success(f"wrote {out}")
                except Exception as e:
                    st.error(str(e))
        with cols[2]:
            if st.button("Delete", type="primary") and chosen:
                delete_session(opts[chosen])
                st.rerun()

    st.divider()
    st.subheader("Vault sync")
    if st.button("Plan sync"):
        plan = plan_sync()
        st.info(
            f"added: {len(plan.added)}  ·  changed: {len(plan.changed)}  ·  deleted: {len(plan.deleted)}"
        )
        ss["_pending_plan"] = plan
    if "_pending_plan" in ss and st.button("Apply sync"):
        with st.spinner("Applying sync…"):
            res = apply_sync(ss["_pending_plan"])
        st.success(f"+{res['upserts']} chunks, -{res['deletes']} sources")
        ss.pop("_pending_plan", None)
        ss.graph_epoch += 1  # invalidate cached vault graph
        _vault_graph.clear()  # type: ignore[attr-defined]
        _agent.clear()  # type: ignore[attr-defined]

    st.divider()
    st.subheader("Index")
    if st.button("Index stats"):
        try:
            st.info(f"points in collection: {_store_handle().count()}")
        except Exception as e:
            st.error(f"qdrant error: {e}")
    if st.button("Refresh vault graph"):
        _vault_graph.clear()  # type: ignore[attr-defined]
        _agent.clear()  # type: ignore[attr-defined]
        ss.graph_epoch += 1
        st.success("graph + agent caches cleared")

# --- tabs ------------------------------------------------------------------
tab_chat, tab_mentor, tab_graph, tab_settings = st.tabs(
    ["Chat", "Mentor", "Knowledge Graph", "Settings"]
)


def _save_snippet(citation_source: str, snippet: str) -> Path:
    target = settings.paths.vault / "akb_snippets"
    target.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9-_]+", "-", citation_source)[:60] or "snippet"
    fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{slug}.md"
    body = (
        "---\n"
        f"source: {citation_source}\n"
        "tags: [akb/snippet]\n"
        "---\n\n"
        f"From [[{Path(citation_source).stem if citation_source.endswith('.md') else citation_source}]]:\n\n"
        f"> {snippet}\n"
    )
    out = target / fname
    out.write_text(body, encoding="utf-8")
    return out


# --------- Chat
with tab_chat:
    st.header("Chat")
    st.caption(
        "Stream answer · `/help` for slash commands · "
        "`[[Note]]` to pin a note's chunks into context"
    )

    if ss.chat_session_id is None:
        st.info("Start or load a chat session from the sidebar.")
    else:
        history = get_session_history(ss.chat_session_id)
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        # Reasoning panel for the most recent answer
        if ss.last_reasoning:
            with st.expander("why this answer (last turn)"):
                st.json(ss.last_reasoning)
        # Related-notes panel (1-hop wikilinks of cited sources)
        if ss.last_citations:
            graph = _vault_graph()
            related: set[str] = set()
            for c in ss.last_citations[:5]:
                related |= graph.neighbours(c.source_id, hops=1)
            related -= {c.source_id for c in ss.last_citations}
            if related:
                with st.expander(f"related notes ({len(related)})"):
                    for sid in sorted(related)[:20]:
                        st.markdown(f"- `{sid}`")

        prompt = st.chat_input("Ask, or /help…")
        if prompt:
            sc = parse_slash(prompt)
            if sc.show_help:
                with st.chat_message("assistant"):
                    st.markdown(f"```\n{HELP_TEXT}\n```")
            else:
                add_message_to_session(ss.chat_session_id, "user", prompt)
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    holder = st.empty()
                    buf = ""
                    final_state = None
                    answer = None
                    for evt in _agent(ss.graph_epoch).stream_answer(
                        sc.query or prompt,
                        history=history,
                        force_path=sc.force_path,
                        cite_only=sc.cite_only,
                        dry_run=sc.dry_run,
                    ):
                        if evt.kind == "state":
                            final_state = evt.state
                        elif evt.kind == "token":
                            buf += evt.token
                            holder.markdown(buf + "▌")
                        elif evt.kind == "done":
                            answer = evt.answer
                            holder.markdown(buf or (answer.text if answer else ""))
                    if answer is not None:
                        # Stash reasoning + citations for the next render
                        ss.last_reasoning = {
                            "path": final_state.get("path") if final_state else None,
                            "iterations": final_state.get("iterations") if final_state else 0,
                            "critic_verdict": final_state.get("critic_verdict") if final_state else None,
                            "critic_notes": final_state.get("critic_notes") if final_state else "",
                            "improved_query": final_state.get("improved_query") if final_state else "",
                            "pinned_titles": final_state.get("pinned_titles") if final_state else [],
                            "cite_only": bool(final_state.get("cite_only")) if final_state else False,
                            "dry_run": bool(final_state.get("dry_run")) if final_state else False,
                        }
                        ss.last_citations = answer.citations or []
                        if answer.citations:
                            with st.expander(f"sources ({len(answer.citations)})"):
                                for i, c in enumerate(answer.citations[:8]):
                                    cc = st.columns([5, 1])
                                    with cc[0]:
                                        st.markdown(
                                            f"- `{c.source_id}` (score={c.score:.3f})\n  > {c.snippet}"
                                        )
                                    with cc[1]:
                                        if st.button("save", key=f"save-{i}-{c.chunk_id}"):
                                            out = _save_snippet(c.source_id, c.snippet)
                                            st.success(out.name)
                        add_message_to_session(
                            ss.chat_session_id, "assistant", answer.text or buf
                        )
                st.rerun()

# --------- Mentor
with tab_mentor:
    st.header("Mentor")
    if ss.mentor_session_id is None:
        topic = st.text_input("What do you want to master?")
        if st.button("Start mentoring") and topic:
            ss.mentor_session_id = create_new_session(f"Mentor: {topic}")
            buf = ""
            with st.spinner("Building your plan…"):
                stream, _cit, ms = start_session(topic)
                for chunk in stream:
                    buf += chunk
            ms.plan = parse_plan(buf)
            ms.commit_reply(buf)
            ss.mentor_state = ms
            add_message_to_session(ss.mentor_session_id, "user", f"Master: {topic}")
            add_message_to_session(ss.mentor_session_id, "assistant", buf)
            st.rerun()
    else:
        for m in get_session_history(ss.mentor_session_id):
            with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else "user"):
                st.markdown(m["content"])
        if ss.mentor_state is not None and ss.mentor_state.plan:
            st.markdown("**Plan**")
            for i, t in enumerate(ss.mentor_state.plan):
                pointer = "▶ " if i == ss.mentor_state.current_index else "  "
                st.markdown(f"{pointer}{i + 1}. {t}")
        msg = st.chat_input("Ask, or say 'next'…")
        if msg and ss.mentor_state is not None:
            add_message_to_session(ss.mentor_session_id, "user", msg)
            buf = ""
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Mentor is thinking…"):
                    stream, _cit, new_state = continue_session(ss.mentor_state, msg)
                    placeholder = st.empty()
                    for chunk in stream:
                        buf += chunk
                        placeholder.markdown(buf + "▌")
                    placeholder.markdown(buf)
            ss.mentor_state = new_state.commit_reply(buf)
            add_message_to_session(ss.mentor_session_id, "assistant", buf)
            st.rerun()

# --------- Knowledge Graph
with tab_graph:
    st.header("Knowledge Graph (wikilink-based)")
    from streamlit_agraph import Config, Edge, Node, agraph

    seed = st.text_input("Seed note title (case-insensitive)")
    hops = st.slider("hops", 1, 3, 1)
    if st.button("Build graph") and seed:
        with st.spinner("Walking vault…"):
            g = _vault_graph()
        import unicodedata

        key = unicodedata.normalize("NFC", seed.strip()).casefold()
        target_sid = g.title_to_source.get(key)
        if not target_sid:
            st.warning("Note not found by title or alias.")
        else:
            nbrs = g.neighbours(target_sid, hops=hops) | {target_sid}
            sub_nx = g.to_networkx().subgraph(nbrs)
            nodes = [
                Node(id=n, label=n.split(":")[-1], color="#FF4B4B" if n == target_sid else "#9CCC65")
                for n in sub_nx.nodes
            ]
            edges = [Edge(source=u, target=v) for u, v in sub_nx.edges]
            agraph(
                nodes=nodes,
                edges=edges,
                config=Config(width=1100, height=720, directed=True, physics=True),
            )

# --------- Settings
with tab_settings:
    st.header("Settings (resolved)")
    st.caption("Edit `configs/local.yaml` to override defaults, or set `AKB_*` env vars.")
    st.json(
        {
            "embed": settings.embed.model_dump(),
            "retrieve": settings.retrieve.model_dump(),
            "agent": settings.agent.model_dump(),
            "ingest": settings.ingest.model_dump(),
            "llm": settings.llm.model_dump(),
        }
    )
