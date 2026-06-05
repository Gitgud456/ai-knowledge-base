"""Streamlit shell on top of the new ``akb`` package.

Same 4-tab shape as the legacy UI (Chat / Mentor / Knowledge Graph / Settings),
but every internal call now goes through:
  * ``akb.agents.graph.ChatAgent`` (LangGraph router + CRAG)
  * ``akb.agents.mentor`` (plan + lesson + QA)
  * ``akb.ingest.sync`` (incremental, hash-keyed)
  * ``akb.store.qdrant_store`` (Qdrant embedded, hybrid)

Run: ``akb serve`` (or ``streamlit run src/akb/ui/app.py``).
"""

from __future__ import annotations

import streamlit as st

from akb.agents.graph import ChatAgent
from akb.agents.mentor import continue_session, parse_plan, start_session
from akb.config import load_settings
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


@st.cache_resource
def _agent() -> ChatAgent:
    return ChatAgent()


@st.cache_resource
def _store_handle():
    return get_store()


init_history_db()
settings = load_settings()

# --- session-state defaults -----------------------------------------------
ss = st.session_state
ss.setdefault("chat_session_id", None)
ss.setdefault("mentor_session_id", None)
ss.setdefault("mentor_state", None)

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
        chosen = st.selectbox("Load / delete", [""] + list(opts.keys()))
        col_a, col_b = st.columns(2)
        with col_a:
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
        with col_b:
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

    st.divider()
    st.subheader("Index")
    if st.button("Index stats"):
        try:
            st.info(f"points in collection: {_store_handle().count()}")
        except Exception as e:
            st.error(f"qdrant error: {e}")

# --- tabs ------------------------------------------------------------------
tab_chat, tab_mentor, tab_graph, tab_settings = st.tabs(
    ["Chat", "Mentor", "Knowledge Graph", "Settings"]
)

# --------- Chat
with tab_chat:
    st.header("Chat (LangGraph router + CRAG)")
    if ss.chat_session_id is None:
        st.info("Start or load a chat session from the sidebar.")
    else:
        history = get_session_history(ss.chat_session_id)
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        prompt = st.chat_input("Ask your knowledge base…")
        if prompt:
            add_message_to_session(ss.chat_session_id, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    ans = _agent().invoke(prompt, history=history)
                st.markdown(ans.text)
                if ans.citations:
                    with st.expander("sources"):
                        for c in ans.citations[:8]:
                            st.markdown(f"- `{c.source_id}` (score={c.score:.3f})\n  > {c.snippet}")
            add_message_to_session(ss.chat_session_id, "assistant", ans.text)
            st.rerun()

# --------- Mentor
with tab_mentor:
    st.header("Mentor (plan + lessons + QA)")
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
            ms.commit_reply(buf)  # keep history so the next turn has continuity
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

# --------- Knowledge Graph (real wikilink graph from the vault)
with tab_graph:
    st.header("Knowledge Graph (wikilink-based)")
    from streamlit_agraph import Config, Edge, Node, agraph

    from akb.ingest.graph import build_graph
    from akb.ingest.obsidian_loader import iter_vault

    seed = st.text_input("Seed note title (case-insensitive)")
    hops = st.slider("hops", 1, 3, 1)
    if st.button("Build graph") and seed:
        with st.spinner("Walking vault…"):
            docs = list(iter_vault())
            g = build_graph(docs)
        target_sid = g.title_to_source.get(seed.strip().lower())
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

# --------- Settings (read-only display; edits via configs/local.yaml)
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
