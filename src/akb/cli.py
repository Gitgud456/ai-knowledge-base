"""`akb` CLI entrypoint.

    akb ingest [PATH]      — ingest a single file or a directory into the index
    akb sync               — incremental sync of the Obsidian vault (Phase 6 wires watchfiles)
    akb reindex            — drop + rebuild the index from scratch
    akb chat               — interactive REPL chat with the agent
    akb eval               — run the RAGAS golden-set evaluation
    akb serve              — launch the Streamlit UI
    akb info               — print resolved config and index stats
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from akb import __version__
from akb.cli_ops import doctor_checks, export_session, gather_stats
from akb.config import load_settings
from akb.obs.logging import configure_logging

configure_logging()

app = typer.Typer(
    name="akb",
    help="Personal AI knowledge base over Obsidian + PDFs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _phase_stub(phase: str, what: str) -> None:
    console.print(
        f"[yellow]⏳ Not wired yet — lands in Phase {phase}.[/yellow] {what}",
    )
    raise typer.Exit(code=2)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit.", is_eager=True),
) -> None:
    if version:
        console.print(f"akb {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def info() -> None:
    """Show resolved configuration and basic index stats."""
    s = load_settings()
    t = Table(title="akb configuration")
    t.add_column("key", style="cyan")
    t.add_column("value", style="white")
    t.add_row("version", __version__)
    t.add_row("vault", str(s.paths.vault))
    t.add_row("data_dir", str(s.paths.data_dir))
    t.add_row("qdrant_dir", str(s.paths.qdrant_dir))
    t.add_row("embed.model", s.embed.model)
    t.add_row("llm.local_model", s.llm.local_model)
    t.add_row("llm.deep_provider", s.llm.deep_provider)
    t.add_row("retrieve.top_k", str(s.retrieve.top_k))
    t.add_row("retrieve.use_reranker", str(s.retrieve.use_reranker))
    t.add_row("agent.framework", s.agent.framework)
    t.add_row("ingest.contextual_retrieval", str(s.ingest.contextual_retrieval))
    console.print(t)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="File or directory to ingest."),
    force: bool = typer.Option(False, "--force", help="Re-embed even if content hash matches."),
) -> None:
    """Ingest a single file or directory into the index."""
    from akb.ingest.pipeline import chunks_for_path
    from akb.ingest.upsert import upsert_chunks

    _ = force  # honored by Phase 6 incremental sync; full ingest is always force
    with console.status(f"[cyan]Loading + chunking[/cyan] {path}"):
        chunks = chunks_for_path(path)
    console.print(f"[green]✓[/green] {len(chunks)} chunks ready")
    with console.status("[cyan]Embedding + upserting (BGE-M3 → Qdrant)[/cyan]"):
        n = upsert_chunks(chunks)
    console.print(f"[green]✓[/green] upserted {n} points into Qdrant")


@app.command()
def sync(
    watch: bool = typer.Option(False, "--watch", help="Run watchfiles observer forever."),
) -> None:
    """Incremental sync of the Obsidian vault (added/changed/deleted notes only)."""
    from akb.ingest.sync import plan_sync, apply_sync
    from akb.ingest.watcher import run_watcher_forever

    if watch:
        console.print("[cyan]Watching vault for changes (Ctrl-C to stop)…[/cyan]")
        run_watcher_forever()
        return

    with console.status("[cyan]Planning sync (hashing changed notes)[/cyan]"):
        plan = plan_sync()
    console.print(
        f"[green]✓[/green] plan: [bold]{len(plan.added)}[/bold] added, "
        f"[bold]{len(plan.changed)}[/bold] changed, [bold]{len(plan.deleted)}[/bold] deleted"
    )
    if plan.total() == 0:
        console.print("[dim]nothing to do.[/dim]")
        return
    with console.status("[cyan]Applying sync (embed + upsert affected docs only)[/cyan]"):
        result = apply_sync(plan)
    console.print(
        f"[green]✓[/green] +{result['upserts']} chunks upserted, "
        f"{result['deletes']} sources deleted"
    )


@app.command()
def reindex(
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Nuke + rebuild the index from scratch. Destructive."""
    if not yes:
        confirm = typer.confirm("This will delete the existing index. Continue?")
        if not confirm:
            raise typer.Abort()
    from akb.ingest.pipeline import chunks_for_path
    from akb.ingest.upsert import upsert_chunks
    from akb.store.qdrant_store import get_store

    store = get_store()
    with console.status("[red]Dropping existing collection[/red]"):
        store.recreate()
    console.print("[green]✓[/green] collection recreated")
    with console.status("[cyan]Walking vault → chunks[/cyan]"):
        chunks = chunks_for_path(None)  # None → use configured vault
    console.print(f"[green]✓[/green] {len(chunks)} chunks ready")
    with console.status("[cyan]Embedding + upserting[/cyan]"):
        n = upsert_chunks(chunks)
    console.print(f"[green]✓[/green] upserted {n} points")


@app.command()
def chat() -> None:
    """Interactive REPL chat with the LangGraph agent.

    Slash commands work inline: ``/search foo``, ``/web foo``, ``/cite foo``,
    ``/dry-run foo``, ``/help``. Wikilinks like ``[[Note]] what's left?`` pin
    that note's chunks into the agent's context.
    """
    from akb.agents.graph import ChatAgent
    from akb.agents.slash import HELP_TEXT, parse as parse_slash

    agent = ChatAgent()
    history: list[dict[str, str]] = []
    console.print("[bold cyan]akb chat[/bold cyan] — Ctrl-C or `:q` to exit; `/help` for slash commands\n")
    while True:
        try:
            user = console.input("[bold green]you[/bold green] ▸ ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return
        if user.strip() in {":q", ":quit", "exit"}:
            return
        if not user.strip():
            continue
        sc = parse_slash(user)
        if sc.show_help:
            console.print(HELP_TEXT)
            continue
        history.append({"role": "user", "content": user})

        # Stream tokens for the actual answer
        console.print("\n[bold magenta]akb[/bold magenta] ▸ ", end="")
        final_text = ""
        from akb.agents.graph import StreamEvent  # avoid top-level circular

        for evt in agent.stream_answer(
            sc.query or user,
            history=history,
            force_path=sc.force_path,
            cite_only=sc.cite_only,
            dry_run=sc.dry_run,
        ):
            if evt.kind == "token":
                console.print(evt.token, end="")
                final_text += evt.token
            elif evt.kind == "done" and evt.answer is not None:
                history.append({"role": "assistant", "content": evt.answer.text or final_text})
                if evt.answer.citations:
                    console.print("\n\n[dim]sources:[/dim]")
                    for c in evt.answer.citations[:5]:
                        console.print(f"  [dim]- {c.source_id} (score={c.score:.3f})[/dim]")
        console.print()


@app.command(name="eval")
def eval_(
    golden: Path | None = typer.Option(None, "--golden", help="Override golden set path."),
    no_ragas: bool = typer.Option(False, "--no-ragas", help="Skip RAGAS metrics (cheap heuristics only)."),
    json_out: Path | None = typer.Option(None, "--json", help="Write the full report to a JSON file."),
) -> None:
    """Run the evaluation against the golden set."""
    import json as _json

    from akb.eval.ragas_runner import report_to_dict, run_eval

    with console.status("[cyan]Running golden-set eval (this may take a few minutes)…[/cyan]"):
        report = run_eval(golden_path=golden, use_ragas=not no_ragas)

    t = Table(title=f"akb eval — {report.n_items} items in {report.elapsed_s:.1f}s")
    t.add_column("metric", style="cyan")
    t.add_column("value", style="white")
    t.add_row("citation_hit_rate", f"{report.citation_hit_rate:.3f}")
    t.add_row("substring_hit_rate", f"{report.substring_hit_rate:.3f}")
    for k, v in report.ragas_means.items():
        t.add_row(f"ragas.{k}", f"{v:.3f}")
    console.print(t)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] full report written to {json_out}")


@app.command()
def serve(
    port: int = typer.Option(8501, "--port"),
) -> None:
    """Launch the Streamlit UI."""
    import subprocess
    import sys

    ui_path = Path(__file__).parent / "ui" / "app.py"
    if not ui_path.exists():
        _phase_stub("0-5", "Streamlit UI lands progressively. Use the legacy app.py for now.")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path), "--server.port", str(port)],
        check=False,
    )


@app.command()
def doctor() -> None:
    """Preflight checks: Ollama reachable, model weights cached, vault + paths OK."""
    report = doctor_checks()
    t = Table(title="akb doctor")
    t.add_column("check", style="cyan")
    t.add_column("status", justify="center")
    t.add_column("detail", style="white")
    for c in report.checks:
        status = "[green]ok[/green]" if c.ok else "[red]fail[/red]"
        t.add_row(c.name, status, c.detail)
    console.print(t)
    if not report.healthy:
        raise typer.Exit(code=1)


@app.command()
def stats() -> None:
    """Index size, sources by type, cache rows, ingest_state vs Qdrant drift."""
    s = gather_stats()
    t = Table(title="akb stats")
    t.add_column("metric", style="cyan")
    t.add_column("value", style="white")
    t.add_row("qdrant.points", str(s.qdrant_points))
    t.add_row("qdrant.sources", str(s.qdrant_sources))
    t.add_row("ingest_state.sources", str(s.ingest_state_sources))
    t.add_row("ingest_state.chunks", str(s.ingest_state_chunks))
    t.add_row("drift (qdrant vs state)", str(s.drift))
    t.add_row("context_cache.rows", str(s.context_cache_rows))
    t.add_row("sessions", str(s.session_count))
    for st, n in sorted(s.by_source_type.items()):
        t.add_row(f"by_source_type:{st}", str(n))
    console.print(t)


@app.command(name="export-chat")
def export_chat(
    session_id: int = typer.Argument(..., help="Session ID (see `akb info` / Streamlit sidebar)."),
    out_dir: Path = typer.Option(None, "--out", help="Override output directory."),
) -> None:
    """Export a saved chat session as a markdown note in {vault}/akb_chats/."""
    try:
        path = export_session(session_id, out_dir=out_dir)
    except ValueError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[green]✓[/green] wrote {path}")


@app.command()
def summarize(
    target: str = typer.Argument(
        ...,
        help="What to summarise: 'source:<id>', 'tag:<name>', 'type:<source_type>', or a bare source_id.",
    ),
    json_out: Path = typer.Option(None, "--json", help="Write the full report (notes + brief) as JSON."),
) -> None:
    """Map-reduce summarize a source or a tag-scoped slice of the vault."""
    from akb.agents.summarize import summarize_dispatch

    with console.status(f"[cyan]Summarising {target}…[/cyan]"):
        res = summarize_dispatch(target)
    console.print(f"\n[bold]Scope:[/bold] {res.scope}\n")
    console.print(res.text)
    console.print(f"\n[dim]{len(res.chunks)} chunk(s) consumed[/dim]")
    if json_out:
        import json as _json

        payload = {
            "scope": res.scope,
            "model": res.model,
            "text": res.text,
            "map_notes": res.map_notes,
            "n_chunks": len(res.chunks),
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {json_out}")


@app.command(name="ingest-url")
def ingest_url(
    url: str = typer.Argument(..., help="Article URL (HTML)."),
) -> None:
    """Ingest a single web URL via trafilatura."""
    from akb.ingest.pipeline import chunks_for
    from akb.ingest.upsert import upsert_chunks
    from akb.ingest.web_loader import load_url

    with console.status(f"[cyan]Fetching {url}…[/cyan]"):
        doc = load_url(url)
    console.print(f"[green]✓[/green] {doc.title}")
    with console.status("[cyan]Chunking + embedding + upserting…[/cyan]"):
        chunks = chunks_for([doc])
        n = upsert_chunks(chunks)
    console.print(f"[green]✓[/green] {n} chunks upserted")


@app.command(name="ingest-youtube")
def ingest_youtube(
    target: str = typer.Argument(..., help="YouTube URL or 11-character video id."),
) -> None:
    """Ingest a YouTube video's transcript."""
    from akb.ingest.pipeline import chunks_for
    from akb.ingest.upsert import upsert_chunks
    from akb.ingest.youtube_loader import load_youtube

    with console.status(f"[cyan]Fetching transcript for {target}…[/cyan]"):
        doc = load_youtube(target)
    console.print(f"[green]✓[/green] {doc.title}")
    with console.status("[cyan]Chunking + embedding + upserting…[/cyan]"):
        chunks = chunks_for([doc])
        n = upsert_chunks(chunks)
    console.print(f"[green]✓[/green] {n} chunks upserted")


@app.command()
def backup(
    out_dir: Path = typer.Option(None, "--out", help="Override backup directory."),
) -> None:
    """Tar+gzip the data/ directory (Qdrant + SQLite + caches)."""
    from akb.ops.backup import backup as do_backup

    info = do_backup(out_dir=out_dir)
    console.print(
        f"[green]✓[/green] {info.path}  ({info.size_bytes / (1024 * 1024):.1f} MB)"
    )


@app.command()
def restore(
    archive: Path = typer.Argument(..., exists=True, help="Path to a .tar.gz backup archive."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Restore data/ from a backup archive. The live data dir is moved aside."""
    from akb.ops.backup import restore as do_restore

    if not yes:
        confirm = typer.confirm("This will replace the live data/ directory. Continue?")
        if not confirm:
            raise typer.Abort()
    aside = do_restore(archive)
    console.print(f"[green]✓[/green] restored from {archive}; previous data at {aside}")


schedule_app = typer.Typer(help="Manage scheduled queries.")
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("add")
def schedule_add(
    name: str = typer.Option(..., "--name", help="Unique handle."),
    cron: str = typer.Option(..., "--cron", help="Cron expression (5 fields)."),
    query: str = typer.Option(..., "--query"),
    out: str = typer.Option(..., "--out", help="Output markdown file (vault-relative or absolute)."),
) -> None:
    """Add a recurring query."""
    from akb.ops.schedules import add as do_add

    s = do_add(name=name, cron=cron, query=query, out_path=out)
    console.print(f"[green]✓[/green] schedule #{s.id} added")


@schedule_app.command("list")
def schedule_list() -> None:
    from akb.ops.schedules import list_all

    items = list_all()
    if not items:
        console.print("[dim]no schedules.[/dim]")
        return
    t = Table(title="schedules")
    t.add_column("id", style="cyan")
    t.add_column("name")
    t.add_column("cron")
    t.add_column("query")
    t.add_column("out")
    t.add_column("last run")
    for s in items:
        t.add_row(str(s.id), s.name, s.cron, s.query[:40], s.out_path, s.last_run or "—")
    console.print(t)


@schedule_app.command("delete")
def schedule_delete(name_or_id: str = typer.Argument(...)) -> None:
    from akb.ops.schedules import delete

    if delete(name_or_id):
        console.print("[green]✓[/green] removed")
    else:
        console.print(f"[yellow]no schedule matched {name_or_id!r}[/yellow]")


@schedule_app.command("run")
def schedule_run() -> None:
    """Run every schedule that's currently due, then exit. Wire to OS cron."""
    from akb.ops.schedules import run_due

    res = run_due()
    console.print(f"[green]✓[/green] ran {res['ran']}, skipped {res['skipped']}")


# --- Tier 3 opt-in features --------------------------------------------------

raptor_app = typer.Typer(help="RAPTOR hierarchical summary index (opt-in).")
app.add_typer(raptor_app, name="raptor")


@raptor_app.command("build")
def raptor_build() -> None:
    """Cluster leaf chunks, summarise each cluster, repeat to build the tree."""
    from akb.ingest.raptor import build_tree

    with console.status("[cyan]Building RAPTOR tree (may take a while)…[/cyan]"):
        stats = build_tree()
    t = Table(title="RAPTOR")
    t.add_column("metric", style="cyan")
    t.add_column("value")
    t.add_row("levels", str(stats.levels))
    t.add_row("summaries_written", str(stats.summaries_written))
    t.add_row("failed_summaries", str(stats.failed_summaries))
    for i, n in enumerate(stats.per_level, 1):
        t.add_row(f"level {i}", str(n))
    console.print(t)


@raptor_app.command("delete")
def raptor_delete() -> None:
    """Drop every RAPTOR summary node from the main collection."""
    from akb.ingest.raptor import delete_tree

    n = delete_tree()
    console.print(f"[green]✓[/green] removed {n} summary chunks")


communities_app = typer.Typer(help="Wikilink community summaries (light Graph-RAG, opt-in).")
app.add_typer(communities_app, name="communities")


@communities_app.command("build")
def communities_build() -> None:
    """Run Louvain on the wikilink graph and summarise each large community."""
    from akb.ingest.communities import build_communities

    with console.status("[cyan]Building community summaries…[/cyan]"):
        stats = build_communities()
    t = Table(title="Communities")
    t.add_column("metric", style="cyan")
    t.add_column("value")
    t.add_row("found", str(stats.communities_found))
    t.add_row("summarised", str(stats.summarised))
    t.add_row("failed", str(stats.failed))
    if stats.per_community_size:
        t.add_row(
            "avg size",
            f"{sum(stats.per_community_size) / len(stats.per_community_size):.1f}",
        )
    console.print(t)


@communities_app.command("delete")
def communities_delete() -> None:
    """Drop every community summary chunk from the main collection."""
    from akb.ingest.communities import delete_communities

    n = delete_communities()
    console.print(f"[green]✓[/green] removed {n} community summaries")


images_app = typer.Typer(help="SigLIP image search across vault images (opt-in).")
app.add_typer(images_app, name="images")


@images_app.command("ingest")
def images_ingest() -> None:
    """Embed every image referenced by a vault note into the image collection."""
    from akb.ingest.image_loader import ingest_images

    with console.status("[cyan]Discovering + embedding images…[/cyan]"):
        n = ingest_images()
    console.print(f"[green]✓[/green] {n} images upserted")


@images_app.command("search")
def images_search(
    query: str = typer.Argument(..., help="Free-text query."),
    top_k: int = typer.Option(8, "--k"),
) -> None:
    """Cross-modal text → image search."""
    from akb.ingest.image_loader import search_images

    hits = search_images(query, top_k=top_k)
    if not hits:
        console.print("[dim]no hits.[/dim]")
        return
    t = Table(title=f"image search: {query!r}")
    t.add_column("score", style="cyan")
    t.add_column("image")
    t.add_column("note")
    for h in hits:
        t.add_row(f"{h.score:.3f}", h.image_path, h.note_path)
    console.print(t)


if __name__ == "__main__":
    app()
