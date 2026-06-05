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
    """Interactive REPL chat with the LangGraph agent."""
    from akb.agents.graph import ChatAgent

    agent = ChatAgent()
    history: list[dict[str, str]] = []
    console.print("[bold cyan]akb chat[/bold cyan] — Ctrl-C or `:q` to exit\n")
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
        history.append({"role": "user", "content": user})
        ans = agent.invoke(user, history=history)
        history.append({"role": "assistant", "content": ans.text})
        console.print(f"\n[bold magenta]akb[/bold magenta] ▸ {ans.text}\n")
        if ans.citations:
            console.print("[dim]sources:[/dim]")
            for c in ans.citations[:5]:
                console.print(f"  [dim]- {c.source_id} (score={c.score:.3f})[/dim]")


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


if __name__ == "__main__":
    app()
