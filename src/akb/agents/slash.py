"""Slash-command parsing for chat input.

The Streamlit chat input and ``akb chat`` REPL accept lines like::

    /web what's the latest on X
    /cite ARP spoofing
    /search how do I tcpdump on Linux
    /dry-run summarize my March notes
    /help

The result of :func:`parse` is a :class:`SlashCommand` value that the agent
honors via the :class:`~akb.agents.graph.ChatState` knobs:

  * ``force_path``  — bypass the LLM router and go straight to ``retrieve_kb``
    / ``retrieve_web`` / ``direct``.
  * ``cite_only``   — return citations + a one-line preamble; skip synthesis +
    critic loop entirely. Fastest path, useful for "what notes do I have on X".
  * ``dry_run``     — run the router + retrieve, return what *would* be sent
    to the LLM. No tokens consumed by the synthesizer.
"""

from __future__ import annotations

from dataclasses import dataclass


HELP_TEXT = """\
Slash commands:

  /search <query>   force the vault-retrieval path (skip the router)
  /web <query>      force the web-search path
  /cite <query>     show top-cited chunks only, skip synthesis (fast)
  /dry-run <query>  run retrieval, show what would be sent to the LLM, stop
  /help             this help
"""


@dataclass(frozen=True)
class SlashCommand:
    cmd: str | None         # 'search', 'web', 'cite', 'dry_run', 'help', or None
    query: str              # the remainder (or the original line if no command)
    force_path: str | None  # 'retrieve' | 'web' | 'direct' | None
    cite_only: bool = False
    dry_run: bool = False
    show_help: bool = False


def parse(line: str) -> SlashCommand:
    """Parse a single chat-input line."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return SlashCommand(cmd=None, query=line, force_path=None)
    head, _, rest = stripped[1:].partition(" ")
    cmd = head.lower()
    query = rest.strip()
    if cmd in ("h", "help", "?"):
        return SlashCommand(
            cmd="help", query="", force_path=None, show_help=True
        )
    if cmd in ("search", "kb", "vault"):
        return SlashCommand(cmd="search", query=query, force_path="retrieve")
    if cmd == "web":
        return SlashCommand(cmd="web", query=query, force_path="web")
    if cmd in ("cite", "citations"):
        return SlashCommand(cmd="cite", query=query, force_path="retrieve", cite_only=True)
    if cmd in ("dry-run", "dry_run", "dry"):
        return SlashCommand(cmd="dry_run", query=query, force_path="retrieve", dry_run=True)
    # Unknown slash — treat as plain text so user isn't punished for typos.
    return SlashCommand(cmd=None, query=line, force_path=None)
