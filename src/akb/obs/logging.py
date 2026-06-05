"""structlog configuration.

One ``configure_logging()`` call wires up structlog with:
  * a JSON renderer (`obs.json_logs: true`) or a colourised console renderer
  * stdlib-bridge so libraries that use ``logging`` flow through the same
    processor pipeline
  * context binding: ``trace_id``, ``query_id``, ``session_id`` propagated
    automatically using ``contextvars``

Call once at process start (CLI, Streamlit, tests). Repeat calls are no-ops.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from akb.config import ObsConfig, load_settings

_CONFIGURED = False


def configure_logging(cfg: ObsConfig | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = cfg or load_settings().obs

    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*pre_chain, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name or "akb")


def bind(**fields: Any) -> None:
    """Add fields to the current contextvars-scoped logging context."""
    structlog.contextvars.bind_contextvars(**fields)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
