"""Optional Langfuse tracing via OpenTelemetry.

This file is intentionally a thin bootstrap. If `langfuse` and the OTel SDK
aren't installed (they're in the ``[tracing]`` optional-deps), every public
helper here is a no-op — no import errors, no warnings, no startup cost.

Three things instrumented:
  * Retrieval pipeline (`retrieve.pipeline.retrieve`)
  * Reranker call
  * LLM calls (ollama.chat / generate)

Use :func:`span` as a context manager or :func:`traced` as a decorator.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from akb.config import load_settings
from akb.obs.logging import get_logger

T = TypeVar("T")
_log = get_logger(__name__)

_tracer: Any | None = None
_initialized = False


def _try_init() -> None:
    global _tracer, _initialized
    if _initialized:
        return
    _initialized = True

    cfg = load_settings().obs
    if not cfg.enable_tracing:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "akb"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("akb")
        _log.info("tracing.enabled", host=cfg.langfuse_host)
    except Exception as e:
        _log.warning("tracing.disabled", reason=str(e))
        _tracer = None


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    _try_init()
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as sp:  # type: ignore[union-attr]
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v)
            except Exception:
                pass
        yield


def traced(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        def inner(*args: Any, **kwargs: Any) -> T:
            with span(name or fn.__qualname__):
                return fn(*args, **kwargs)

        return inner

    return deco
