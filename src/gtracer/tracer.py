"""Generic span-based tracing framework for LangChain/LangGraph agents.

Produces structured JSONL spans via Python's standard logging at a custom
TRACE level (25).  Any log handler that writes to stdout/CloudWatch/file
will capture them.

Two APIs:
  span()                         context manager — for code with clear scope
  open_span() / close_span()     open/close pair — for LangChain callbacks
                                 where start and end live in separate methods

Quick start:
    from gtracer import tracer, configure

    configure(truncation_limit=50_000)   # optional, this is the default
    tracer.start_trace(session_id)

    with tracer.span("run", tags={"user_id": "u123"}):
        with tracer.span("agent", attrs={"agent": "main"}) as span:
            result = await agent.ainvoke(...)
            span.set("output", result)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Custom TRACE log level (25 — between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------

_TRACE_LEVEL: int = 25
logging.addLevelName(_TRACE_LEVEL, "TRACE")


def _trace(self: logging.Logger, msg: object, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(_TRACE_LEVEL):
        self._log(_TRACE_LEVEL, msg, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.trace = _trace  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_truncation_limit: int = 50_000


def configure(truncation_limit: int = 50_000) -> None:
    """Configure global tracer settings.

    Call once at application startup, before any spans are opened.

    Args:
        truncation_limit: Maximum characters for string content fields in span
            attrs (``delta``, ``response``, ``result``).  Longer strings are
            truncated with ``...`` appended.  Default: 50,000.
    """
    global _truncation_limit  # noqa: PLW0603
    _truncation_limit = truncation_limit


def _trunc_limit() -> int:
    return _truncation_limit


# ---------------------------------------------------------------------------
# Span taxonomy — enforced at runtime (warning only, never raises)
# ---------------------------------------------------------------------------

VALID_CHILDREN: dict[str | None, set[str]] = {
    None:        {"run"},
    "run":       {"agent"},
    "agent":     {"llm_call"},
    "llm_call":  {"tool_call"},
    "tool_call": {"agent"},   # sub-agent nested inside a tool
}

# ---------------------------------------------------------------------------
# ContextVars — async-safe, propagate through LangGraph tasks automatically
# ---------------------------------------------------------------------------

_span_id:    ContextVar[str | None] = ContextVar("gtracer_span_id",    default=None)
_span_name:  ContextVar[str | None] = ContextVar("gtracer_span_name",  default=None)
_trace_id:   ContextVar[str | None] = ContextVar("gtracer_trace_id",   default=None)
_tags:       ContextVar[dict]       = ContextVar("gtracer_tags",       default={})
_agent_name: ContextVar[str]        = ContextVar("gtracer_agent_name", default="unknown")


# ---------------------------------------------------------------------------
# SpanContext — one instance per open span
# ---------------------------------------------------------------------------

@dataclass
class SpanContext:
    """Mutable container for a span in flight.

    Start-time attrs are passed via ``open_span()`` / ``span()``; end-time
    attrs are accumulated via ``.set()`` during execution and flushed into
    ``span.end`` / ``span.error``.
    """

    span_id:        str
    name:           str
    parent_span_id: str | None
    trace_id:       str | None
    tags:           dict[str, Any]
    attrs:          dict[str, Any] = field(default_factory=dict)
    _start:         float          = field(default_factory=time.monotonic, repr=False)
    _failed:        bool           = field(default=False, repr=False)
    _fail_reason:   str            = field(default="",   repr=False)

    def set(self, key: str, value: Any) -> None:
        """Accumulate an end-time attribute.  Flushed on ``span.end`` / ``span.error``."""
        self.attrs[key] = value

    def fail(self, reason: str = "") -> None:
        """Mark as a business-level failure.

        The span still closes cleanly (``span.end``) but with ``status: error``.
        Use this when the operation completed but the outcome is bad
        (e.g. LLM refusal, fixer exhausted retries).
        Python exceptions that propagate out of ``span()`` automatically use
        ``span.error`` instead — do not call ``fail()`` in that case.
        """
        self._failed = True
        self._fail_reason = reason


# ---------------------------------------------------------------------------
# LangChain message serialisation helper
# ---------------------------------------------------------------------------

def serialise_lc_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain ``BaseMessage`` instances to JSON-serialisable dicts.

    Truncates string content to the configured ``truncation_limit`` chars.
    Keeps tool_use / tool_call blocks intact (they are always small).
    """
    out: list[dict[str, Any]] = []
    limit = _trunc_limit()
    for msg in messages:
        msg_type = getattr(msg, "type", "unknown")
        content  = getattr(msg, "content", "")

        if isinstance(content, str):
            if len(content) > limit:
                content = content[:limit] + " ...[truncated]"
        elif isinstance(content, list):
            content = [_truncate_content_block(b) for b in content]

        entry: dict[str, Any] = {"type": msg_type, "content": content}

        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id

        out.append(entry)
    return out


def _truncate_content_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    if block.get("type") == "text":
        text = block.get("text", "")
        limit = _trunc_limit()
        if len(text) > limit:
            return {**block, "text": text[:limit] + " ...[truncated]"}
    return block


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Emit structured span events via ``logger.trace()``.

    Import the module-level singleton ``tracer`` rather than instantiating
    this class directly.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def start_trace(self, trace_id: str) -> None:
        """Set the current trace ID.  Call once at the top of each run."""
        _trace_id.set(trace_id)

    # ------------------------------------------------------------------ #
    # Context manager API — for code with a clear lexical scope            #
    # ------------------------------------------------------------------ #

    @contextmanager
    def span(
        self,
        name:           str,
        attrs:          dict[str, Any] | None = None,
        tags:           dict[str, Any] | None = None,
        parent_span_id: str | None            = None,
    ) -> Generator[SpanContext, None, None]:
        """Open a span, yield ``SpanContext``, emit ``span.end`` or ``span.error`` on exit.

        Updates ContextVars so nested spans automatically resolve their parent.
        Pass ``parent_span_id`` to override the ContextVar-derived parent (e.g. for
        causal tool_call → llm_call linking across asyncio task boundaries).
        """
        ctx = self._make_span(name, attrs, tags, parent_span_id=parent_span_id)

        span_tok      = _span_id.set(ctx.span_id)
        name_tok      = _span_name.set(ctx.name)
        tags_tok      = _tags.set(ctx.tags)
        agent_name_tok = (
            _agent_name.set((attrs or {}).get("agent", "unknown"))
            if name == "agent" else None
        )
        try:
            yield ctx
            self._emit_end(ctx)
        except Exception as exc:
            self._emit_error(ctx, exc)
            raise
        finally:
            _span_id.reset(span_tok)
            _span_name.reset(name_tok)
            _tags.reset(tags_tok)
            if agent_name_tok is not None:
                _agent_name.reset(agent_name_tok)

    # ------------------------------------------------------------------ #
    # Open/close API — for LangChain callbacks                             #
    # ------------------------------------------------------------------ #

    def open_span(
        self,
        name:           str,
        attrs:          dict[str, Any] | None = None,
        tags:           dict[str, Any] | None = None,
        parent_span_id: str | None            = None,
    ) -> SpanContext:
        """Open a span without a context manager.

        Captures ``parent_span_id`` from the current ContextVar state at call time.
        Does NOT update ContextVars — the caller must call ``close_span()`` or
        ``error_span()`` later.  Use ``span()`` for code that nests further spans inside.
        """
        return self._make_span(name, attrs, tags, parent_span_id=parent_span_id)

    def close_span(
        self,
        ctx:       SpanContext,
        end_attrs: dict[str, Any] | None = None,
    ) -> None:
        """Close a span opened with ``open_span()``."""
        if end_attrs:
            ctx.attrs.update(end_attrs)
        self._emit_end(ctx)

    def error_span(self, ctx: SpanContext, exc: Exception) -> None:
        """Mark a span opened with ``open_span()`` as failed due to an exception."""
        self._emit_error(ctx, exc)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_span(
        self,
        name:           str,
        attrs:          dict[str, Any] | None,
        tags:           dict[str, Any] | None,
        parent_span_id: str | None = None,
    ) -> SpanContext:
        if parent_span_id is not None:
            resolved_parent_id   = parent_span_id
            resolved_parent_name = "llm_call"
        else:
            resolved_parent_id   = _span_id.get()
            resolved_parent_name = _span_name.get()

        allowed = VALID_CHILDREN.get(resolved_parent_name, set())
        if name not in allowed:
            self._log.warning(
                "gtracer: illegal span nesting — parent=%r → child=%r (allowed: %s)",
                resolved_parent_name, name, sorted(allowed) or "none",
            )

        ctx = SpanContext(
            span_id        = uuid.uuid4().hex[:8],
            name           = name,
            parent_span_id = resolved_parent_id,
            trace_id       = _trace_id.get(),
            tags           = {**_tags.get(), **(tags or {})},
            attrs          = dict(attrs or {}),
        )
        self._emit("span.start", ctx)
        return ctx

    def _emit_end(self, ctx: SpanContext) -> None:
        duration_ms = int((time.monotonic() - ctx._start) * 1000)
        status = "error" if ctx._failed else "ok"
        if ctx._failed and ctx._fail_reason:
            ctx.attrs["fail_reason"] = ctx._fail_reason
        self._emit("span.end", ctx, duration_ms=duration_ms, status=status)

    def _emit_error(self, ctx: SpanContext, exc: Exception) -> None:
        duration_ms = int((time.monotonic() - ctx._start) * 1000)
        ctx.attrs["error"]      = str(exc)
        ctx.attrs["error_type"] = type(exc).__name__
        self._emit("span.error", ctx, duration_ms=duration_ms, status="error")

    def _emit(
        self,
        event:       str,
        ctx:         SpanContext,
        duration_ms: int | None = None,
        status:      str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event":          event,
            "span_name":      ctx.name,
            "trace_id":       ctx.trace_id,
            "span_id":        ctx.span_id,
            "parent_span_id": ctx.parent_span_id,
            **ctx.tags,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if status is not None:
            payload["status"] = status
        if ctx.attrs:
            payload["attrs"] = self._truncate_attrs(ctx.attrs)

        try:
            self._log.trace(  # type: ignore[attr-defined]
                "[%s] %s", ctx.name, event, extra=payload,
            )
        except Exception as exc:  # noqa: BLE001
            import sys
            print(
                f"[GTRACER EMIT ERROR] event={event} span={ctx.span_id} "
                f"name={ctx.name} error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _truncate_attrs(self, attrs: dict[str, Any]) -> dict[str, Any]:
        limit = _trunc_limit()
        out: dict[str, Any] = {}
        for k, v in attrs.items():
            if k in ("delta", "response"):
                out[k] = self._trunc_message_list(v, limit)
            elif k == "result":
                out[k] = _trunc_str(v, limit)
            elif isinstance(v, str) and len(v) > limit:
                out[k] = v[:limit] + " ...[truncated]"
            else:
                out[k] = v
        return out

    def _trunc_message_list(self, messages: Any, limit: int) -> Any:
        if not isinstance(messages, list):
            return messages
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > limit:
                    msg = {**msg, "content": content[:limit] + " ...[truncated]"}
            result.append(msg)
        return result


def _trunc_str(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + " ...[truncated]"
    return value


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tracer = Tracer(logging.getLogger("gtracer"))
