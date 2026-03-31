"""Generic span-based tracing framework for LangChain/LangGraph agents.

Produces structured JSONL spans via Python's standard logging at a custom
TRACE level (5).  Any log handler that writes to stdout/CloudWatch/file
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
            span.set_attr("output", result)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import uuid
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generator, Mapping

from gtracer.logger import _TRACE_LEVEL


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_truncation_limit: int = 50_000


def configure(
    truncation_limit: int | None = None,
    enabled: bool | None = None,
    log_to_file: bool | None = None,
    extra_children: dict[str, set[str]] | None = None,
) -> None:
    """Configure global tracer settings.

    Call once at application startup, before any spans are opened.

    Args:
        truncation_limit: Maximum characters for string content fields in span
            attrs (``delta``, ``response``, ``result``).  Longer strings are
            truncated with ``...`` appended.  Default: 50,000.  ``None``
            (default) leaves the current setting unchanged.
        enabled: Enable/disable stdout span output.  Overrides the
            ``GTRACER_ENABLED`` env var.  ``None`` (default) leaves
            the current setting unchanged.
        log_to_file: Enable/disable local ``.jsonl`` file logging.  Overrides
            the ``GTRACER_LOG_TO_FILE`` env var.  ``None`` (default) leaves
            the current setting unchanged.
        extra_children: Extend the span taxonomy.  Keys are parent span names,
            values are sets of allowed child names.  Merged with the built-in
            rules (additive, never removes existing rules).  Example::

                configure(extra_children={
                    "agent": {"retrieval", "embedding"},
                    "retrieval": {"llm_call"},
                })
    """
    global _truncation_limit  # noqa: PLW0603
    if truncation_limit is not None:
        _truncation_limit = truncation_limit

    if enabled is not None or log_to_file is not None:
        from gtracer.logger import _configure, _current_enabled, _current_log_to_file
        resolved_enabled = enabled if enabled is not None else _current_enabled
        resolved_log_to_file = log_to_file if log_to_file is not None else _current_log_to_file
        _configure(resolved_enabled, resolved_log_to_file)

    if extra_children is not None:
        for parent, children in extra_children.items():
            existing = _valid_children.get(parent, frozenset())
            _valid_children[parent] = existing | frozenset(children)


def _trunc_limit() -> int:
    return _truncation_limit


# ---------------------------------------------------------------------------
# Span taxonomy — enforced at runtime (warning only, never raises)
# ---------------------------------------------------------------------------

_valid_children: dict[str | None, frozenset[str]] = {
    None:        frozenset({"run"}),
    "run":       frozenset({"agent"}),
    "agent":     frozenset({"llm_call"}),
    "llm_call":  frozenset({"tool_call"}),
    "tool_call": frozenset({"agent"}),   # sub-agent nested inside a tool
}
VALID_CHILDREN: Mapping[str | None, frozenset[str]] = MappingProxyType(_valid_children)

_RESERVED_SPAN_KEYS: frozenset[str] = frozenset({
    "event", "span_name", "trace_id", "span_id", "parent_span_id",
    "duration_ms", "status", "attrs",
})

# ---------------------------------------------------------------------------
# ContextVars — async-safe, propagate through LangGraph tasks automatically
# ---------------------------------------------------------------------------

_span_id:    ContextVar[str | None] = ContextVar("gtracer_span_id",    default=None)
_span_name:  ContextVar[str | None] = ContextVar("gtracer_span_name",  default=None)
_trace_id:   ContextVar[str | None] = ContextVar("gtracer_trace_id",   default=None)
_tags:       ContextVar[Mapping[str, Any]] = ContextVar("gtracer_tags", default=MappingProxyType({}))
_agent_name: ContextVar[str]        = ContextVar("gtracer_agent_name", default="unknown")
_trace_metadata: ContextVar[Mapping[str, Any]] = ContextVar("gtracer_trace_metadata", default=MappingProxyType({}))


# ---------------------------------------------------------------------------
# SpanContext — one instance per open span
# ---------------------------------------------------------------------------

@dataclass
class SpanContext:
    """Mutable container for an in-flight span.

    One instance is created per open span and lives until ``span.end`` or
    ``span.error`` is emitted.  Do not share a SpanContext across asyncio
    Tasks — ``attrs`` is not protected by a lock.

    Attributes:
        span_id: 32-character hex UUID, unique per span.
        name: Span type (``"run"``, ``"agent"``, ``"llm_call"``, ``"tool_call"``).
        parent_span_id: ID of the parent span, or ``None`` on the root span.
        trace_id: Session identifier set by ``tracer.start_trace()``.
        tags: Tags inherited at span open time, promoted to top-level JSON.
        attrs: Mutable attribute dict.  Accumulate end-time data via ``.set()``.
    """

    span_id:        str
    name:           str
    parent_span_id: str | None
    trace_id:       str | None
    tags:           dict[str, Any]
    attrs:          dict[str, Any] = field(default_factory=dict)
    _start:          float      = field(default_factory=time.monotonic, init=False, repr=False)
    _failed:         bool       = field(default=False,                  init=False, repr=False)
    _fail_reason:    str        = field(default="",                     init=False, repr=False)
    _end_attr_keys:  set[str]   = field(default_factory=set,            init=False, repr=False)

    def set_attr(self, key: str, value: Any) -> None:
        """Accumulate an end-time attribute.

        Stores the value in ``attrs`` and flushes it into the ``span.end``
        or ``span.error`` payload when the span closes.

        Args:
            key: Attribute name (e.g. ``"result"``, ``"output_type"``).
            value: Any JSON-serialisable value.
        """
        self.attrs[key] = value
        self._end_attr_keys.add(key)

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

def serialize_lc_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain ``BaseMessage`` instances to JSON-serializable dicts.

    Truncates ``str`` content to the configured ``truncation_limit`` chars.
    Keeps tool_use / tool_call content blocks intact (they are always small).

    Args:
        messages: List of LangChain ``BaseMessage`` objects.

    Returns:
        List of dicts with keys ``type``, ``content``, and optionally
        ``tool_call_id``.
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
    """Emit structured span events as JSONL via ``logger.trace()``.

    Import the module-level singleton rather than instantiating this class
    directly::

        from gtracer import tracer

    Two span APIs:

    - ``tracer.span()`` — context manager.  Updates ContextVars so nested
      spans automatically resolve their parent.  Use for code you control.

    - ``tracer.open_span()`` / ``tracer.close_span()`` — open/close pair.
      Does **not** update ContextVars.  Use inside LangChain callbacks where
      start and end happen in separate methods.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def start_trace(self, trace_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Set the current trace/session ID and optional trace-level metadata.

        Call once per invocation in the entry-point coroutine or thread,
        **before** opening any spans.  All spans opened afterwards inherit
        this value in their ``trace_id`` field.

        Args:
            trace_id: Unique identifier for the session or request
                      (e.g. a session UUID or request ID).
            metadata: Key/value pairs promoted to top-level JSON fields on
                      every span in this trace.  Lower priority than span-level
                      tags (span tags override metadata on key collision).

        Example::

            tracer.start_trace(session_id, metadata={"user_id": "u42", "env": "prod"})
            with tracer.span("run", tags={"session_id": session_id}):
                ...
        """
        _trace_id.set(trace_id)
        _trace_metadata.set(MappingProxyType(metadata) if metadata else MappingProxyType({}))

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

        Args:
            name: Span type — one of ``"run"``, ``"agent"``, ``"llm_call"``,
                  ``"tool_call"``.  Must satisfy ``VALID_CHILDREN`` nesting rules.
            attrs: Start-time attributes merged into the span payload.  End-time
                   data is added via ``SpanContext.set()`` during the ``with`` block.
            tags: Key/value pairs promoted to top-level JSON fields.  Inherited
                  by all descendant spans.
            parent_span_id: Explicit parent span ID.  Overrides ContextVar-based
                            resolution.  Required when calling across asyncio Task
                            boundaries (e.g. inside a LangGraph tool node).

        Yields:
            SpanContext: Mutable container for the open span.

        Example::

            with tracer.span("tool_call",
                             attrs={"tool": "search", "input": {"q": query}},
                             parent_span_id=llm_parent) as span:
                result = await search(query)
                span.set_attr("result", result)
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
        Does **not** update ContextVars — call ``close_span()`` or ``error_span()``
        later.  Use ``span()`` for code you control; use this only inside LangChain
        callbacks where start and end live in separate methods.

        Args:
            name: Span type — one of ``"run"``, ``"agent"``, ``"llm_call"``,
                  ``"tool_call"``.
            attrs: Start-time attributes.
            tags: Key/value pairs promoted to top-level JSON fields.
            parent_span_id: Explicit parent override.  Defaults to ContextVar state.

        Returns:
            SpanContext: The open span.  Pass to ``close_span()`` or ``error_span()``.
        """
        return self._make_span(name, attrs, tags, parent_span_id=parent_span_id)

    def close_span(
        self,
        ctx:       SpanContext,
        end_attrs: dict[str, Any] | None = None,
    ) -> None:
        """Close a span opened with ``open_span()``.

        Args:
            ctx: The SpanContext returned by ``open_span()``.
            end_attrs: Additional attributes to merge into the span before closing.
                       Equivalent to calling ``ctx.set(k, v)`` for each key.
        """
        if end_attrs:
            ctx.attrs.update(end_attrs)
            ctx._end_attr_keys.update(end_attrs.keys())
        self._emit_end(ctx)

    def error_span(self, ctx: SpanContext, exc: Exception) -> None:
        """Mark a span opened with ``open_span()`` as failed due to an exception.

        Emits ``span.error`` with ``error``, ``error_type``, and ``duration_ms``.

        Args:
            ctx: The SpanContext returned by ``open_span()``.
            exc: The exception that caused the failure.
        """
        self._emit_error(ctx, exc)

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def current_span_id() -> str | None:
        """Return the current active span ID (from ContextVar)."""
        return _span_id.get()

    @staticmethod
    def current_trace_id() -> str | None:
        """Return the current trace/session ID (from ContextVar)."""
        return _trace_id.get()

    # ------------------------------------------------------------------ #
    # Tool decorator                                                       #
    # ------------------------------------------------------------------ #

    def tool(self, name: str | None = None) -> Any:
        """Decorator that instruments a function as a traced ``tool_call`` span.

        Automatically resolves the parent ``llm_call`` span, captures input
        arguments, and records the return value as the span result.

        Can be used with or without parentheses::

            @tool
            @tracer.tool
            async def search(query: str) -> str: ...

            @tool
            @tracer.tool()
            async def search(query: str) -> str: ...

            @tool
            @tracer.tool(name="custom_name")
            async def search(query: str) -> str: ...

        Args:
            name: Tool name for the span.  Defaults to the function name.
        """
        if callable(name):
            return self.tool()(name)

        def decorator(func: Any) -> Any:
            try:
                from langchain_core.tools import BaseTool
                if isinstance(func, BaseTool):
                    raise TypeError(
                        "gtracer: @tracer.tool() received a LangChain Tool object — "
                        "@tool must be stacked above @tracer.tool(), not below it"
                    )
            except ImportError:
                pass
            tool_name = name or func.__name__
            try:
                sig = inspect.signature(func)  # cache once at decoration time
            except (ValueError, TypeError):
                sig = None

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                from gtracer.callbacks import tracing_handler
                llm_parent = tracing_handler.last_llm_span()
                input_data = _bind_args(args, kwargs, sig)
                with self.span("tool_call",
                               attrs={"tool": tool_name, "input": input_data},
                               parent_span_id=llm_parent) as span:
                    result = await func(*args, **kwargs)
                    span.set_attr("result", result)
                    return result

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                from gtracer.callbacks import tracing_handler
                llm_parent = tracing_handler.last_llm_span()
                input_data = _bind_args(args, kwargs, sig)
                with self.span("tool_call",
                               attrs={"tool": tool_name, "input": input_data},
                               parent_span_id=llm_parent) as span:
                    result = func(*args, **kwargs)
                    span.set_attr("result", result)
                    return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

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
            # Explicit override — caller is bridging across task boundaries.
            # Skip hierarchy validation: we don't know the parent's span type.
            resolved_parent_id   = parent_span_id
            resolved_parent_name = _span_name.get()  # for ContextVar reset only
        else:
            resolved_parent_id   = _span_id.get()
            resolved_parent_name = _span_name.get()
            allowed = VALID_CHILDREN.get(resolved_parent_name, set())
            if name not in allowed:
                self._log.warning(
                    "gtracer: illegal span nesting — parent=%r → child=%r (allowed: %s)",
                    resolved_parent_name, name, sorted(allowed) or "none",
                )

        if name == "run" and _trace_id.get() is None:
            warnings.warn(
                "gtracer: span('run') opened without start_trace() — "
                "trace_id will be None on all spans",
                stacklevel=3,
            )

        ctx = SpanContext(
            span_id        = uuid.uuid4().hex,
            name           = name,
            parent_span_id = resolved_parent_id,
            trace_id       = _trace_id.get(),
            tags           = {**_trace_metadata.get(), **_tags.get(), **(tags or {})},
            attrs          = dict(attrs or {}),
        )
        # Check tag collisions once per span, not on every emit.
        collisions = _RESERVED_SPAN_KEYS & ctx.tags.keys()
        if collisions:
            self._log.warning(
                "gtracer: tag keys %s collide with built-in span fields and will be ignored",
                sorted(collisions),
            )
        self._emit("span.start", ctx)
        return ctx

    def _emit_end(self, ctx: SpanContext) -> None:
        duration_ms = int((time.monotonic() - ctx._start) * 1000)
        status = "error" if ctx._failed else "ok"
        # Only emit attrs set/updated after span.start to avoid double-emitting
        # large start-time payloads (e.g. delta) while preserving updated values
        # (e.g. model resolved from response metadata).
        end_attrs = {k: v for k, v in ctx.attrs.items() if k in ctx._end_attr_keys}
        if ctx._failed and ctx._fail_reason:
            end_attrs["fail_reason"] = ctx._fail_reason
        self._emit("span.end", ctx, duration_ms=duration_ms, status=status, attrs=end_attrs)

    def _emit_error(self, ctx: SpanContext, exc: Exception) -> None:
        duration_ms = int((time.monotonic() - ctx._start) * 1000)
        end_attrs = {k: v for k, v in ctx.attrs.items() if k in ctx._end_attr_keys}
        end_attrs["error"]      = str(exc)
        end_attrs["error_type"] = type(exc).__name__
        self._emit("span.error", ctx, duration_ms=duration_ms, status="error", attrs=end_attrs)

    def _emit(
        self,
        event:       str,
        ctx:         SpanContext,
        duration_ms: int | None = None,
        status:      str | None = None,
        attrs:       dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            **ctx.tags,
            "event":          event,
            "span_name":      ctx.name,
            "trace_id":       ctx.trace_id,
            "span_id":        ctx.span_id,
            "parent_span_id": ctx.parent_span_id,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if status is not None:
            payload["status"] = status
        effective_attrs = attrs if attrs is not None else ctx.attrs
        if effective_attrs:
            payload["attrs"] = self._truncate_attrs(effective_attrs)

        try:
            self._log.log(
                _TRACE_LEVEL, "[%s] %s", ctx.name, event, extra=payload,
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


def _bind_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    sig: inspect.Signature | None,
) -> dict[str, Any]:
    """Best-effort binding of positional + keyword args to parameter names."""
    if sig is None:
        return dict(kwargs) if kwargs else {}
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (ValueError, TypeError):
        return dict(kwargs) if kwargs else {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tracer = Tracer(logging.getLogger("gtracer"))
